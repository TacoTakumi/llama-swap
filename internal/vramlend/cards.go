package vramlend

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"time"

	"github.com/mostlygeek/llama-swap/internal/logmon"
)

// DefaultStateTimeout bounds each of llama-swap's own /vram/state reads.
const DefaultStateTimeout = 35 * time.Second

// ServeFunc serves req through the named model's process only when the model
// is ready, and reports whether it did. It must never start, queue or swap a
// model; router.LocalRouter.ServeIfReady is the one implementation.
type ServeFunc func(modelID string, w http.ResponseWriter, req *http.Request) bool

// State is an engine's /vram/state answer. Amounts are in MiB.
type State struct {
	GPUs   []GPUState `json:"gpus"`
	Leases []Lease    `json:"leases"`

	// Run is the run token of the process run that answered, set by
	// Cards.ReadAll. It is not part of the engine's answer.
	Run string `json:"-"`
}

type GPUState struct {
	GPU      string `json:"gpu"`
	Lendable int64  `json:"lendable"`
	Lent     int64  `json:"lent"`
}

type Lease struct {
	ID  string `json:"id"`
	GPU string `json:"gpu"`
	MiB int64  `json:"mib"`
}

var errNotReady = errors.New("model is not ready")

// Cards keeps the cards each lend model uses, for routing a lend to the models
// that hold the card. A list belongs to one process run: it is read when the
// model becomes ready and dropped when it leaves the ready state, so a model
// that restarts on other cards routes by its new list.
//
// Every read goes through the ServeFunc, so nothing here can start a model.
// A nil *Cards holds no lend models; its methods are safe to call.
type Cards struct {
	// Timeout bounds each state read. Set it before the first use.
	Timeout time.Duration

	ctx    context.Context
	models []string
	serve  ServeFunc
	logger *logmon.Monitor

	mu   sync.Mutex
	runs map[string]*cardRun
	// tokens holds the run token of each model's current process run. See
	// Run for how a token is tied to one run.
	tokens map[string]string
}

// cardRun is one state read for one process run. cards is written before
// done closes and is read only after.
type cardRun struct {
	done  chan struct{}
	cards []string
}

// NewCards tracks the cards of models, the config keys of the models that
// lend VRAM. Lookups return them in the order given. Reads end when ctx does.
func NewCards(ctx context.Context, models []string, serve ServeFunc, logger *logmon.Monitor) *Cards {
	return &Cards{
		Timeout: DefaultStateTimeout,
		ctx:     ctx,
		models:  slices.Clone(models),
		serve:   serve,
		logger:  logger,
		runs:    make(map[string]*cardRun),
		tokens:  make(map[string]string),
	}
}

// Ready starts a background read of the model's cards for its new process
// run, replacing any list from before, and makes sure the run has a run
// token. It returns at once, so it is safe to call from the event dispatcher.
func (c *Cards) Ready(modelID string) {
	if c == nil || !slices.Contains(c.models, modelID) {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.startLocked(modelID)
	c.tokenLocked(modelID)
}

// Drop forgets the model's cards and run token. Call it when the model
// leaves the ready state; the cards of its next run are read when it is
// ready again, and the next run gets a new token.
func (c *Cards) Drop(modelID string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.runs, modelID)
	delete(c.tokens, modelID)
}

// Run returns the run token of the model's current process run, choosing one
// if the run has none yet. Lease ids carry it, so an id from an earlier run
// never names a lease of this one. It returns "" for a model that does not
// lend.
//
// A token lives from its first use until the next Drop. Drop runs on every
// state that is not ready, and a process run always passes through one
// (starting) before it is ready, so a token that exists was chosen in the
// current run. That is why Ready keeps a token it finds: an on-demand lend
// or state read can reach a new run before its ready event reaches Ready,
// and the leases lent in that window must keep their ids.
func (c *Cards) Run(modelID string) string {
	if c == nil || !slices.Contains(c.models, modelID) {
		return ""
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.tokenLocked(modelID)
}

// IsRun reports whether run is the token of the model's current process
// run. It never chooses a token: a model with none has lent nothing yet.
func (c *Cards) IsRun(modelID, run string) bool {
	if c == nil || run == "" {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.tokens[modelID] == run
}

// tokenLocked returns the model's run token, choosing one if there is none.
// c.mu must be held.
func (c *Cards) tokenLocked(modelID string) string {
	if tok, ok := c.tokens[modelID]; ok {
		return tok
	}
	tok := newRunToken()
	c.tokens[modelID] = tok
	return tok
}

// newRunToken returns 8 random lowercase hex characters, which never hold
// the ":" that separates the parts of a lease id.
func newRunToken() string {
	b := make([]byte, 4)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// Holders returns every lend model that uses card, a normalized PCI address,
// in the order the models were given to NewCards.
//
// A model with no list yet is read on demand, and a lookup that finds a read
// in flight waits for it rather than starting another. A model that is not
// ready answers no list at once, and a read is never waited on past Timeout.
// Holders returns nil when no model holds card or ctx ends first.
func (c *Cards) Holders(ctx context.Context, card string) []string {
	if c == nil {
		return nil
	}

	c.mu.Lock()
	runs := make([]*cardRun, len(c.models))
	for i, model := range c.models {
		r, ok := c.runs[model]
		if !ok {
			r = c.startLocked(model)
		}
		runs[i] = r
	}
	c.mu.Unlock()

	var holders []string
	for i, r := range runs {
		select {
		case <-r.done:
		case <-ctx.Done():
			return nil
		}
		if slices.Contains(r.cards, card) {
			holders = append(holders, c.models[i])
		}
	}
	return holders
}

// ReadAll reads the state of every lend model in parallel, each bounded by
// Timeout. A model that is not ready, fails or runs out of time is left out,
// and so is one whose run token changed during the read, since its leases
// cannot be tied to one run. Each state's Run is the token of the run that
// answered.
func (c *Cards) ReadAll(ctx context.Context) map[string]State {
	states := make(map[string]State)
	if c == nil {
		return states
	}

	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, model := range c.models {
		wg.Go(func() {
			// The token is taken before the read, so a restart during the
			// read cannot give the old run's leases the new run's token.
			run := c.Run(model)
			st, err := c.readState(ctx, model)
			if err != nil {
				c.logReadError(model, err)
				return
			}
			if !c.IsRun(model, run) {
				return
			}
			st.Run = run
			mu.Lock()
			states[model] = st
			mu.Unlock()
		})
	}
	wg.Wait()
	return states
}

// startLocked starts a read of the model's cards and records it as the
// model's current run. c.mu must be held.
func (c *Cards) startLocked(modelID string) *cardRun {
	r := &cardRun{done: make(chan struct{})}
	c.runs[modelID] = r

	go func() {
		defer close(r.done)
		st, err := c.readState(c.ctx, modelID)
		if err == nil {
			r.cards, err = normalizeCards(st)
		}
		if err == nil {
			return
		}
		c.logReadError(modelID, err)

		// A failed read must not stand for the whole run: the next lookup
		// reads again. A newer run that replaced this one is left alone.
		c.mu.Lock()
		if c.runs[modelID] == r {
			delete(c.runs, modelID)
		}
		c.mu.Unlock()
	}()
	return r
}

// readState reads one engine's /vram/state within Timeout.
//
// The read runs on its own goroutine so a process that does not honor the
// request's context still cannot hold the caller past the timeout. Its
// recorder is touched by nothing else once the caller has gone.
func (c *Cards) readState(ctx context.Context, modelID string) (State, error) {
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()

	type result struct {
		st  State
		err error
	}
	ch := make(chan result, 1)
	go func() {
		st, err := c.fetchState(ctx, modelID)
		ch <- result{st, err}
	}()

	select {
	case res := <-ch:
		return res.st, res.err
	case <-ctx.Done():
		return State{}, fmt.Errorf("reading /vram/state: %w", ctx.Err())
	}
}

func (c *Cards) fetchState(ctx context.Context, modelID string) (State, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "/vram/state", nil)
	if err != nil {
		return State{}, err
	}
	rec := httptest.NewRecorder()
	if !c.serve(modelID, rec, req) {
		return State{}, errNotReady
	}
	if rec.Code != http.StatusOK {
		return State{}, fmt.Errorf("/vram/state answered %d", rec.Code)
	}

	var st State
	if err := json.Unmarshal(rec.Body.Bytes(), &st); err != nil {
		return State{}, fmt.Errorf("decoding /vram/state: %w", err)
	}
	return st, nil
}

// normalizeCards returns the normalized address of every card in st. One bad
// address rejects the whole list rather than routing on part of it.
func normalizeCards(st State) ([]string, error) {
	cards := make([]string, 0, len(st.GPUs))
	for _, g := range st.GPUs {
		card, err := NormalizePCI(g.GPU)
		if err != nil {
			return nil, err
		}
		cards = append(cards, card)
	}
	return cards, nil
}

func (c *Cards) logReadError(modelID string, err error) {
	// A model that is not ready has no state to give, which is normal.
	if errors.Is(err, errNotReady) || c.logger == nil {
		return
	}
	c.logger.Warnf("<%s> vram: %v", modelID, err)
}
