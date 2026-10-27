package vramlend

import (
	"context"
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

// Cards keeps the cards each lend model uses, for routing a lend to the model
// that holds the card. A list belongs to one process run: it is read when the
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
}

// cardRun is one state read for one process run. cards is written before
// done closes and is read only after.
type cardRun struct {
	done  chan struct{}
	cards []string
}

// NewCards tracks the cards of models, the config keys of the models that
// lend VRAM. Lookups match them in the order given. Reads end when ctx does.
func NewCards(ctx context.Context, models []string, serve ServeFunc, logger *logmon.Monitor) *Cards {
	return &Cards{
		Timeout: DefaultStateTimeout,
		ctx:     ctx,
		models:  slices.Clone(models),
		serve:   serve,
		logger:  logger,
		runs:    make(map[string]*cardRun),
	}
}

// Ready starts a background read of the model's cards for its new process
// run, replacing any list from before. It returns at once, so it is safe to
// call from the event dispatcher.
func (c *Cards) Ready(modelID string) {
	if c == nil || !slices.Contains(c.models, modelID) {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.startLocked(modelID)
}

// Drop forgets the model's cards. Call it when the model leaves the ready
// state; the cards of its next run are read when it is ready again.
func (c *Cards) Drop(modelID string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.runs, modelID)
}

// Holder returns the lend model that uses card, a normalized PCI address.
//
// A model with no list yet is read on demand, and a lookup that finds a read
// in flight waits for it rather than starting another. A model that is not
// ready answers no list at once, and a read is never waited on past Timeout.
// Holder returns false when no model holds card or ctx ends first.
func (c *Cards) Holder(ctx context.Context, card string) (string, bool) {
	if c == nil {
		return "", false
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

	// A list that is already in answers without waiting on a slower read of
	// another model.
	for i, r := range runs {
		select {
		case <-r.done:
			if slices.Contains(r.cards, card) {
				return c.models[i], true
			}
		default:
		}
	}

	for i, r := range runs {
		select {
		case <-r.done:
		case <-ctx.Done():
			return "", false
		}
		if slices.Contains(r.cards, card) {
			return c.models[i], true
		}
	}
	return "", false
}

// ReadAll reads the state of every lend model in parallel, each bounded by
// Timeout. A model that is not ready, fails or runs out of time is left out.
func (c *Cards) ReadAll(ctx context.Context) map[string]State {
	states := make(map[string]State)
	if c == nil {
		return states
	}

	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, model := range c.models {
		wg.Go(func() {
			st, err := c.readState(ctx, model)
			if err != nil {
				c.logReadError(model, err)
				return
			}
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
