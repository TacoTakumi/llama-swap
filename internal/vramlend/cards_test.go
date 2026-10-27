package vramlend

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/logmon"
)

// fakeEngine is one lend model behind a fake router. Its state answer takes
// delay and lists gpus.
type fakeEngine struct {
	mu    sync.Mutex
	ready bool
	gpus  []string
	delay time.Duration
	// gate, when set, holds every state read until it is closed.
	gate chan struct{}
	// ignoreCtx makes a read sleep its full delay even after the request's
	// context ended, like an upstream that does not honor cancellation.
	ignoreCtx bool

	reads atomic.Int32
}

// fakeRouter stands in for the local router. serveIfReady is the only entry
// point Cards gets; serveHTTP is the load path, which starts a model, and
// counts every call so a test can prove no /vram read went through it.
type fakeRouter struct {
	engines map[string]*fakeEngine
	starts  atomic.Int32
}

func newFakeRouter(engines map[string]*fakeEngine) *fakeRouter {
	return &fakeRouter{engines: engines}
}

func (f *fakeRouter) serveHTTP(w http.ResponseWriter, r *http.Request) {
	f.starts.Add(1)
	w.WriteHeader(http.StatusOK)
}

func (f *fakeRouter) serveIfReady(modelID string, w http.ResponseWriter, req *http.Request) bool {
	e, ok := f.engines[modelID]
	if !ok {
		return false
	}
	e.mu.Lock()
	ready, gpus, delay, gate, ignoreCtx := e.ready, e.gpus, e.delay, e.gate, e.ignoreCtx
	e.mu.Unlock()
	if !ready {
		return false
	}
	if req.Method != http.MethodGet || req.URL.Path != "/vram/state" {
		w.WriteHeader(http.StatusNotFound)
		return true
	}
	e.reads.Add(1)

	if gate != nil {
		<-gate
	}
	if ignoreCtx {
		time.Sleep(delay)
	} else {
		select {
		case <-time.After(delay):
		case <-req.Context().Done():
			w.WriteHeader(http.StatusBadGateway)
			return true
		}
	}

	type gpu struct {
		GPU      string `json:"gpu"`
		Lendable int64  `json:"lendable"`
		Lent     int64  `json:"lent"`
	}
	body := struct {
		GPUs   []gpu `json:"gpus"`
		Leases []any `json:"leases"`
	}{Leases: []any{}}
	for _, g := range gpus {
		body.GPUs = append(body.GPUs, gpu{GPU: g, Lendable: 1024})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(body)
	return true
}

func (e *fakeEngine) set(ready bool, gpus ...string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.ready = ready
	e.gpus = gpus
}

func newTestCards(t *testing.T, f *fakeRouter, timeout time.Duration, models ...string) *Cards {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	c := NewCards(ctx, models, f.serveIfReady, logmon.NewWriter(io.Discard))
	c.Timeout = timeout
	return c
}

// assertNoStart fails when any /vram read went through the load path.
func assertNoStart(t *testing.T, f *fakeRouter) {
	t.Helper()
	if got := f.starts.Load(); got != 0 {
		t.Errorf("starts=%d want 0: a /vram read went through the load path", got)
	}
}

func requireHolders(t *testing.T, c *Cards, card string, want ...string) {
	t.Helper()
	got := c.Holders(context.Background(), card)
	if !slices.Equal(got, want) {
		t.Fatalf("Holders(%q)=%q want %q", card, got, want)
	}
}

func requireNoHolder(t *testing.T, c *Cards, card string) {
	t.Helper()
	if got := c.Holders(context.Background(), card); len(got) != 0 {
		t.Fatalf("Holders(%q)=%q want no holder", card, got)
	}
}

// waitReads waits until e has answered n state reads.
func waitReads(t *testing.T, e *fakeEngine, n int32) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for e.reads.Load() < n {
		if time.Now().After(deadline) {
			t.Fatalf("reads=%d want %d", e.reads.Load(), n)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestVramLend_CardsOneReadForThreeLookups(t *testing.T) {
	e := &fakeEngine{ready: true, gpus: []string{"0000:0a:00.0"}}
	f := newFakeRouter(map[string]*fakeEngine{"lend": e})
	c := newTestCards(t, f, time.Second, "lend")

	c.Ready("lend")
	for range 3 {
		requireHolders(t, c, "0000:0a:00.0", "lend")
	}
	if got := e.reads.Load(); got != 1 {
		t.Errorf("reads=%d want 1", got)
	}
	assertNoStart(t, f)
}

func TestVramLend_CardsReadInBackgroundOnReady(t *testing.T) {
	// The engine takes delay to answer state. A lend arriving well after
	// the model became ready must not pay that delay again: the read ran in
	// the background when the model became ready.
	const delay = 200 * time.Millisecond
	e := &fakeEngine{ready: true, gpus: []string{"0000:0a:00.0"}, delay: delay}
	f := newFakeRouter(map[string]*fakeEngine{"lend": e})
	c := newTestCards(t, f, time.Second, "lend")

	c.Ready("lend")
	waitReads(t, e, 1)
	time.Sleep(delay * 3 / 2)

	start := time.Now()
	requireHolders(t, c, "0000:0a:00.0", "lend")
	if elapsed := time.Since(start); elapsed > delay/10 {
		t.Errorf("lookup took %v, want under %v", elapsed, delay/10)
	}
	if got := e.reads.Load(); got != 1 {
		t.Errorf("reads=%d want 1", got)
	}
	assertNoStart(t, f)
}

func TestVramLend_CardsLookupJoinsInFlightRead(t *testing.T) {
	for _, tc := range []struct {
		name  string
		ready bool
	}{
		// a lend that arrives while the ready read is still running
		{"after ready", true},
		// a lend that arrives before the ready event was delivered reads on
		// demand, and the lends behind it join that read
		{"before ready event", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gate := make(chan struct{})
			e := &fakeEngine{ready: true, gpus: []string{"0000:0a:00.0"}, gate: gate}
			f := newFakeRouter(map[string]*fakeEngine{"lend": e})
			c := newTestCards(t, f, 5*time.Second, "lend")

			if tc.ready {
				c.Ready("lend")
				waitReads(t, e, 1)
			}

			const lookups = 3
			results := make(chan string, lookups)
			for range lookups {
				go func() {
					got := c.Holders(context.Background(), "0000:0a:00.0")
					results <- strings.Join(got, ",")
				}()
			}
			waitReads(t, e, 1)
			// let every lookup reach the wait before the read answers
			time.Sleep(20 * time.Millisecond)
			close(gate)

			for range lookups {
				select {
				case got := <-results:
					if got != "lend" {
						t.Errorf("Holders=%q want lend", got)
					}
				case <-time.After(5 * time.Second):
					t.Fatal("lookup never answered")
				}
			}
			if got := e.reads.Load(); got != 1 {
				t.Errorf("reads=%d want 1", got)
			}
			assertNoStart(t, f)
		})
	}
}

func TestVramLend_CardsDroppedWhenModelLeavesReady(t *testing.T) {
	e := &fakeEngine{ready: true, gpus: []string{"0000:0a:00.0"}}
	f := newFakeRouter(map[string]*fakeEngine{"lend": e})
	c := newTestCards(t, f, time.Second, "lend")

	c.Ready("lend")
	requireHolders(t, c, "0000:0a:00.0", "lend")

	// The model stops. The list goes with the process run.
	e.set(false)
	c.Drop("lend")
	requireNoHolder(t, c, "0000:0a:00.0")

	// It comes back on a different card.
	e.set(true, "0000:0b:00.0")
	c.Ready("lend")
	requireHolders(t, c, "0000:0b:00.0", "lend")
	requireNoHolder(t, c, "0000:0a:00.0")
	assertNoStart(t, f)
}

func TestVramLend_CardsRoutesToModelHoldingCard(t *testing.T) {
	a := &fakeEngine{ready: true, gpus: []string{"0000:01:00.0"}}
	b := &fakeEngine{ready: true, gpus: []string{"00000000:0A:00.0", "0000:0b:00.0"}}
	stopped := &fakeEngine{gpus: []string{"0000:0c:00.0"}}
	f := newFakeRouter(map[string]*fakeEngine{"a": a, "b": b, "stopped": stopped})
	c := newTestCards(t, f, time.Second, "a", "b", "stopped")

	c.Ready("a")
	c.Ready("b")
	requireHolders(t, c, "0000:01:00.0", "a")
	// the engine's nvidia-smi form is normalized before matching
	requireHolders(t, c, "0000:0a:00.0", "b")
	requireHolders(t, c, "0000:0b:00.0", "b")
	requireNoHolder(t, c, "0000:0c:00.0")
	requireNoHolder(t, c, "0000:0d:00.0")
	if got := stopped.reads.Load(); got != 0 {
		t.Errorf("stopped reads=%d want 0", got)
	}
	assertNoStart(t, f)
}

func TestVramLendOrder_CardsHoldersInModelOrder(t *testing.T) {
	// Every holder of the card comes back, in the order the models were
	// given, even when a later model's list is in first.
	gate := make(chan struct{})
	a := &fakeEngine{ready: true, gpus: []string{"0000:0a:00.0"}, gate: gate}
	b := &fakeEngine{ready: true, gpus: []string{"0000:01:00.0"}}
	c3 := &fakeEngine{ready: true, gpus: []string{"0000:0a:00.0", "0000:0b:00.0"}}
	stopped := &fakeEngine{gpus: []string{"0000:0a:00.0"}}
	f := newFakeRouter(map[string]*fakeEngine{"a": a, "b": b, "c": c3, "stopped": stopped})
	c := newTestCards(t, f, 5*time.Second, "a", "b", "c", "stopped")

	c.Ready("b")
	c.Ready("c")
	waitReads(t, c3, 1)
	time.AfterFunc(50*time.Millisecond, func() { close(gate) })

	requireHolders(t, c, "0000:0a:00.0", "a", "c")
	requireHolders(t, c, "0000:0b:00.0", "c")
	requireHolders(t, c, "0000:01:00.0", "b")
	assertNoStart(t, f)
}

func TestVramLend_CardsIgnoreModelsThatDoNotLend(t *testing.T) {
	e := &fakeEngine{ready: true, gpus: []string{"0000:0a:00.0"}}
	f := newFakeRouter(map[string]*fakeEngine{"other": e})
	c := newTestCards(t, f, time.Second, "lend")

	c.Ready("other")
	requireNoHolder(t, c, "0000:0a:00.0")
	if got := e.reads.Load(); got != 0 {
		t.Errorf("reads=%d want 0", got)
	}
	assertNoStart(t, f)
}

func TestVramLend_CardsReadTimeout(t *testing.T) {
	// The engine never answers within the timeout and ignores cancellation.
	// The lookup must give up at the timeout, and the next one reads again.
	const timeout = 100 * time.Millisecond
	e := &fakeEngine{ready: true, gpus: []string{"0000:0a:00.0"}, delay: 20 * timeout, ignoreCtx: true}
	f := newFakeRouter(map[string]*fakeEngine{"lend": e})
	c := newTestCards(t, f, timeout, "lend")

	c.Ready("lend")
	start := time.Now()
	requireNoHolder(t, c, "0000:0a:00.0")
	if elapsed := time.Since(start); elapsed > timeout*3/2 {
		t.Errorf("lookup took %v, want under %v", elapsed, timeout*3/2)
	}

	e.mu.Lock()
	e.delay, e.ignoreCtx = 0, false
	e.mu.Unlock()
	requireHolders(t, c, "0000:0a:00.0", "lend")
	if got := e.reads.Load(); got != 2 {
		t.Errorf("reads=%d want 2", got)
	}
	assertNoStart(t, f)
}

func TestVramLend_CardsLookupHonorsContext(t *testing.T) {
	gate := make(chan struct{})
	t.Cleanup(func() { close(gate) })
	e := &fakeEngine{ready: true, gpus: []string{"0000:0a:00.0"}, gate: gate}
	f := newFakeRouter(map[string]*fakeEngine{"lend": e})
	c := newTestCards(t, f, 5*time.Second, "lend")

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if got := c.Holders(ctx, "0000:0a:00.0"); len(got) != 0 {
		t.Fatalf("Holders=%q want no holder after the caller gave up", got)
	}
	assertNoStart(t, f)
}

func TestVramLend_CardsReadAllInParallel(t *testing.T) {
	// Scaled from a 35 s timeout: two engines answer in 0.5 s, one takes
	// longer than the timeout and is left out.
	const (
		delay   = 500 * time.Millisecond
		timeout = 875 * time.Millisecond
	)
	engines := map[string]*fakeEngine{
		"a":       {ready: true, gpus: []string{"0000:01:00.0"}, delay: delay},
		"b":       {ready: true, gpus: []string{"0000:0a:00.0"}, delay: delay},
		"slow":    {ready: true, gpus: []string{"0000:0b:00.0"}, delay: 2 * timeout},
		"stopped": {gpus: []string{"0000:0c:00.0"}},
	}
	f := newFakeRouter(engines)
	c := newTestCards(t, f, timeout, "a", "b", "slow", "stopped")

	start := time.Now()
	states := c.ReadAll(context.Background())
	elapsed := time.Since(start)

	if elapsed > timeout+timeout/16 {
		t.Errorf("ReadAll took %v, want under %v", elapsed, timeout+timeout/16)
	}
	if len(states) != 2 {
		t.Fatalf("states=%v want a and b", states)
	}
	for model, card := range map[string]string{"a": "0000:01:00.0", "b": "0000:0a:00.0"} {
		st, ok := states[model]
		if !ok || len(st.GPUs) != 1 || st.GPUs[0].GPU != card {
			t.Errorf("states[%s]=%+v want one gpu %s", model, st, card)
		}
	}
	assertNoStart(t, f)
}

func TestVramLend_CardsReadAllFastEngines(t *testing.T) {
	const delay = 200 * time.Millisecond
	engines := map[string]*fakeEngine{
		"a": {ready: true, gpus: []string{"0000:01:00.0"}, delay: delay},
		"b": {ready: true, gpus: []string{"0000:0a:00.0"}, delay: delay},
	}
	f := newFakeRouter(engines)
	c := newTestCards(t, f, time.Second, "a", "b")

	start := time.Now()
	states := c.ReadAll(context.Background())
	if elapsed := time.Since(start); elapsed > delay*5/4 {
		t.Errorf("ReadAll took %v, want under %v: reads did not run in parallel", elapsed, delay*5/4)
	}
	if len(states) != 2 {
		t.Errorf("states=%v want a and b", states)
	}
	assertNoStart(t, f)
}

func TestVramLend_CardsRejectBadAddress(t *testing.T) {
	// An engine that reports an address that is not PCI gives no list, so
	// no lend is routed on a guess.
	e := &fakeEngine{ready: true, gpus: []string{"0000:0a:00.0", "gpu0"}}
	f := newFakeRouter(map[string]*fakeEngine{"lend": e})
	c := newTestCards(t, f, time.Second, "lend")

	c.Ready("lend")
	requireNoHolder(t, c, "0000:0a:00.0")
	assertNoStart(t, f)
}

func TestVramLend_CardsNilIsSafe(t *testing.T) {
	var c *Cards
	c.Ready("lend")
	c.Drop("lend")
	if got := c.Holders(context.Background(), "0000:0a:00.0"); len(got) != 0 {
		t.Error("nil Cards found a holder")
	}
	if got := c.ReadAll(context.Background()); len(got) != 0 {
		t.Errorf("nil Cards ReadAll=%v want empty", got)
	}
}
