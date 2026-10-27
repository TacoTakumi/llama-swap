package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/store/sqlite"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// runEngine is a fake lend engine that, like a real one, numbers its leases
// from 1 again in each process run. restart starts a new run with no leases.
type runEngine struct {
	srv *httptest.Server

	mu      sync.Mutex
	next    int
	leases  map[string]bool
	returns []string
}

func newRunEngine(t *testing.T) *runEngine {
	t.Helper()
	e := &runEngine{leases: map[string]bool{}}
	e.srv = httptest.NewServer(http.HandlerFunc(e.serve))
	t.Cleanup(e.srv.Close)
	return e
}

func (e *runEngine) serve(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	e.mu.Lock()
	defer e.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	switch r.URL.Path {
	case "/vram/state":
		leases := []map[string]any{}
		for i := 1; i <= e.next; i++ {
			if id := strconv.Itoa(i); e.leases[id] {
				leases = append(leases, map[string]any{"id": id, "gpu": "0000:0a:00.0", "mib": 64})
			}
		}
		json.NewEncoder(w).Encode(map[string]any{
			"gpus":   []map[string]any{{"gpu": "0000:0a:00.0", "lendable": 8192, "lent": 64 * len(leases)}},
			"leases": leases,
		})
	case "/vram/lend":
		e.next++
		id := strconv.Itoa(e.next)
		e.leases[id] = true
		fmt.Fprintf(w, `{"id":%q,"gpu":"0000:0a:00.0","mib":64}`, id)
	case "/vram/return":
		e.returns = append(e.returns, string(body))
		var req struct {
			ID string `json:"id"`
		}
		json.Unmarshal(body, &req)
		if !e.leases[req.ID] {
			w.WriteHeader(http.StatusNotFound)
			io.WriteString(w, `{"error":"no such lease"}`)
			return
		}
		delete(e.leases, req.ID)
		fmt.Fprintf(w, `{"id":%q,"gpu":"0000:0a:00.0","mib":64}`, req.ID)
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

func (e *runEngine) restart() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.next = 0
	e.leases = map[string]bool{}
	e.returns = nil
}

func (e *runEngine) lent(id string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.leases[id]
}

func (e *runEngine) returned() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.returns...)
}

// restartFixture is one lend model "a" in front of a runEngine. The stub's
// ServeHTTP, the path that loads models, only counts.
type restartFixture struct {
	engine *runEngine
	ready  atomic.Bool
	starts atomic.Int32
}

func newRestartFixture(t *testing.T) (*restartFixture, *Server) {
	t.Helper()
	f := &restartFixture{engine: newRunEngine(t)}
	f.ready.Store(true)
	stub := newStubRouter([]string{"a"}, "")
	stub.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
		f.starts.Add(1)
		w.WriteHeader(http.StatusOK)
	}
	stub.serveIfReady = func(modelID string, w http.ResponseWriter, r *http.Request) bool {
		if modelID != "a" || !f.ready.Load() {
			return false
		}
		target, _ := url.Parse(f.engine.srv.URL)
		httputil.NewSingleHostReverseProxy(target).ServeHTTP(w, r)
		return true
	}
	s := newTestServerWithConfig(lendModels(map[string]bool{"a": true}), stub, newStubRouter(nil, ""))
	s.vramCards.Ready("a")
	return f, s
}

// restart stops the model and starts it again, with the state events a
// process run sends on the way: stopping, stopped, starting, then ready.
func (f *restartFixture) restart(s *Server) {
	f.ready.Store(false)
	for range 3 {
		s.vramCards.Drop("a")
	}
	f.engine.restart()
	f.ready.Store(true)
	s.vramCards.Ready("a")
}

func TestServer_VramReturnRouteStaleRunID(t *testing.T) {
	f, s := newRestartFixture(t)
	lend := `{"gpu":"0000:0a:00.0","mib":64}`

	w := postLend(s, lend)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	oldID := answerID(t, w)

	// The model restarts. Its new run numbers leases from 1 again, so the
	// new borrower's lease has the same engine id as the old one.
	f.restart(s)
	w = postLend(s, lend)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	newID := answerID(t, w)
	require.True(t, f.engine.lent("1"))

	// A return of the id from before the restart must not reach the new
	// run, where it would return the new borrower's lease.
	w = postReturn(s, fmt.Sprintf(`{"id":%q}`, oldID))
	assert.Equal(t, http.StatusNotFound, w.Code, w.Body.String())
	assert.Empty(t, f.engine.returned(), "the new run must get no return of an old id")
	assert.True(t, f.engine.lent("1"), "the new run's lease must still be lent")
	assert.NotEqual(t, oldID, newID, "a lease id must name one process run")

	// The new id returns the new lease.
	w = postReturn(s, fmt.Sprintf(`{"id":%q}`, newID))
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, newID, answerID(t, w))
	assert.Equal(t, []string{`{"id":"1"}`}, f.engine.returned())
	assert.False(t, f.engine.lent("1"))
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramLendRouteRunToken(t *testing.T) {
	f, s := newRestartFixture(t)
	lend := `{"gpu":"0000:0a:00.0","mib":64}`

	// Two lends in one run carry the same token.
	run := s.vramCards.Run("a")
	w := postLend(s, lend)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "a:"+run+":1", answerID(t, w))
	w = postLend(s, lend)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "a:"+run+":2", answerID(t, w))

	w = getVramState(s)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, `{
		"gpus": [{"gpu":"0000:0a:00.0","lendable":8192,"lent":128,"model":"a"}],
		"leases": [
			{"id":"a:`+run+`:1","gpu":"0000:0a:00.0","mib":64,"model":"a"},
			{"id":"a:`+run+`:2","gpu":"0000:0a:00.0","mib":64,"model":"a"}
		]
	}`, w.Body.String())

	// After a restart the token differs, in lend answers and in the state.
	f.restart(s)
	next := s.vramCards.Run("a")
	require.NotEqual(t, run, next)
	w = postLend(s, lend)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "a:"+next+":1", answerID(t, w))

	w = getVramState(s)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, `{
		"gpus": [{"gpu":"0000:0a:00.0","lendable":8192,"lent":64,"model":"a"}],
		"leases": [{"id":"a:`+next+`:1","gpu":"0000:0a:00.0","mib":64,"model":"a"}]
	}`, w.Body.String())
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

// A lend can reach a new run before its ready event does. The lease it gets
// keeps its id when the event arrives.
func TestServer_VramLendRouteBeforeReadyEvent(t *testing.T) {
	f, s := newRestartFixture(t)
	f.ready.Store(false)
	s.vramCards.Drop("a")
	f.engine.restart()
	f.ready.Store(true)

	w := postLend(s, `{"gpu":"0000:0a:00.0","mib":64}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	id := answerID(t, w)

	s.vramCards.Ready("a")
	w = postReturn(s, fmt.Sprintf(`{"id":%q}`, id))
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, id, answerID(t, w))
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

// TestServer_VramReturnRouteRealRouterRestart lends from a simple-responder
// process under the real router, unloads and reloads it, and lends again.
// The new process numbers its leases from 1 again.
func TestServer_VramReturnRouteRealRouterRestart(t *testing.T) {
	bin := simpleResponderBinary(t)
	port := freePort(t)
	yaml := fmt.Sprintf(`
healthCheckTimeout: 15
models:
  lend:
    cmd: %s -port %d -silent -vram-gpus 0000:0a:00.0 -vram-lendable 8192
    proxy: http://127.0.0.1:%d
    vramLend: true
`, filepath.ToSlash(bin), port, port)
	cfg, err := config.LoadConfigFromReader(strings.NewReader(yaml))
	require.NoError(t, err)

	st, err := sqlite.New("")
	require.NoError(t, err)
	t.Cleanup(func() { st.Close() })
	s, err := New(cfg, logmon.NewGroup(io.Discard, true, true, true), nil, st, BuildInfo{}, nil, nil)
	require.NoError(t, err)
	t.Cleanup(func() { s.Shutdown(5 * time.Second) })

	load := func() {
		t.Helper()
		w := httptest.NewRecorder()
		s.ServeHTTP(w, chatRequest("lend"))
		require.Equal(t, http.StatusOK, w.Code, w.Body.String())
		require.Equal(t, map[string]process.ProcessState{"lend": process.StateReady}, s.local.RunningModels())
	}
	lend := `{"gpu":"0a:00.0","mib":2048}`

	load()
	w := postLend(s, lend)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	oldID := answerID(t, w)
	oldRun := s.vramCards.Run("lend")
	require.Equal(t, "lend:"+oldRun+":1", oldID)

	s.local.Unload(5*time.Second, "lend")
	require.Empty(t, s.local.RunningModels())
	require.Eventually(t, func() bool { return !s.vramCards.IsRun("lend", oldRun) },
		5*time.Second, 10*time.Millisecond, "the stop must end the run")

	load()
	w = postLend(s, lend)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	newID := answerID(t, w)
	assert.Equal(t, "lend:"+s.vramCards.Run("lend")+":1", newID)
	require.NotEqual(t, oldID, newID)

	// The old id answers 404 and leaves the new lease alone.
	w = postReturn(s, fmt.Sprintf(`{"id":%q}`, oldID))
	assert.Equal(t, http.StatusNotFound, w.Code, w.Body.String())

	w = postReturn(s, fmt.Sprintf(`{"id":%q}`, newID))
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, newID, answerID(t, w))
	assert.Equal(t, map[string]process.ProcessState{"lend": process.StateReady}, s.local.RunningModels())

	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/debug/vram", port))
	require.NoError(t, err)
	defer resp.Body.Close()
	var stats map[string]struct {
		Count    int    `json:"count"`
		LastBody string `json:"last_body"`
	}
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&stats))
	assert.Equal(t, 1, stats["return"].Count, "only the new id reached the new process")
	assert.Equal(t, `{"id":"1"}`, stats["return"].LastBody)
}
