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

const stopCard = "0000:0a:00.0"

// stoppingEngine is a fake upstream whose lend and return block until the
// test lets them go, so the model can stop while the call is in flight. A
// released call answers with stopStatus and stopBody, or, when stopStatus is
// 0, drops the connection as a process that exits does.
type stoppingEngine struct {
	srv        *httptest.Server
	stopStatus int
	stopBody   string

	arrived chan string
	release chan struct{}
	once    sync.Once

	mu    sync.Mutex
	calls []string
}

func newStoppingEngine(t *testing.T, stopStatus int, stopBody string) *stoppingEngine {
	t.Helper()
	e := &stoppingEngine{
		stopStatus: stopStatus,
		stopBody:   stopBody,
		arrived:    make(chan string, 4),
		release:    make(chan struct{}),
	}
	e.srv = httptest.NewServer(http.HandlerFunc(e.serve))
	t.Cleanup(e.srv.Close)
	// Runs before srv.Close, so a failed test never leaves a call blocked.
	t.Cleanup(e.letGo)
	return e
}

func (e *stoppingEngine) serve(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	if r.URL.Path == "/vram/state" {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"gpus":[{"gpu":%q,"lendable":8192,"lent":0}],"leases":[]}`, stopCard)
		return
	}

	e.mu.Lock()
	e.calls = append(e.calls, r.URL.Path+" "+string(body))
	e.mu.Unlock()
	e.arrived <- r.URL.Path
	<-e.release

	if e.stopStatus == 0 {
		conn, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			conn.Close()
		}
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.stopStatus)
	io.WriteString(w, e.stopBody)
}

func (e *stoppingEngine) letGo() { e.once.Do(func() { close(e.release) }) }

func (e *stoppingEngine) requests() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.calls...)
}

// stopFixture is a stub router in front of one stopping engine for model
// "a". ServeIfReady proxies to the engine while ready is set and records
// whether the model was ready for each forward; the stub's ServeHTTP, the
// path that loads models, only counts.
type stopFixture struct {
	engine *stoppingEngine
	ready  atomic.Bool
	starts atomic.Int32

	mu       sync.Mutex
	forwards []string
}

func newStopFixture(t *testing.T, stopStatus int, stopBody string) *stopFixture {
	f := &stopFixture{engine: newStoppingEngine(t, stopStatus, stopBody)}
	f.ready.Store(true)
	return f
}

func (f *stopFixture) server() *Server {
	stub := newStubRouter([]string{"a"}, "")
	stub.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
		f.starts.Add(1)
		w.WriteHeader(http.StatusOK)
	}
	stub.serveIfReady = func(modelID string, w http.ResponseWriter, r *http.Request) bool {
		if modelID != "a" || !f.ready.Load() {
			return false
		}
		if r.URL.Path != "/vram/state" {
			f.mu.Lock()
			f.forwards = append(f.forwards, r.URL.Path)
			f.mu.Unlock()
		}
		target, _ := url.Parse(f.engine.srv.URL)
		httputil.NewSingleHostReverseProxy(target).ServeHTTP(w, r)
		return true
	}
	return newTestServerWithConfig(lendModels(map[string]bool{"a": true}), stub, newStubRouter(nil, ""))
}

func (f *stopFixture) forwarded() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.forwards...)
}

// stopDuring sends a /vram request, waits until the engine holds it, takes
// the model out of ready, then lets the engine answer. It returns the
// client's answer and fails if the answer does not come.
func (f *stopFixture) stopDuring(t *testing.T, send func() *httptest.ResponseRecorder, path string) *httptest.ResponseRecorder {
	t.Helper()
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- send() }()

	select {
	case got := <-f.engine.arrived:
		require.Equal(t, path, got)
	case <-time.After(5 * time.Second):
		t.Fatalf("%s never reached the engine", path)
	}
	f.ready.Store(false)
	f.engine.letGo()

	select {
	case w := <-done:
		return w
	case <-time.After(5 * time.Second):
		t.Fatalf("%s did not answer after the model stopped", path)
		return nil
	}
}

func TestServer_VramStopDuringLend(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
		want   int
	}{
		{name: "engine answers 503", status: http.StatusServiceUnavailable, body: `{"error":"shutting down"}`, want: http.StatusServiceUnavailable},
		{name: "connection drops", want: http.StatusBadGateway},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newStopFixture(t, tc.status, tc.body)
			s := f.server()

			lend := `{"gpu":"0a:00.0","mib":2048}`
			w := f.stopDuring(t, func() *httptest.ResponseRecorder { return postLend(s, lend) }, "/vram/lend")

			assert.Equal(t, tc.want, w.Code, w.Body.String())
			assert.Equal(t, tc.body, w.Body.String(), "the process's answer is passed through")
			assert.Equal(t, []string{"/vram/lend " + lend}, f.engine.requests())
			assert.Equal(t, []string{"/vram/lend"}, f.forwarded(), "the lend was forwarded once, while the model was ready")
			assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")

			// The stopped model gets no further lend and is not started.
			assertNoLend(t, postLend(s, lend))
			assert.Len(t, f.engine.requests(), 1)
			assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
		})
	}
}

func TestServer_VramStopDuringReturn(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
		want   int
	}{
		{name: "engine answers 503", status: http.StatusServiceUnavailable, body: `{"error":"shutting down"}`, want: http.StatusServiceUnavailable},
		{name: "connection drops", want: http.StatusBadGateway},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newStopFixture(t, tc.status, tc.body)
			s := f.server()

			w := f.stopDuring(t, func() *httptest.ResponseRecorder { return postReturn(s, `{"id":"a:7"}`) }, "/vram/return")

			assert.Equal(t, tc.want, w.Code, w.Body.String())
			assert.Equal(t, tc.body, w.Body.String(), "the process's answer is passed through")
			assert.Equal(t, []string{`/vram/return {"id":"7"}`}, f.engine.requests())
			assert.Equal(t, []string{"/vram/return"}, f.forwarded(), "the return was forwarded once, while the model was ready")
			assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")

			// A return to the stopped model answers 404 and starts nothing.
			w = postReturn(s, `{"id":"a:7"}`)
			assert.Equal(t, http.StatusNotFound, w.Code, w.Body.String())
			assert.Len(t, f.engine.requests(), 1)
			assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
		})
	}
}

// TestServer_VramStopDuringLendRealRouter unloads a simple-responder lend
// model under the real router while its lend waits in the engine.
func TestServer_VramStopDuringLendRealRouter(t *testing.T) {
	bin := simpleResponderBinary(t)
	port := freePort(t)
	yaml := fmt.Sprintf(`
healthCheckTimeout: 15
models:
  lend:
    cmd: %s -port %d -silent -vram-gpus %s -vram-lendable 8192 -vram-lend-delay 30s
    proxy: http://127.0.0.1:%d
    vramLend: true
`, filepath.ToSlash(bin), port, stopCard, port)
	cfg, err := config.LoadConfigFromReader(strings.NewReader(yaml))
	require.NoError(t, err)

	st, err := sqlite.New("")
	require.NoError(t, err)
	t.Cleanup(func() { st.Close() })
	s, err := New(cfg, logmon.NewGroup(io.Discard, true, true, true), nil, st, BuildInfo{}, nil, nil)
	require.NoError(t, err)
	t.Cleanup(func() { s.Shutdown(5 * time.Second) })

	// Load the model with a normal request.
	w := httptest.NewRecorder()
	s.ServeHTTP(w, chatRequest("lend"))
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Equal(t, map[string]process.ProcessState{"lend": process.StateReady}, s.local.RunningModels())

	lend := `{"gpu":"0a:00.0","mib":2048}`
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- postLend(s, lend) }()

	// Wait until the engine holds the lend.
	require.Eventually(t, func() bool {
		resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/debug/vram", port))
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		var stats map[string]struct {
			Count int `json:"count"`
		}
		return json.NewDecoder(resp.Body).Decode(&stats) == nil && stats["lend"].Count == 1
	}, 5*time.Second, 10*time.Millisecond, "the lend never reached the engine")

	s.local.Unload(5*time.Second, "lend")

	select {
	case w = <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the lend did not answer after the model stopped")
	}
	assert.Equal(t, http.StatusBadGateway, w.Code, w.Body.String())
	assert.Never(t, func() bool { return len(s.local.RunningModels()) != 0 },
		300*time.Millisecond, 10*time.Millisecond, "a lend must not start the model")

	// A lend after the stop answers 409 and starts nothing.
	assertNoLend(t, postLend(s, lend))
	assert.Never(t, func() bool { return len(s.local.RunningModels()) != 0 },
		300*time.Millisecond, 10*time.Millisecond, "a lend must not start the model")
}
