package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
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

// lendEngine is a fake upstream that serves the VRAM Lend routes for the
// cards it lists and records every /vram request it gets.
type lendEngine struct {
	srv  *httptest.Server
	gpus []string

	mu         sync.Mutex
	vramCalls  int
	lendBodies []string
}

func newLendEngine(t *testing.T, gpus ...string) *lendEngine {
	t.Helper()
	e := &lendEngine{gpus: gpus}
	e.srv = httptest.NewServer(http.HandlerFunc(e.serve))
	t.Cleanup(e.srv.Close)
	return e
}

func (e *lendEngine) serve(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	if strings.HasPrefix(r.URL.Path, "/vram/") {
		e.mu.Lock()
		e.vramCalls++
		if r.URL.Path == "/vram/lend" {
			e.lendBodies = append(e.lendBodies, string(body))
		}
		e.mu.Unlock()
	}

	w.Header().Set("Content-Type", "application/json")
	switch r.URL.Path {
	case "/vram/state":
		gpus := make([]map[string]any, 0, len(e.gpus))
		for _, g := range e.gpus {
			gpus = append(gpus, map[string]any{"gpu": g, "lendable": 8192, "lent": 0})
		}
		json.NewEncoder(w).Encode(map[string]any{"gpus": gpus, "leases": []any{}})
	case "/vram/lend":
		var req struct {
			GPU string `json:"gpu"`
			MiB int64  `json:"mib"`
		}
		json.Unmarshal(body, &req)
		fmt.Fprintf(w, `{"id":"7","gpu":%q,"mib":%d}`, req.GPU, req.MiB)
	case "/vram/return":
		io.WriteString(w, `{"id":"7","gpu":"0000:0a:00.0","mib":2048}`)
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

func (e *lendEngine) calls() (int, []string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.vramCalls, append([]string(nil), e.lendBodies...)
}

// lendFixture is a stub router in front of fake engines. ServeIfReady proxies
// to the engine of a ready model; the stub's ServeHTTP, the path that loads
// models, only counts.
type lendFixture struct {
	engines map[string]*lendEngine
	ready   map[string]bool
	starts  atomic.Int32
}

func (f *lendFixture) router() *stubRouter {
	ids := make([]string, 0, len(f.engines))
	for id := range f.engines {
		ids = append(ids, id)
	}
	stub := newStubRouter(ids, "")
	stub.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
		f.starts.Add(1)
		w.WriteHeader(http.StatusOK)
	}
	stub.serveIfReady = func(modelID string, w http.ResponseWriter, r *http.Request) bool {
		e, ok := f.engines[modelID]
		if !ok || !f.ready[modelID] {
			return false
		}
		target, _ := url.Parse(e.srv.URL)
		httputil.NewSingleHostReverseProxy(target).ServeHTTP(w, r)
		return true
	}
	return stub
}

func (f *lendFixture) server(cfg config.Config) *Server {
	return newTestServerWithConfig(cfg, f.router(), newStubRouter(nil, ""))
}

func lendModels(lend map[string]bool) config.Config {
	cfg := config.Config{Models: map[string]config.ModelConfig{}}
	for id, lends := range lend {
		cfg.Models[id] = config.ModelConfig{Cmd: "llama-server", Proxy: "http://127.0.0.1:1", VramLend: lends}
	}
	return cfg
}

func postLend(s *Server, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/vram/lend", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)
	return w
}

func assertNoLend(t *testing.T, w *httptest.ResponseRecorder) {
	t.Helper()
	assert.Equal(t, http.StatusConflict, w.Code)
	assert.JSONEq(t, `{"max_mib":0}`, w.Body.String())
}

func TestServer_VramLendRouteReachesHolder(t *testing.T) {
	f := &lendFixture{
		engines: map[string]*lendEngine{
			"a": newLendEngine(t, "0000:0a:00.0"),
			"b": newLendEngine(t, "00000000:0B:00.0"),
		},
		ready: map[string]bool{"a": true, "b": true},
	}
	s := f.server(lendModels(map[string]bool{"a": true, "b": true}))

	body := `{"gpu":"0B:00.0","mib":2048}`
	w := postLend(s, body)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, `{"id":"b:7","gpu":"0B:00.0","mib":2048}`, w.Body.String())

	_, aLends := f.engines["a"].calls()
	assert.Empty(t, aLends, "the lend must reach only the holder")
	_, bLends := f.engines["b"].calls()
	assert.Equal(t, []string{body}, bLends, "the holder gets the body unchanged")
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramLendRouteNothingRunning(t *testing.T) {
	f := &lendFixture{
		engines: map[string]*lendEngine{"a": newLendEngine(t, "0000:0a:00.0")},
		ready:   map[string]bool{},
	}
	s := f.server(lendModels(map[string]bool{"a": true}))

	assertNoLend(t, postLend(s, `{"gpu":"0000:0a:00.0","mib":2048}`))

	_, lends := f.engines["a"].calls()
	assert.Empty(t, lends)
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramLendRouteNoModelLends(t *testing.T) {
	f := &lendFixture{engines: map[string]*lendEngine{}, ready: map[string]bool{}}
	s := f.server(config.Config{})

	assertNoLend(t, postLend(s, `{"gpu":"0000:0a:00.0","mib":2048}`))
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramLendRouteCardNotListed(t *testing.T) {
	f := &lendFixture{
		engines: map[string]*lendEngine{"a": newLendEngine(t, "0000:0a:00.0")},
		ready:   map[string]bool{"a": true},
	}
	s := f.server(lendModels(map[string]bool{"a": true}))

	assertNoLend(t, postLend(s, `{"gpu":"0000:0b:00.0","mib":2048}`))

	_, lends := f.engines["a"].calls()
	assert.Empty(t, lends, "a model that does not list the card gets no lend")
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramLendRouteSkipsModelWithoutVramLend(t *testing.T) {
	f := &lendFixture{
		engines: map[string]*lendEngine{
			"lend":  newLendEngine(t, "0000:0a:00.0"),
			"plain": newLendEngine(t, "0000:0b:00.0"),
		},
		ready: map[string]bool{"lend": true, "plain": true},
	}
	s := f.server(lendModels(map[string]bool{"lend": true, "plain": false}))

	assertNoLend(t, postLend(s, `{"gpu":"0000:0b:00.0","mib":2048}`))

	calls, _ := f.engines["plain"].calls()
	assert.Zero(t, calls, "a model without vramLend gets no /vram request")
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramLendRouteBadGPU(t *testing.T) {
	f := &lendFixture{
		engines: map[string]*lendEngine{"a": newLendEngine(t, "0000:0a:00.0")},
		ready:   map[string]bool{"a": true},
	}
	s := f.server(lendModels(map[string]bool{"a": true}))

	w := postLend(s, `{"gpu":"gpu0","mib":2048}`)
	assert.Equal(t, http.StatusBadRequest, w.Code)

	calls, _ := f.engines["a"].calls()
	assert.Zero(t, calls)
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func simpleResponderBinary(t *testing.T) string {
	t.Helper()
	name := fmt.Sprintf("simple-responder_%s_%s", runtime.GOOS, runtime.GOARCH)
	if runtime.GOOS == "windows" {
		name = "simple-responder.exe"
	}
	path, err := filepath.Abs(filepath.Join("..", "..", "build", name))
	require.NoError(t, err)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("simple-responder not found at %s, run `make simple-responder`", path)
	}
	return path
}

func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// TestServer_VramLendRouteRealRouter runs a lend model as a simple-responder
// process under the real router.
func TestServer_VramLendRouteRealRouter(t *testing.T) {
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

	lend := `{"gpu":"0a:00.0","mib":2048}`

	// While the model is stopped a lend answers 409 and starts nothing.
	assertNoLend(t, postLend(s, lend))
	assert.Never(t, func() bool { return len(s.local.RunningModels()) != 0 },
		200*time.Millisecond, 10*time.Millisecond, "a lend must not start the model")

	// Load the model with a normal request.
	w := httptest.NewRecorder()
	s.ServeHTTP(w, chatRequest("lend"))
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Equal(t, map[string]process.ProcessState{"lend": process.StateReady}, s.local.RunningModels())

	w = postLend(s, lend)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, `{"id":"lend:1","gpu":"0a:00.0","mib":2048}`, w.Body.String())
	assert.Equal(t, map[string]process.ProcessState{"lend": process.StateReady}, s.local.RunningModels())

	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/debug/vram", port))
	require.NoError(t, err)
	defer resp.Body.Close()
	var stats map[string]struct {
		Count    int    `json:"count"`
		LastBody string `json:"last_body"`
	}
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&stats))
	assert.Equal(t, 1, stats["lend"].Count)
	assert.Equal(t, lend, stats["lend"].LastBody)
}
