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

// returnEngine is a fake upstream that answers /vram/return with a fixed
// status and body and records every /vram request it gets.
type returnEngine struct {
	srv    *httptest.Server
	status int
	body   string

	mu     sync.Mutex
	bodies []string
}

func newReturnEngine(t *testing.T, status int, body string) *returnEngine {
	t.Helper()
	e := &returnEngine{status: status, body: body}
	e.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		e.mu.Lock()
		e.bodies = append(e.bodies, r.URL.Path+" "+string(b))
		e.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(e.status)
		io.WriteString(w, e.body)
	}))
	t.Cleanup(e.srv.Close)
	return e
}

func (e *returnEngine) requests() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.bodies...)
}

// returnFixture is a stub router in front of return engines. ServeIfReady
// proxies to the engine of a ready model; the stub's ServeHTTP, the path that
// loads models, only counts.
type returnFixture struct {
	engines map[string]*returnEngine
	ready   map[string]bool
	starts  atomic.Int32
}

func (f *returnFixture) server(cfg config.Config) *Server {
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
	return newTestServerWithConfig(cfg, stub, newStubRouter(nil, ""))
}

func postReturn(s *Server, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/vram/return", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)
	return w
}

const returnAnswer = `{"id":"7","gpu":"0000:0a:00.0","mib":8192}`

func TestServer_VramReturnRouteReachesModel(t *testing.T) {
	f := &returnFixture{
		engines: map[string]*returnEngine{
			"a": newReturnEngine(t, http.StatusOK, returnAnswer),
			"b": newReturnEngine(t, http.StatusOK, returnAnswer),
		},
		ready: map[string]bool{"a": true, "b": true},
	}
	s := f.server(lendModels(map[string]bool{"a": true, "b": true}))

	w := postReturn(s, `{"id":"b:7"}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, `{"id":"b:7","gpu":"0000:0a:00.0","mib":8192}`, w.Body.String())
	assert.Empty(t, f.engines["a"].requests(), "the return must reach only the named model")
	assert.Equal(t, []string{`/vram/return {"id":"7"}`}, f.engines["b"].requests())
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramReturnRouteNumericEngineID(t *testing.T) {
	f := &returnFixture{
		engines: map[string]*returnEngine{
			"a": newReturnEngine(t, http.StatusOK, `{"id":7,"gpu":"0000:0a:00.0","mib":8192}`),
		},
		ready: map[string]bool{"a": true},
	}
	s := f.server(lendModels(map[string]bool{"a": true}))

	w := postReturn(s, `{"id":"a:7"}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, `{"id":"a:7","gpu":"0000:0a:00.0","mib":8192}`, w.Body.String())
	assert.Equal(t, []string{`/vram/return {"id":"7"}`}, f.engines["a"].requests())
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramReturnRouteKeepsOtherFields(t *testing.T) {
	f := &returnFixture{
		engines: map[string]*returnEngine{"a": newReturnEngine(t, http.StatusOK, returnAnswer)},
		ready:   map[string]bool{"a": true},
	}
	s := f.server(lendModels(map[string]bool{"a": true}))

	w := postReturn(s, `{"id":"a:7","note":"x"}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, []string{`/vram/return {"id":"7","note":"x"}`}, f.engines["a"].requests())
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramReturnRouteNotFound(t *testing.T) {
	tests := []struct {
		name, body string
	}{
		{"unknown model", `{"id":"c:7"}`},
		{"model without vramLend", `{"id":"plain:7"}`},
		{"model not ready", `{"id":"stopped:7"}`},
		{"no prefix", `{"id":"7"}`},
		{"number id", `{"id":7}`},
		{"empty engine id", `{"id":"a:"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &returnFixture{
				engines: map[string]*returnEngine{
					"a":       newReturnEngine(t, http.StatusOK, returnAnswer),
					"plain":   newReturnEngine(t, http.StatusOK, returnAnswer),
					"stopped": newReturnEngine(t, http.StatusOK, returnAnswer),
				},
				ready: map[string]bool{"a": true, "plain": true},
			}
			s := f.server(lendModels(map[string]bool{"a": true, "plain": false, "stopped": true}))

			w := postReturn(s, tt.body)

			assert.Equal(t, http.StatusNotFound, w.Code, w.Body.String())
			for id, e := range f.engines {
				assert.Empty(t, e.requests(), "model %s must get no request", id)
			}
			assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
		})
	}
}

func TestServer_VramReturnRouteMissingID(t *testing.T) {
	for _, body := range []string{`{}`, `{"id":""}`, `{"id":null}`, `not json`} {
		t.Run(body, func(t *testing.T) {
			f := &returnFixture{
				engines: map[string]*returnEngine{"a": newReturnEngine(t, http.StatusOK, returnAnswer)},
				ready:   map[string]bool{"a": true},
			}
			s := f.server(lendModels(map[string]bool{"a": true}))

			w := postReturn(s, body)

			assert.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
			assert.Empty(t, f.engines["a"].requests())
			assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
		})
	}
}

func TestServer_VramReturnRouteEngineErrorPassesThrough(t *testing.T) {
	answer := `{"error":"the engine did not answer"}`
	f := &returnFixture{
		engines: map[string]*returnEngine{"a": newReturnEngine(t, http.StatusServiceUnavailable, answer)},
		ready:   map[string]bool{"a": true},
	}
	s := f.server(lendModels(map[string]bool{"a": true}))

	w := postReturn(s, `{"id":"a:7"}`)

	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Equal(t, answer, w.Body.String())
	assert.Equal(t, []string{`/vram/return {"id":"7"}`}, f.engines["a"].requests())
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

// TestServer_VramReturnRouteRealRouter lends and returns through a
// simple-responder process under the real router.
func TestServer_VramReturnRouteRealRouter(t *testing.T) {
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

	// While the model is stopped a return answers 404 and starts nothing.
	w := postReturn(s, `{"id":"lend:1"}`)
	assert.Equal(t, http.StatusNotFound, w.Code, w.Body.String())
	assert.Never(t, func() bool { return len(s.local.RunningModels()) != 0 },
		200*time.Millisecond, 10*time.Millisecond, "a return must not start the model")

	// Load the model with a normal request.
	w = httptest.NewRecorder()
	s.ServeHTTP(w, chatRequest("lend"))
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Equal(t, map[string]process.ProcessState{"lend": process.StateReady}, s.local.RunningModels())

	w = postLend(s, `{"gpu":"0a:00.0","mib":2048}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, `{"id":"lend:1","gpu":"0a:00.0","mib":2048}`, w.Body.String())

	w = postReturn(s, `{"id":"lend:1"}`)
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
	assert.Equal(t, 1, stats["return"].Count)
	assert.Equal(t, `{"id":"1"}`, stats["return"].LastBody)
}
