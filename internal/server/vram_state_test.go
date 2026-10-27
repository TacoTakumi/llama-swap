package server

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"path/filepath"
	"strings"
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

// stateEngine is a fake upstream whose /vram/state answers status and body
// after delay. It counts every /vram request it gets.
type stateEngine struct {
	srv       *httptest.Server
	status    int
	body      string
	delay     time.Duration
	vramCalls atomic.Int32
}

func newStateEngine(t *testing.T, status int, body string, delay time.Duration) *stateEngine {
	t.Helper()
	e := &stateEngine{status: status, body: body, delay: delay}
	e.srv = httptest.NewServer(http.HandlerFunc(e.serve))
	t.Cleanup(e.srv.Close)
	return e
}

func (e *stateEngine) serve(w http.ResponseWriter, r *http.Request) {
	if !strings.HasPrefix(r.URL.Path, "/vram/") {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	e.vramCalls.Add(1)
	if e.delay > 0 {
		select {
		case <-time.After(e.delay):
		case <-r.Context().Done():
			return
		}
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.status)
	io.WriteString(w, e.body)
}

// stateFixture is a stub router in front of state engines, every one of them
// ready. ServeIfReady proxies to the model's engine; the stub's ServeHTTP,
// the path that loads models, only counts.
type stateFixture struct {
	engines map[string]*stateEngine
	starts  atomic.Int32
}

func (f *stateFixture) server(t *testing.T, cfg config.Config, timeout time.Duration) *Server {
	t.Helper()
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
		if !ok {
			return false
		}
		target, _ := url.Parse(e.srv.URL)
		httputil.NewSingleHostReverseProxy(target).ServeHTTP(w, r)
		return true
	}
	s := newTestServerWithConfig(cfg, stub, newStubRouter(nil, ""))
	if timeout > 0 && s.vramCards != nil {
		s.vramCards.Timeout = timeout
	}
	return s
}

func getVramState(s *Server) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/vram/state", nil))
	return w
}

func stateBody(gpu string, lendable, lent int64, leaseID string, mib int64) string {
	return fmt.Sprintf(`{"gpus":[{"gpu":%q,"lendable":%d,"lent":%d}],"leases":[{"id":%q,"gpu":%q,"mib":%d}]}`,
		gpu, lendable, lent, leaseID, gpu, mib)
}

func TestServer_VramStateNothingRunning(t *testing.T) {
	f := &stateFixture{engines: map[string]*stateEngine{}}
	s := f.server(t, lendModels(map[string]bool{"a": true, "b": true}), 0)

	w := getVramState(s)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.JSONEq(t, withRun(s, `{"gpus":[],"leases":[]}`), w.Body.String())
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramStateNoModelLends(t *testing.T) {
	f := &stateFixture{engines: map[string]*stateEngine{}}
	s := f.server(t, config.Config{}, 0)

	w := getVramState(s)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, withRun(s, `{"gpus":[],"leases":[]}`), w.Body.String())
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramStateSharedCard(t *testing.T) {
	f := &stateFixture{engines: map[string]*stateEngine{
		"b": newStateEngine(t, http.StatusOK, stateBody("0A:00.0", 1024, 2048, "3", 2048), 0),
		"a": newStateEngine(t, http.StatusOK, stateBody("00000000:0A:00.0", 8192, 4096, "7", 4096), 0),
	}}
	s := f.server(t, lendModels(map[string]bool{"a": true, "b": true}), 0)

	w := getVramState(s)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, withRun(s, `{
		"gpus": [
			{"gpu":"0000:0a:00.0","lendable":8192,"lent":4096,"model":"a"},
			{"gpu":"0000:0a:00.0","lendable":1024,"lent":2048,"model":"b"}
		],
		"leases": [
			{"id":"a:{run:a}:7","gpu":"0000:0a:00.0","mib":4096,"model":"a"},
			{"id":"b:{run:b}:3","gpu":"0000:0a:00.0","mib":2048,"model":"b"}
		]
	}`), w.Body.String())
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramStateLeavesOutFailingEngine(t *testing.T) {
	for name, bad := range map[string]*stateEngine{
		"500":      newStateEngine(t, http.StatusInternalServerError, `{"error":"boom"}`, 0),
		"not json": newStateEngine(t, http.StatusOK, `not json`, 0),
		"bad gpu":  newStateEngine(t, http.StatusOK, stateBody("gpu0", 0, 0, "1", 0), 0),
	} {
		t.Run(name, func(t *testing.T) {
			f := &stateFixture{engines: map[string]*stateEngine{
				"a": bad,
				"b": newStateEngine(t, http.StatusOK, stateBody("0000:0b:00.0", 8192, 64, "1", 64), 0),
			}}
			s := f.server(t, lendModels(map[string]bool{"a": true, "b": true}), 0)

			w := getVramState(s)

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			assert.JSONEq(t, withRun(s, `{
				"gpus": [{"gpu":"0000:0b:00.0","lendable":8192,"lent":64,"model":"b"}],
				"leases": [{"id":"b:{run:b}:1","gpu":"0000:0b:00.0","mib":64,"model":"b"}]
			}`), w.Body.String())
			assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
		})
	}
}

func TestServer_VramStateSkipsModelWithoutVramLend(t *testing.T) {
	f := &stateFixture{engines: map[string]*stateEngine{
		"lend":  newStateEngine(t, http.StatusOK, stateBody("0000:0a:00.0", 8192, 0, "1", 0), 0),
		"plain": newStateEngine(t, http.StatusOK, stateBody("0000:0b:00.0", 8192, 0, "1", 0), 0),
	}}
	s := f.server(t, lendModels(map[string]bool{"lend": true, "plain": false}), 0)

	w := getVramState(s)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, withRun(s, `{
		"gpus": [{"gpu":"0000:0a:00.0","lendable":8192,"lent":0,"model":"lend"}],
		"leases": [{"id":"lend:{run:lend}:1","gpu":"0000:0a:00.0","mib":0,"model":"lend"}]
	}`), w.Body.String())
	assert.Zero(t, f.engines["plain"].vramCalls.Load(), "a model without vramLend gets no /vram request")
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

// The timing tests scale the 35 s state timeout down 40 times: 20 s becomes
// 500 ms, 25 s 625 ms, 37 s 925 ms and 40 s 1 s.
const (
	stateTimeout = 875 * time.Millisecond
	stateScale   = 40
)

func scaled(seconds int) time.Duration {
	return time.Duration(seconds) * time.Second / stateScale
}

func TestServer_VramStateReadsInParallel(t *testing.T) {
	f := &stateFixture{engines: map[string]*stateEngine{
		"a": newStateEngine(t, http.StatusOK, stateBody("0000:0a:00.0", 8192, 0, "1", 0), scaled(20)),
		"b": newStateEngine(t, http.StatusOK, stateBody("0000:0b:00.0", 4096, 0, "2", 0), scaled(20)),
	}}
	s := f.server(t, lendModels(map[string]bool{"a": true, "b": true}), stateTimeout)

	start := time.Now()
	w := getVramState(s)
	elapsed := time.Since(start)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Less(t, elapsed, scaled(25), "two engines must be read in parallel")
	assert.JSONEq(t, withRun(s, `{
		"gpus": [
			{"gpu":"0000:0a:00.0","lendable":8192,"lent":0,"model":"a"},
			{"gpu":"0000:0b:00.0","lendable":4096,"lent":0,"model":"b"}
		],
		"leases": [
			{"id":"a:{run:a}:1","gpu":"0000:0a:00.0","mib":0,"model":"a"},
			{"id":"b:{run:b}:2","gpu":"0000:0b:00.0","mib":0,"model":"b"}
		]
	}`), w.Body.String())
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramStateLeavesOutSlowEngine(t *testing.T) {
	f := &stateFixture{engines: map[string]*stateEngine{
		"a":    newStateEngine(t, http.StatusOK, stateBody("0000:0a:00.0", 8192, 0, "1", 0), 0),
		"slow": newStateEngine(t, http.StatusOK, stateBody("0000:0b:00.0", 4096, 0, "2", 0), scaled(40)),
	}}
	s := f.server(t, lendModels(map[string]bool{"a": true, "slow": true}), stateTimeout)

	start := time.Now()
	w := getVramState(s)
	elapsed := time.Since(start)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.GreaterOrEqual(t, elapsed, stateTimeout, "the answer waits for the timeout")
	assert.Less(t, elapsed, scaled(37), "the slow engine must not hold the answer past the timeout")
	assert.JSONEq(t, withRun(s, `{
		"gpus": [{"gpu":"0000:0a:00.0","lendable":8192,"lent":0,"model":"a"}],
		"leases": [{"id":"a:{run:a}:1","gpu":"0000:0a:00.0","mib":0,"model":"a"}]
	}`), w.Body.String())
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

// TestServer_VramStateRealRouter runs a lend model as a simple-responder
// process under the real router.
func TestServer_VramStateRealRouter(t *testing.T) {
	bin := simpleResponderBinary(t)
	port := freePort(t)
	yaml := fmt.Sprintf(`
healthCheckTimeout: 15
models:
  lend:
    cmd: %s -port %d -silent -vram-gpus 0000:0a:00.0,0B:00.0 -vram-lendable 8192
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

	// While the model is stopped the state is empty and starts nothing.
	w := getVramState(s)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, withRun(s, `{"gpus":[],"leases":[]}`), w.Body.String())
	assert.Never(t, func() bool { return len(s.local.RunningModels()) != 0 },
		200*time.Millisecond, 10*time.Millisecond, "a state read must not start the model")

	// Load the model with a normal request, then lend so a lease shows.
	w = httptest.NewRecorder()
	s.ServeHTTP(w, chatRequest("lend"))
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Equal(t, map[string]process.ProcessState{"lend": process.StateReady}, s.local.RunningModels())

	w = postLend(s, `{"gpu":"0B:00.0","mib":2048}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, leaseID(s, "lend", "1"), answerID(t, w), "the lend answer and the state carry the same run token")

	w = getVramState(s)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, withRun(s, `{
		"gpus": [
			{"gpu":"0000:0a:00.0","lendable":8192,"lent":0,"model":"lend"},
			{"gpu":"0000:0b:00.0","lendable":8192,"lent":2048,"model":"lend"}
		],
		"leases": [{"id":"lend:{run:lend}:1","gpu":"0000:0b:00.0","mib":2048,"model":"lend"}]
	}`), w.Body.String())
	assert.Equal(t, map[string]process.ProcessState{"lend": process.StateReady}, s.local.RunningModels())
}
