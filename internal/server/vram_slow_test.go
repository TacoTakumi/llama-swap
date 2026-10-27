package server

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The state timeout is scaled down for these tests, and the engine takes three
// times as long as that timeout to answer a lend or a return.
const (
	slowStateTimeout = 200 * time.Millisecond
	slowAnswerDelay  = 3 * slowStateTimeout
)

// slowEngine is a fake upstream that answers /vram/state at once with one
// card, and /vram/lend and /vram/return with a 200 after delay. It stops
// waiting when the request is cancelled, so a cut call shows as a failure.
type slowEngine struct {
	srv     *httptest.Server
	delay   time.Duration
	lends   atomic.Int32
	returns atomic.Int32
}

func newSlowEngine(t *testing.T, delay time.Duration) *slowEngine {
	t.Helper()
	e := &slowEngine{delay: delay}
	e.srv = httptest.NewServer(http.HandlerFunc(e.serve))
	t.Cleanup(e.srv.Close)
	return e
}

func (e *slowEngine) serve(w http.ResponseWriter, r *http.Request) {
	io.ReadAll(r.Body)
	w.Header().Set("Content-Type", "application/json")
	switch r.URL.Path {
	case "/vram/state":
		io.WriteString(w, `{"gpus":[{"gpu":"0000:0a:00.0","lendable":8192,"lent":0}],"leases":[]}`)
		return
	case "/vram/lend":
		e.lends.Add(1)
	case "/vram/return":
		e.returns.Add(1)
	default:
		w.WriteHeader(http.StatusNotFound)
		return
	}

	select {
	case <-time.After(e.delay):
	case <-r.Context().Done():
		return
	}
	io.WriteString(w, `{"id":"7","gpu":"0000:0a:00.0","mib":2048}`)
}

// slowServer puts a ready model "a" in front of e, with the state timeout set
// to slowStateTimeout and the card list of "a" already read. starts counts the
// requests that reach the load path.
func slowServer(t *testing.T, e *slowEngine, starts *atomic.Int32) *Server {
	t.Helper()
	stub := newStubRouter([]string{"a"}, "")
	stub.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
		starts.Add(1)
		w.WriteHeader(http.StatusOK)
	}
	stub.serveIfReady = func(modelID string, w http.ResponseWriter, r *http.Request) bool {
		if modelID != "a" {
			return false
		}
		target, _ := url.Parse(e.srv.URL)
		httputil.NewSingleHostReverseProxy(target).ServeHTTP(w, r)
		return true
	}
	s := newTestServerWithConfig(lendModels(map[string]bool{"a": true}), stub, newStubRouter(nil, ""))
	require.NotNil(t, s.vramCards)
	s.vramCards.Timeout = slowStateTimeout

	// The state answers at once, so the list is read well within the timeout.
	require.Equal(t, []string{"a"}, s.vramCards.Holders(context.Background(), "0000:0a:00.0"))
	return s
}

func TestServer_VramSlowEngineLend(t *testing.T) {
	e := newSlowEngine(t, slowAnswerDelay)
	var starts atomic.Int32
	s := slowServer(t, e, &starts)

	start := time.Now()
	w := postLend(s, `{"gpu":"0000:0a:00.0","mib":2048}`)
	elapsed := time.Since(start)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, withRun(s, `{"id":"a:{run:a}:7","gpu":"0000:0a:00.0","mib":2048}`), w.Body.String())
	assert.GreaterOrEqual(t, elapsed, slowAnswerDelay, "the answer is the engine's late one")
	assert.Equal(t, int32(1), e.lends.Load())
	assert.Zero(t, starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramSlowEngineReturn(t *testing.T) {
	e := newSlowEngine(t, slowAnswerDelay)
	var starts atomic.Int32
	s := slowServer(t, e, &starts)

	start := time.Now()
	w := postReturn(s, withRun(s, `{"id":"a:{run:a}:7"}`))
	elapsed := time.Since(start)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, withRun(s, `{"id":"a:{run:a}:7","gpu":"0000:0a:00.0","mib":2048}`), w.Body.String())
	assert.GreaterOrEqual(t, elapsed, slowAnswerDelay, "the answer is the engine's late one")
	assert.Equal(t, int32(1), e.returns.Load())
	assert.Zero(t, starts.Load(), "no /vram request may go through the load path")
}

// TestServer_VramSlowEngineStateStillTimesOut checks that the scaled timeout
// is in force: the same engine, slow on its state answer, is left out.
func TestServer_VramSlowEngineStateStillTimesOut(t *testing.T) {
	f := &stateFixture{engines: map[string]*stateEngine{
		"a": newStateEngine(t, http.StatusOK, stateBody("0000:0a:00.0", 8192, 0, "1", 0), slowAnswerDelay),
	}}
	s := f.server(t, lendModels(map[string]bool{"a": true}), slowStateTimeout)

	start := time.Now()
	w := getVramState(s)
	elapsed := time.Since(start)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, `{"gpus":[],"leases":[]}`, w.Body.String())
	assert.Less(t, elapsed, slowAnswerDelay, "the state read must stop at the %v timeout", slowStateTimeout)
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}
