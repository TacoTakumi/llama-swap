package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newAnsweringEngine is a lend engine that answers every lend with status
// and body. It serves /vram/state like newLendEngine.
func newAnsweringEngine(t *testing.T, status int, body string, gpus ...string) *lendEngine {
	t.Helper()
	e := &lendEngine{gpus: gpus}
	e.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/vram/lend" {
			e.serve(w, r)
			return
		}
		got, _ := io.ReadAll(r.Body)
		e.mu.Lock()
		e.vramCalls++
		e.lendBodies = append(e.lendBodies, string(got))
		e.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write([]byte(body))
	}))
	t.Cleanup(e.srv.Close)
	return e
}

// twoLenders is a fixture with lend models a and b, both ready on one card.
func twoLenders(a, b *lendEngine) (*lendFixture, *Server) {
	f := &lendFixture{
		engines: map[string]*lendEngine{"a": a, "b": b},
		ready:   map[string]bool{"a": true, "b": true},
	}
	return f, f.server(lendModels(map[string]bool{"a": true, "b": true}))
}

func lendCount(e *lendEngine) int {
	_, lends := e.calls()
	return len(lends)
}

const orderCard = "0000:0a:00.0"

func TestServer_VramLendOrderNextAfterConflict(t *testing.T) {
	f, s := twoLenders(
		newAnsweringEngine(t, http.StatusConflict, `{"max_mib":100}`, orderCard),
		newLendEngine(t, orderCard),
	)

	body := `{"gpu":"0a:00.0","mib":2048}`
	w := postLend(s, body)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, `{"id":"b:7","gpu":"0a:00.0","mib":2048}`, w.Body.String())
	_, aLends := f.engines["a"].calls()
	_, bLends := f.engines["b"].calls()
	assert.Equal(t, []string{body}, aLends, "the first model is asked first")
	assert.Equal(t, []string{body}, bLends, "the next model gets the body unchanged")
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramLendOrderAllConflict(t *testing.T) {
	f, s := twoLenders(
		newAnsweringEngine(t, http.StatusConflict, `{"max_mib":100}`, orderCard),
		newAnsweringEngine(t, http.StatusConflict, `{"max_mib":300}`, orderCard),
	)

	w := postLend(s, `{"gpu":"0a:00.0","mib":4096}`)

	assert.Equal(t, http.StatusConflict, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.JSONEq(t, `{"max_mib":300}`, w.Body.String())
	assert.Equal(t, 1, lendCount(f.engines["a"]))
	assert.Equal(t, 1, lendCount(f.engines["b"]))
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramLendOrderConflictWithoutMaxCountsAsZero(t *testing.T) {
	f, s := twoLenders(
		newAnsweringEngine(t, http.StatusConflict, `{"error":"busy"}`, orderCard),
		newAnsweringEngine(t, http.StatusConflict, `not json`, orderCard),
	)

	assertNoLend(t, postLend(s, `{"gpu":"0a:00.0","mib":4096}`))
	assert.Equal(t, 1, lendCount(f.engines["a"]))
	assert.Equal(t, 1, lendCount(f.engines["b"]))
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramLendOrderStopsOnOtherAnswer(t *testing.T) {
	f, s := twoLenders(
		newAnsweringEngine(t, http.StatusBadRequest, `{"error":"mib too small"}`, orderCard),
		newLendEngine(t, orderCard),
	)

	w := postLend(s, `{"gpu":"0a:00.0","mib":1}`)

	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Equal(t, `{"error":"mib too small"}`, w.Body.String())
	assert.Equal(t, 1, lendCount(f.engines["a"]))
	assert.Zero(t, lendCount(f.engines["b"]), "the search ends at the first answer that is not 409")
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramLendOrderSkipsModelThatLeftReady(t *testing.T) {
	f := &lendFixture{
		engines: map[string]*lendEngine{
			"a": newLendEngine(t, orderCard),
			"b": newLendEngine(t, orderCard),
		},
		ready: map[string]bool{"a": true, "b": true},
	}
	// a lists the card, then stops before the lend reaches it.
	stub := f.router()
	serve := stub.serveIfReady
	stub.serveIfReady = func(modelID string, w http.ResponseWriter, r *http.Request) bool {
		if modelID == "a" && r.URL.Path == "/vram/lend" {
			return false
		}
		return serve(modelID, w, r)
	}
	s := newTestServerWithConfig(lendModels(map[string]bool{"a": true, "b": true}), stub, newStubRouter(nil, ""))

	w := postLend(s, `{"gpu":"0a:00.0","mib":2048}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.JSONEq(t, `{"id":"b:7","gpu":"0a:00.0","mib":2048}`, w.Body.String())
	assert.Zero(t, lendCount(f.engines["a"]))
	assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
}

func TestServer_VramLendOrderByModelID(t *testing.T) {
	// The config is a map, so its order changes from run to run. Build the
	// server several times: every time, the lowest model id is asked first.
	for range 8 {
		f := &lendFixture{
			engines: map[string]*lendEngine{
				"zz": newLendEngine(t, orderCard),
				"aa": newLendEngine(t, orderCard),
				"mm": newLendEngine(t, orderCard),
			},
			ready: map[string]bool{"zz": true, "aa": true, "mm": true},
		}
		s := f.server(lendModels(map[string]bool{"zz": true, "aa": true, "mm": true}))

		w := postLend(s, `{"gpu":"0a:00.0","mib":2048}`)

		require.Equal(t, http.StatusOK, w.Code, w.Body.String())
		assert.JSONEq(t, `{"id":"aa:7","gpu":"0a:00.0","mib":2048}`, w.Body.String())
		assert.Zero(t, lendCount(f.engines["mm"]))
		assert.Zero(t, lendCount(f.engines["zz"]))
		assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
	}
}

func TestServer_VramLendPassthroughEngineError(t *testing.T) {
	for _, status := range []int{http.StatusServiceUnavailable, http.StatusInternalServerError} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			f, s := twoLenders(
				newAnsweringEngine(t, status, `{"error": "x"}`, orderCard),
				newLendEngine(t, orderCard),
			)

			w := postLend(s, `{"gpu":"0a:00.0","mib":2048}`)

			assert.Equal(t, status, w.Code)
			assert.Equal(t, `{"error": "x"}`, w.Body.String(), "the body reaches the client byte for byte")
			assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
			assert.Zero(t, lendCount(f.engines["b"]))
			assert.Zero(t, f.starts.Load(), "no /vram request may go through the load path")
		})
	}
}
