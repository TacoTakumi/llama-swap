package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// badFixture is one ready lend model "a" on card 0000:0a:00.0. Both the
// local and the peer router count every request that would load a model.
func badFixture(t *testing.T, keys ...string) (*lendFixture, *Server) {
	t.Helper()
	f := &lendFixture{
		engines: map[string]*lendEngine{"a": newLendEngine(t, "0000:0a:00.0")},
		ready:   map[string]bool{"a": true},
	}
	peer := newStubRouter(nil, "")
	peer.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
		f.starts.Add(1)
		w.WriteHeader(http.StatusOK)
	}
	cfg := lendModels(map[string]bool{"a": true})
	cfg.RequiredAPIKeys = keys
	return f, newTestServerWithConfig(cfg, f.router(), peer)
}

func vramRequest(s *Server, method, path, body, key string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)
	return w
}

func assertNoEngine(t *testing.T, f *lendFixture, msg string) {
	t.Helper()
	calls, _ := f.engines["a"].calls()
	assert.Zero(t, calls, "%s: no engine may get a request", msg)
	assert.Zero(t, f.starts.Load(), "%s: no /vram request may go through the load path", msg)
}

func TestServer_VramBadLend(t *testing.T) {
	for _, body := range []string{
		``,
		`not json`,
		`[]`,
		`{"mib":2048}`,
		`{"gpu":7,"mib":2048}`,
		`{"gpu":null,"mib":2048}`,
		`{"gpu":"gpu0","mib":2048}`,
		`{"gpu":"0000:0a:00.0"}`,
		`{"gpu":"0000:0a:00.0","mib":0}`,
		`{"gpu":"0000:0a:00.0","mib":-1}`,
		`{"gpu":"0000:0a:00.0","mib":1.5}`,
		`{"gpu":"0000:0a:00.0","mib":"2048"}`,
		`{"gpu":"0000:0a:00.0","mib":true}`,
		`{"gpu":"0000:0a:00.0","mib":null}`,
	} {
		f, s := badFixture(t)
		w := vramRequest(s, http.MethodPost, "/vram/lend", body, "")
		assert.Equal(t, http.StatusBadRequest, w.Code, "body %q: %s", body, w.Body.String())
		assertNoEngine(t, f, body)
	}
}

func TestServer_VramBadReturn(t *testing.T) {
	for _, body := range []string{
		``,
		`not json`,
		`[]`,
		`{}`,
		`{"id":""}`,
		`{"id":null}`,
		`{"id":true}`,
		`{"id":{}}`,
	} {
		f, s := badFixture(t)
		w := vramRequest(s, http.MethodPost, "/vram/return", body, "")
		assert.Equal(t, http.StatusBadRequest, w.Code, "body %q: %s", body, w.Body.String())
		assertNoEngine(t, f, body)
	}
}

// A /vram path that no route serves must never fall through to a handler
// that could start a model, even with a body that names one.
func TestServer_VramBadPath(t *testing.T) {
	const body = `{"model":"a","gpu":"0000:0a:00.0","mib":2048,"id":"a:7"}`
	tests := []struct {
		method, path string
		want         int
	}{
		{http.MethodGet, "/vram", http.StatusNotFound},
		{http.MethodPost, "/vram", http.StatusNotFound},
		{http.MethodGet, "/vram/", http.StatusNotFound},
		{http.MethodPost, "/vram/", http.StatusNotFound},
		{http.MethodPost, "/vram/foo", http.StatusNotFound},
		{http.MethodPost, "/vram/lend/a", http.StatusNotFound},
		{http.MethodGet, "/vram/state/a", http.StatusNotFound},
		{http.MethodGet, "/vram/lend", http.StatusMethodNotAllowed},
		{http.MethodPut, "/vram/lend", http.StatusMethodNotAllowed},
		{http.MethodGet, "/vram/return", http.StatusMethodNotAllowed},
		{http.MethodDelete, "/vram/return", http.StatusMethodNotAllowed},
		{http.MethodPost, "/vram/state", http.StatusMethodNotAllowed},
		{http.MethodPut, "/vram/state", http.StatusMethodNotAllowed},
	}
	for _, tt := range tests {
		f, s := badFixture(t)
		w := vramRequest(s, tt.method, tt.path, body, "")
		msg := tt.method + " " + tt.path
		assert.Equal(t, tt.want, w.Code, "%s: %s", msg, w.Body.String())
		assertNoEngine(t, f, msg)
	}
}

func TestServer_VramAuth(t *testing.T) {
	routes := []struct {
		method, path, body, want string
	}{
		{http.MethodPost, "/vram/lend", `{"gpu":"0000:0a:00.0","mib":2048}`, `{"id":"a:{run:a}:7","gpu":"0000:0a:00.0","mib":2048}`},
		{http.MethodPost, "/vram/return", `{"id":"a:{run:a}:7"}`, `{"id":"a:{run:a}:7","gpu":"0000:0a:00.0","mib":2048}`},
		{http.MethodGet, "/vram/state", ``, `{"gpus":[{"gpu":"0000:0a:00.0","lendable":8192,"lent":0,"model":"a"}],"leases":[]}`},
	}
	for _, rt := range routes {
		msg := rt.method + " " + rt.path
		for _, key := range []string{"", "wrong"} {
			f, s := badFixture(t, "secret")
			w := vramRequest(s, rt.method, rt.path, withRun(s, rt.body), key)
			assert.Equal(t, http.StatusUnauthorized, w.Code, "%s key %q: %s", msg, key, w.Body.String())
			assertNoEngine(t, f, msg)
		}

		f, s := badFixture(t, "secret")
		w := vramRequest(s, rt.method, rt.path, withRun(s, rt.body), "secret")
		require.Equal(t, http.StatusOK, w.Code, "%s: %s", msg, w.Body.String())
		assert.JSONEq(t, withRun(s, rt.want), w.Body.String(), msg)
		assert.Zero(t, f.starts.Load(), "%s: no /vram request may go through the load path", msg)
	}
}

// Without API keys the same routes need no key.
func TestServer_VramAuthNoKeys(t *testing.T) {
	_, s := badFixture(t)
	w := vramRequest(s, http.MethodGet, "/vram/state", "", "")
	assert.Equal(t, http.StatusOK, w.Code, w.Body.String())
}
