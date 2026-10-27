package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newVramTestServer(t *testing.T, cfg vramConfig) *httptest.Server {
	t.Helper()
	gin.SetMode(gin.TestMode)
	r := gin.New()
	newVramResponder(cfg).register(r)
	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)
	return srv
}

func vramDo(t *testing.T, method, url, body string) (int, string) {
	t.Helper()
	req, err := http.NewRequest(method, url, strings.NewReader(body))
	require.NoError(t, err)
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	return resp.StatusCode, string(b)
}

func TestSimpleResponder_ParseVramGPUs(t *testing.T) {
	assert.Equal(t, []string{"0000:0a:00.0", "0000:01:00.0"}, parseVramGPUs(" 0000:0a:00.0, 0000:01:00.0,,"))
	empty := parseVramGPUs("")
	assert.NotNil(t, empty)
	assert.Empty(t, empty)
}

func TestSimpleResponder_VramStateListsCards(t *testing.T) {
	srv := newVramTestServer(t, vramConfig{
		GPUs:     []string{"0000:01:00.0", "0000:0a:00.0"},
		Lendable: 4096,
	})

	status, body := vramDo(t, http.MethodGet, srv.URL+"/vram/state", "")
	assert.Equal(t, http.StatusOK, status)
	assert.JSONEq(t, `{
		"gpus": [
			{"gpu": "0000:01:00.0", "lendable": 4096, "lent": 0},
			{"gpu": "0000:0a:00.0", "lendable": 4096, "lent": 0}
		],
		"leases": []
	}`, body)
}

func TestSimpleResponder_VramStateNoCards(t *testing.T) {
	srv := newVramTestServer(t, vramConfig{GPUs: parseVramGPUs("")})

	status, body := vramDo(t, http.MethodGet, srv.URL+"/vram/state", "")
	assert.Equal(t, http.StatusOK, status)
	assert.JSONEq(t, `{"gpus": [], "leases": []}`, body)
}

func TestSimpleResponder_VramLendAndReturn(t *testing.T) {
	srv := newVramTestServer(t, vramConfig{GPUs: []string{"0000:0a:00.0"}, Lendable: 8192})

	status, body := vramDo(t, http.MethodPost, srv.URL+"/vram/lend", `{"gpu":"0000:0a:00.0","mib":2048}`)
	assert.Equal(t, http.StatusOK, status)
	assert.JSONEq(t, `{"id":"1","gpu":"0000:0a:00.0","mib":2048}`, body)

	status, body = vramDo(t, http.MethodPost, srv.URL+"/vram/lend", `{"gpu":"0000:0a:00.0","mib":1024}`)
	assert.Equal(t, http.StatusOK, status)
	assert.JSONEq(t, `{"id":"2","gpu":"0000:0a:00.0","mib":1024}`, body)

	status, body = vramDo(t, http.MethodGet, srv.URL+"/vram/state", "")
	assert.Equal(t, http.StatusOK, status)
	assert.JSONEq(t, `{
		"gpus": [{"gpu": "0000:0a:00.0", "lendable": 8192, "lent": 3072}],
		"leases": [
			{"id": "1", "gpu": "0000:0a:00.0", "mib": 2048},
			{"id": "2", "gpu": "0000:0a:00.0", "mib": 1024}
		]
	}`, body)

	// a JSON number is accepted as the id
	status, body = vramDo(t, http.MethodPost, srv.URL+"/vram/return", `{"id":1}`)
	assert.Equal(t, http.StatusOK, status)
	assert.JSONEq(t, `{"id":"1","gpu":"0000:0a:00.0","mib":2048}`, body)

	status, _ = vramDo(t, http.MethodPost, srv.URL+"/vram/return", `{"id":"1"}`)
	assert.Equal(t, http.StatusNotFound, status)

	status, body = vramDo(t, http.MethodGet, srv.URL+"/vram/state", "")
	assert.Equal(t, http.StatusOK, status)
	assert.JSONEq(t, `{
		"gpus": [{"gpu": "0000:0a:00.0", "lendable": 8192, "lent": 1024}],
		"leases": [{"id": "2", "gpu": "0000:0a:00.0", "mib": 1024}]
	}`, body)
}

func TestSimpleResponder_VramBadBody(t *testing.T) {
	srv := newVramTestServer(t, vramConfig{GPUs: []string{"0000:0a:00.0"}})

	status, _ := vramDo(t, http.MethodPost, srv.URL+"/vram/lend", `not json`)
	assert.Equal(t, http.StatusBadRequest, status)

	status, _ = vramDo(t, http.MethodPost, srv.URL+"/vram/return", `not json`)
	assert.Equal(t, http.StatusBadRequest, status)
}

func TestSimpleResponder_VramConfiguredAnswers(t *testing.T) {
	srv := newVramTestServer(t, vramConfig{
		GPUs:   []string{"0000:0a:00.0"},
		State:  vramRouteConfig{Body: `{"gpus":[],"leases":[]}`},
		Lend:   vramRouteConfig{Status: http.StatusConflict, Body: `{"max_mib":0}`},
		Return: vramRouteConfig{Status: http.StatusServiceUnavailable},
	})

	status, body := vramDo(t, http.MethodGet, srv.URL+"/vram/state", "")
	assert.Equal(t, http.StatusOK, status)
	assert.JSONEq(t, `{"gpus":[],"leases":[]}`, body)

	status, body = vramDo(t, http.MethodPost, srv.URL+"/vram/lend", `{"gpu":"0000:0a:00.0","mib":2048}`)
	assert.Equal(t, http.StatusConflict, status)
	assert.JSONEq(t, `{"max_mib":0}`, body)

	status, body = vramDo(t, http.MethodPost, srv.URL+"/vram/return", `{"id":"1"}`)
	assert.Equal(t, http.StatusServiceUnavailable, status)
	assert.JSONEq(t, `{"error":"Service Unavailable"}`, body)
}

func TestSimpleResponder_VramDelay(t *testing.T) {
	delay := 100 * time.Millisecond
	srv := newVramTestServer(t, vramConfig{
		GPUs:   []string{"0000:0a:00.0"},
		State:  vramRouteConfig{Delay: delay},
		Lend:   vramRouteConfig{Delay: delay},
		Return: vramRouteConfig{Delay: delay},
	})

	cases := []struct{ method, path, body string }{
		{http.MethodGet, "/vram/state", ""},
		{http.MethodPost, "/vram/lend", `{"gpu":"0000:0a:00.0","mib":64}`},
		{http.MethodPost, "/vram/return", `{"id":"1"}`},
	}
	for _, tc := range cases {
		start := time.Now()
		status, _ := vramDo(t, tc.method, srv.URL+tc.path, tc.body)
		assert.Equal(t, http.StatusOK, status, tc.path)
		assert.GreaterOrEqual(t, time.Since(start), delay, tc.path)
	}
}

func TestSimpleResponder_VramDebugCounts(t *testing.T) {
	srv := newVramTestServer(t, vramConfig{GPUs: []string{"0000:0a:00.0"}})

	vramDo(t, http.MethodGet, srv.URL+"/vram/state", "")
	vramDo(t, http.MethodGet, srv.URL+"/vram/state", "")
	vramDo(t, http.MethodPost, srv.URL+"/vram/lend", `{"gpu":"0000:0a:00.0","mib":64}`)

	status, body := vramDo(t, http.MethodGet, srv.URL+"/debug/vram", "")
	require.Equal(t, http.StatusOK, status)

	var got map[string]vramRouteStats
	require.NoError(t, json.Unmarshal([]byte(body), &got))
	assert.Equal(t, map[string]vramRouteStats{
		"state":  {Count: 2, LastBody: ""},
		"lend":   {Count: 1, LastBody: `{"gpu":"0000:0a:00.0","mib":64}`},
		"return": {Count: 0, LastBody: ""},
	}, got)

	// the debug route does not count itself
	_, body = vramDo(t, http.MethodGet, srv.URL+"/debug/vram", "")
	require.NoError(t, json.Unmarshal([]byte(body), &got))
	assert.Equal(t, 2, got["state"].Count)
}
