package vramlend

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVramReturnRoute_ReturnID(t *testing.T) {
	tests := []struct {
		name, body, want string
	}{
		{"string id", `{"id":"m:7"}`, "m:7"},
		{"number id", `{"id":7}`, "7"},
		{"other fields", `{"id":"m:x","why":"done"}`, "m:x"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ReturnID([]byte(tt.body))
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}

	for _, body := range []string{``, `not json`, `[]`, `{}`, `{"id":""}`, `{"id":null}`, `{"id":true}`, `{"id":{}}`} {
		_, err := ReturnID([]byte(body))
		assert.Error(t, err, "body %q", body)
	}
}

func TestVramReturnRoute_SplitLeaseID(t *testing.T) {
	models := []string{"a", "a:b", "qwen3:8b"}
	tests := []struct {
		id, model, run, engine string
	}{
		{"a:r1:7", "a", "r1", "7"},
		{"a:b:r1:7", "a:b", "r1", "7"},
		// the model is the longest match; the run ends at the next colon
		{"a:c:7", "a", "c", "7"},
		// the engine id is opaque and keeps its colons
		{"a:r1:x:7", "a", "r1", "x:7"},
		{"qwen3:8b:0f3a9c2e:12", "qwen3:8b", "0f3a9c2e", "12"},
	}
	for _, tt := range tests {
		t.Run(tt.id, func(t *testing.T) {
			model, run, engine, ok := SplitLeaseID(tt.id, models)
			require.True(t, ok)
			assert.Equal(t, tt.model, model)
			assert.Equal(t, tt.run, run)
			assert.Equal(t, tt.engine, engine)
		})
	}

	for _, id := range []string{
		"7", "a", "a:", "b:r1:7", "qwen3:r1:7", ":r1:7", "",
		// no run token
		"a:7", "a:b:7", "qwen3:8b:12",
		// empty run token or engine id
		"a::7", "a:r1:", "a::",
	} {
		_, _, _, ok := SplitLeaseID(id, models)
		assert.False(t, ok, "id %q", id)
	}
}

func TestVramReturnRoute_LeaseIDRoundTrip(t *testing.T) {
	models := []string{"a", "a:b"}
	for _, engineID := range []string{"1", "x:7", "0000:0a:00.0#3"} {
		model, run, engine, ok := SplitLeaseID(LeaseID("a:b", "0f3a9c2e", engineID), models)
		require.True(t, ok, "engine id %q", engineID)
		assert.Equal(t, "a:b", model)
		assert.Equal(t, "0f3a9c2e", run)
		assert.Equal(t, engineID, engine)
	}
}

func TestVramReturnRoute_SetReturnID(t *testing.T) {
	got, err := SetReturnID([]byte(`{"id":"m:7","why":"done"}`), "7")
	require.NoError(t, err)
	assert.JSONEq(t, `{"id":"7","why":"done"}`, string(got))
}
