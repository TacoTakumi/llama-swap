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
		id, model, engine string
	}{
		{"a:7", "a", "7"},
		{"a:b:7", "a:b", "7"},
		{"a:c:7", "a", "c:7"},
		{"qwen3:8b:12", "qwen3:8b", "12"},
	}
	for _, tt := range tests {
		t.Run(tt.id, func(t *testing.T) {
			model, engine, ok := SplitLeaseID(tt.id, models)
			require.True(t, ok)
			assert.Equal(t, tt.model, model)
			assert.Equal(t, tt.engine, engine)
		})
	}

	for _, id := range []string{"7", "a", "a:", "b:7", "qwen3:7", ":7", ""} {
		_, _, ok := SplitLeaseID(id, models)
		assert.False(t, ok, "id %q", id)
	}
}

func TestVramReturnRoute_SetReturnID(t *testing.T) {
	got, err := SetReturnID([]byte(`{"id":"m:7","why":"done"}`), "7")
	require.NoError(t, err)
	assert.JSONEq(t, `{"id":"7","why":"done"}`, string(got))
}
