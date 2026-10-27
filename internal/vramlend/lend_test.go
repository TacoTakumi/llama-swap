package vramlend

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVramLendRoute_LendCard(t *testing.T) {
	card, err := LendCard([]byte(`{"gpu":"00000000:0A:00.0","mib":2048}`))
	require.NoError(t, err)
	assert.Equal(t, "0000:0a:00.0", card)

	for _, body := range []string{``, `not json`, `[]`, `{"mib":2048}`, `{"gpu":7}`, `{"gpu":"gpu0"}`} {
		_, err := LendCard([]byte(body))
		assert.Error(t, err, "body %q", body)
	}
}

func TestVramLendRoute_PrefixLeaseID(t *testing.T) {
	tests := []struct {
		name, body, want string
	}{
		{"string id", `{"id":"7","gpu":"0000:0a:00.0","mib":8192}`, `{"id":"m:7","gpu":"0000:0a:00.0","mib":8192}`},
		{"number id", `{"id":7,"gpu":"0000:0a:00.0","mib":8192}`, `{"id":"m:7","gpu":"0000:0a:00.0","mib":8192}`},
		{"other fields kept", `{"gpu":"0a:00.0","id":"x","mib":64,"extra":{"a":1}}`, `{"gpu":"0a:00.0","id":"m:x","mib":64,"extra":{"a":1}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := PrefixLeaseID("m", []byte(tt.body))
			require.NoError(t, err)
			assert.JSONEq(t, tt.want, string(got))
		})
	}

	for _, body := range []string{``, `not json`, `{"gpu":"0000:0a:00.0"}`, `{"id":""}`, `{"id":true}`, `{"id":{}}`} {
		_, err := PrefixLeaseID("m", []byte(body))
		assert.Error(t, err, "body %q", body)
	}
}
