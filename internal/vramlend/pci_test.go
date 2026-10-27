package vramlend

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVramLend_NormalizePCI(t *testing.T) {
	tests := map[string]string{
		"0000:0a:00.0":     "0000:0a:00.0", // canonical
		"0000:0A:00.0":     "0000:0a:00.0", // upper case
		"00000000:0A:00.0": "0000:0a:00.0", // nvidia-smi
		"0a:00.0":          "0000:0a:00.0", // lspci, no domain
		" 0a:00.0 ":        "0000:0a:00.0", // surrounding space
		"0001:81:1f.7":     "0001:81:1f.7", // non-zero domain
		"00000001:81:1F.7": "0001:81:1f.7", // non-zero 8-digit domain
	}
	for in, want := range tests {
		got, err := NormalizePCI(in)
		if assert.NoError(t, err, in) {
			assert.Equal(t, want, got, in)
		}
	}
}

func TestVramLend_NormalizePCIRejects(t *testing.T) {
	tests := []string{
		"",
		"   ",
		"0a:00",             // missing function
		"0a:00.",            // empty function
		"0000:0a:00",        // missing function
		"0000:0a:00.00",     // function too wide
		"0000:0a:0.0",       // device too narrow
		"0000:a:00.0",       // bus too narrow
		"0000:0a0:00.0",     // bus too wide
		"000:0a:00.0",       // domain too narrow
		"00000:0a:00.0",     // domain between widths
		"00010000:0a:00.0",  // 8-digit domain that does not fit 4 digits
		"0000:0g:00.0",      // non-hex bus
		"zzzz:0a:00.0",      // non-hex domain
		"0000:0a:00.x",      // non-hex function
		"0000:0a:00.0.1",    // extra part
		"0000:0000:0a:00.0", // extra part
		"0000:0a:00:00.0",   // extra part
		"+000:0a:00.0",      // sign is not hex
	}
	for _, in := range tests {
		got, err := NormalizePCI(in)
		assert.Error(t, err, in)
		assert.Empty(t, got, in)
	}
}
