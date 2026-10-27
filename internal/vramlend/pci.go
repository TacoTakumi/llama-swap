// Package vramlend holds the logic for lending VRAM from engines that llama-swap runs.
package vramlend

import (
	"fmt"
	"strings"
)

// NormalizePCI returns a PCI bus address in the canonical form 0000:0a:00.0:
// lower case, a 4-digit domain, then bus:device.function. It accepts the
// 4-digit domain form, nvidia-smi's 8-digit domain form (00000000:0A:00.0)
// and lspci's form without a domain (0a:00.0).
func NormalizePCI(addr string) (string, error) {
	s := strings.ToLower(strings.TrimSpace(addr))

	devFn := strings.Split(s, ".")
	if len(devFn) != 2 {
		return "", fmt.Errorf("invalid PCI address %q: want [domain:]bus:device.function", addr)
	}
	parts := strings.Split(devFn[0], ":")

	domain := "0000"
	switch len(parts) {
	case 2:
	case 3:
		domain = parts[0]
		// nvidia-smi pads the domain to 8 digits; the high half must be zero
		if len(domain) == 8 {
			if domain[:4] != "0000" {
				return "", fmt.Errorf("invalid PCI address %q: domain does not fit 4 digits", addr)
			}
			domain = domain[4:]
		}
		parts = parts[1:]
	default:
		return "", fmt.Errorf("invalid PCI address %q: want [domain:]bus:device.function", addr)
	}

	fields := []struct {
		name, value string
		width       int
	}{
		{"domain", domain, 4},
		{"bus", parts[0], 2},
		{"device", parts[1], 2},
		{"function", devFn[1], 1},
	}
	for _, f := range fields {
		if len(f.value) != f.width || !isHex(f.value) {
			return "", fmt.Errorf("invalid PCI address %q: %s must be %d hex digits", addr, f.name, f.width)
		}
	}

	return fmt.Sprintf("%s:%s:%s.%s", domain, parts[0], parts[1], devFn[1]), nil
}

func isHex(s string) bool {
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}
