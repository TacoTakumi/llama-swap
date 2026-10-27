package vramlend

import (
	"errors"
	"fmt"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// NoLendBody is the 409 answer to a lend that no ready lend model can serve.
const NoLendBody = `{"max_mib":0}`

// LendCard returns the normalized card that a POST /vram/lend body asks for.
func LendCard(body []byte) (string, error) {
	if !gjson.ValidBytes(body) || !gjson.ParseBytes(body).IsObject() {
		return "", errors.New("body is not a JSON object")
	}
	gpu := gjson.GetBytes(body, "gpu")
	if gpu.Type != gjson.String {
		return "", errors.New("gpu must be a PCI address string")
	}
	return NormalizePCI(gpu.Str)
}

// PrefixLeaseID rewrites the id of an engine's lease answer to
// "<modelID>:<engine id>", so a later return can be routed back to the model
// that lent it. Every other field is kept as the engine sent it. The engine's
// id may be a JSON string or number; the new id is always a string.
func PrefixLeaseID(modelID string, body []byte) ([]byte, error) {
	if !gjson.ValidBytes(body) || !gjson.ParseBytes(body).IsObject() {
		return nil, errors.New("lease answer is not a JSON object")
	}

	var id string
	switch res := gjson.GetBytes(body, "id"); res.Type {
	case gjson.String:
		id = res.Str
	case gjson.Number:
		id = res.Raw
	default:
		return nil, errors.New("lease answer has no string or number id")
	}
	if id == "" {
		return nil, errors.New("lease answer has an empty id")
	}

	out, err := sjson.SetBytes(body, "id", modelID+":"+id)
	if err != nil {
		return nil, fmt.Errorf("rewriting the lease id: %w", err)
	}
	return out, nil
}
