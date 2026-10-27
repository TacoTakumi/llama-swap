package vramlend

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// NoLendBody is the 409 answer to a lend that no ready lend model can serve.
const NoLendBody = `{"max_mib":0}`

// NoLendAnswer returns the 409 answer to a lend that every asked model
// refused, with the largest max_mib among their answers.
func NoLendAnswer(maxMiB int64) []byte {
	return []byte(`{"max_mib":` + strconv.FormatInt(maxMiB, 10) + `}`)
}

// MaxMiB returns the max_mib of an engine's 409 lend answer. A body without
// a readable, non-negative max_mib counts as 0.
func MaxMiB(body []byte) int64 {
	res := gjson.GetBytes(body, "max_mib")
	if res.Type != gjson.Number || !gjson.ValidBytes(body) {
		return 0
	}
	return max(res.Int(), 0)
}

// LendCard returns the normalized card that a POST /vram/lend body asks for.
// The body must also ask for a whole number of MiB above 0.
func LendCard(body []byte) (string, error) {
	if !gjson.ValidBytes(body) || !gjson.ParseBytes(body).IsObject() {
		return "", errors.New("body is not a JSON object")
	}
	mib := gjson.GetBytes(body, "mib")
	if n, err := strconv.ParseInt(mib.Raw, 10, 64); mib.Type != gjson.Number || err != nil || n <= 0 {
		return "", errors.New("mib must be a whole number above 0")
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

// ReturnID returns the id of a POST /vram/return body. The id may be a JSON
// string or number.
func ReturnID(body []byte) (string, error) {
	if !gjson.ValidBytes(body) || !gjson.ParseBytes(body).IsObject() {
		return "", errors.New("body is not a JSON object")
	}
	var id string
	switch res := gjson.GetBytes(body, "id"); res.Type {
	case gjson.String:
		id = res.Str
	case gjson.Number:
		id = res.Raw
	default:
		return "", errors.New("id must be a string or a number")
	}
	if id == "" {
		return "", errors.New("id is empty")
	}
	return id, nil
}

// SplitLeaseID splits a lease id that PrefixLeaseID made into the model id
// and the engine id. Model ids and engine ids may both contain colons, so
// the split is not at a fixed colon: the model is the longest of models
// that, followed by ":", starts id. It returns false when no model matches
// or the engine id is empty.
func SplitLeaseID(id string, models []string) (model, engineID string, ok bool) {
	for _, m := range models {
		if len(m) > len(model) && strings.HasPrefix(id, m+":") {
			model = m
		}
	}
	if model == "" || len(id) == len(model)+1 {
		return "", "", false
	}
	return model, id[len(model)+1:], true
}

// SetReturnID rewrites the id of a POST /vram/return body to engineID, the
// id the engine knows the lease by. Every other field is kept.
func SetReturnID(body []byte, engineID string) ([]byte, error) {
	out, err := sjson.SetBytes(body, "id", engineID)
	if err != nil {
		return nil, fmt.Errorf("rewriting the return id: %w", err)
	}
	return out, nil
}
