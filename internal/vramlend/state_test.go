package vramlend

import (
	"encoding/json"
	"testing"
)

func mergedJSON(t *testing.T, m MergedState) string {
	t.Helper()
	out, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

func TestVramLend_StateMergeEmpty(t *testing.T) {
	var warned []string
	m := MergeStates(nil, func(modelID string, err error) { warned = append(warned, modelID) })

	if got, want := mergedJSON(t, m), `{"gpus":[],"leases":[]}`; got != want {
		t.Errorf("merged=%s want %s", got, want)
	}
	if len(warned) != 0 {
		t.Errorf("warned=%v want none", warned)
	}
}

func TestVramLend_StateMergeSharedCard(t *testing.T) {
	states := map[string]State{
		"b": {
			GPUs:   []GPUState{{GPU: "0A:00.0", Lendable: 1024, Lent: 2048}},
			Leases: []Lease{{ID: "3", GPU: "0a:00.0", MiB: 2048}},
			Run:    "rb",
		},
		"a": {
			GPUs: []GPUState{
				{GPU: "00000000:0A:00.0", Lendable: 8192, Lent: 4096},
				{GPU: "0000:01:00.0"},
			},
			Leases: []Lease{
				{ID: "7", GPU: "0000:0a:00.0", MiB: 4096},
				{ID: "8", GPU: "0000:0a:00.0", MiB: 0},
			},
			Run: "ra",
		},
	}
	m := MergeStates(states, nil)

	want := `{"gpus":[` +
		`{"gpu":"0000:0a:00.0","lendable":8192,"lent":4096,"model":"a"},` +
		`{"gpu":"0000:01:00.0","lendable":0,"lent":0,"model":"a"},` +
		`{"gpu":"0000:0a:00.0","lendable":1024,"lent":2048,"model":"b"}],` +
		`"leases":[` +
		`{"id":"a:ra:7","gpu":"0000:0a:00.0","mib":4096,"model":"a"},` +
		`{"id":"a:ra:8","gpu":"0000:0a:00.0","mib":0,"model":"a"},` +
		`{"id":"b:rb:3","gpu":"0000:0a:00.0","mib":2048,"model":"b"}]}`
	if got := mergedJSON(t, m); got != want {
		t.Errorf("merged=%s\nwant  %s", got, want)
	}
}

func TestVramLend_StateMergeLeavesOutBadModel(t *testing.T) {
	good := State{
		GPUs:   []GPUState{{GPU: "0000:0a:00.0", Lendable: 8192}},
		Leases: []Lease{{ID: "1", GPU: "0000:0a:00.0", MiB: 64}},
		Run:    "rg",
	}
	for name, bad := range map[string]State{
		"bad gpu address":   {GPUs: []GPUState{{GPU: "gpu0"}}, Run: "rx"},
		"bad lease address": {GPUs: []GPUState{{GPU: "0000:0b:00.0"}}, Leases: []Lease{{ID: "1", GPU: "gpu0"}}, Run: "rx"},
		"empty lease id":    {GPUs: []GPUState{{GPU: "0000:0b:00.0"}}, Leases: []Lease{{GPU: "0000:0b:00.0"}}, Run: "rx"},
		"no run token":      {GPUs: []GPUState{{GPU: "0000:0b:00.0"}}, Leases: []Lease{{ID: "1", GPU: "0000:0b:00.0"}}},
	} {
		t.Run(name, func(t *testing.T) {
			var warned []string
			m := MergeStates(map[string]State{"bad": bad, "good": good},
				func(modelID string, err error) { warned = append(warned, modelID) })

			want := `{"gpus":[{"gpu":"0000:0a:00.0","lendable":8192,"lent":0,"model":"good"}],` +
				`"leases":[{"id":"good:rg:1","gpu":"0000:0a:00.0","mib":64,"model":"good"}]}`
			if got := mergedJSON(t, m); got != want {
				t.Errorf("merged=%s\nwant  %s", got, want)
			}
			if len(warned) != 1 || warned[0] != "bad" {
				t.Errorf("warned=%v want [bad]", warned)
			}
		})
	}
}
