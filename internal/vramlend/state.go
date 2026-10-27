package vramlend

import (
	"errors"
	"maps"
	"slices"
)

// MergedState is llama-swap's GET /vram/state answer: the state of every
// ready lend model, with each entry naming the model it came from.
type MergedState struct {
	GPUs   []ModelGPUState `json:"gpus"`
	Leases []ModelLease    `json:"leases"`
}

// ModelGPUState is one card of one lend model. A card that two models use
// appears once for each.
type ModelGPUState struct {
	GPUState
	Model string `json:"model"`
}

// ModelLease is one open lease, its id prefixed with the model id and run
// token the same way a lend answer's is.
type ModelLease struct {
	Lease
	Model string `json:"model"`
}

// MergeStates merges the state of each lend model, keyed by model id, in
// model id order and in each engine's own order within a model. Card
// addresses are normalized, and lease ids get the LeaseID of the model and
// the state's Run. A model with a bad card address or lease id, or a state
// with no Run, is left out whole and passed to warn, which may be nil. The lists are never
// nil, so they encode as [] rather than null.
func MergeStates(states map[string]State, warn func(modelID string, err error)) MergedState {
	m := MergedState{GPUs: []ModelGPUState{}, Leases: []ModelLease{}}
	for _, model := range slices.Sorted(maps.Keys(states)) {
		gpus, leases, err := modelState(model, states[model])
		if err != nil {
			if warn != nil {
				warn(model, err)
			}
			continue
		}
		m.GPUs = append(m.GPUs, gpus...)
		m.Leases = append(m.Leases, leases...)
	}
	return m
}

// modelState returns one model's entries for the merged state.
func modelState(model string, st State) ([]ModelGPUState, []ModelLease, error) {
	gpus := make([]ModelGPUState, 0, len(st.GPUs))
	for _, g := range st.GPUs {
		card, err := NormalizePCI(g.GPU)
		if err != nil {
			return nil, nil, err
		}
		g.GPU = card
		gpus = append(gpus, ModelGPUState{GPUState: g, Model: model})
	}

	if st.Run == "" {
		return nil, nil, errors.New("/vram/state has no run token")
	}
	leases := make([]ModelLease, 0, len(st.Leases))
	for _, l := range st.Leases {
		if l.ID == "" {
			return nil, nil, errors.New("/vram/state lists a lease with an empty id")
		}
		card, err := NormalizePCI(l.GPU)
		if err != nil {
			return nil, nil, err
		}
		l.ID = LeaseID(model, st.Run, l.ID)
		l.GPU = card
		leases = append(leases, ModelLease{Lease: l, Model: model})
	}
	return gpus, leases, nil
}
