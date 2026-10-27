package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"

	"github.com/mostlygeek/llama-swap/internal/swaputil"
	"github.com/mostlygeek/llama-swap/internal/vramlend"
)

// maxVramBodyBytes bounds a /vram request body. The VRAM Lend bodies are a
// few fields.
const maxVramBodyBytes = 64 << 10

// handleVramLend asks the ready lend models whose card lists hold the
// requested card, in model-id order, and prefixes the lease id of a 200
// answer with the model id. A 409 moves on to the next model; any other
// answer is passed through unchanged and ends the search. The lend goes
// through ServeIfReady only, so it never starts, queues or swaps a model.
// When every model answers 409, or none can serve the lend, the answer is
// 409 with the largest max_mib among the refusals.
func (s *Server) handleVramLend(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxVramBodyBytes))
	if err != nil {
		swaputil.SendResponse(w, r, http.StatusBadRequest, "vram: reading the request body: "+err.Error())
		return
	}
	card, err := vramlend.LendCard(body)
	if err != nil {
		swaputil.SendResponse(w, r, http.StatusBadRequest, "vram: "+err.Error())
		return
	}

	holders := s.vramCards.Holders(r.Context(), card)
	if len(holders) == 0 {
		sendNoLend(w)
		return
	}

	var maxMiB int64
	for _, modelID := range holders {
		// The engine gets the body exactly as the borrower sent it.
		fwd := r.Clone(r.Context())
		fwd.Body = io.NopCloser(bytes.NewReader(body))
		fwd.ContentLength = int64(len(body))
		rec := httptest.NewRecorder()
		if !s.local.ServeIfReady(modelID, rec, fwd) {
			// The model left the ready state after the lookup.
			continue
		}

		out := rec.Body.Bytes()
		switch rec.Code {
		case http.StatusConflict:
			maxMiB = max(maxMiB, vramlend.MaxMiB(out))
			continue
		case http.StatusOK:
			out, err = vramlend.PrefixLeaseID(modelID, out)
			if err != nil {
				s.logs.ProxyLogs.Warnf("<%s> vram: lend answer: %v", modelID, err)
				swaputil.SendResponse(w, r, http.StatusBadGateway, "vram: "+err.Error())
				return
			}
		}

		for k, v := range rec.Header() {
			w.Header()[k] = v
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(out)))
		w.WriteHeader(rec.Code)
		w.Write(out)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusConflict)
	w.Write(vramlend.NoLendAnswer(maxMiB))
}

// handleVramReturn sends a return of "<model id>:<engine id>" to that lend
// model with the engine id alone, and prefixes the id of a 200 answer again.
// Like a lend it goes through ServeIfReady only. A return whose id names no
// ready lend model answers 404, and no engine gets it.
func (s *Server) handleVramReturn(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxVramBodyBytes))
	if err != nil {
		swaputil.SendResponse(w, r, http.StatusBadRequest, "vram: reading the request body: "+err.Error())
		return
	}
	id, err := vramlend.ReturnID(body)
	if err != nil {
		swaputil.SendResponse(w, r, http.StatusBadRequest, "vram: "+err.Error())
		return
	}

	var models []string
	for mid, mc := range s.cfg.Models {
		if mc.VramLend {
			models = append(models, mid)
		}
	}
	modelID, engineID, ok := vramlend.SplitLeaseID(id, models)
	if !ok {
		swaputil.SendResponse(w, r, http.StatusNotFound, "vram: no lend model has lease "+id)
		return
	}
	fwdBody, err := vramlend.SetReturnID(body, engineID)
	if err != nil {
		swaputil.SendResponse(w, r, http.StatusBadRequest, "vram: "+err.Error())
		return
	}

	fwd := r.Clone(r.Context())
	fwd.Body = io.NopCloser(bytes.NewReader(fwdBody))
	fwd.ContentLength = int64(len(fwdBody))
	rec := httptest.NewRecorder()
	if !s.local.ServeIfReady(modelID, rec, fwd) {
		// A lease ends with the process that lent it.
		swaputil.SendResponse(w, r, http.StatusNotFound, "vram: model "+modelID+" is not running")
		return
	}

	out := rec.Body.Bytes()
	if rec.Code == http.StatusOK {
		// The engine has the memory back, so an answer without a usable id
		// still goes out as the engine sent it.
		if prefixed, err := vramlend.PrefixLeaseID(modelID, out); err == nil {
			out = prefixed
		} else {
			s.logs.ProxyLogs.Warnf("<%s> vram: return answer: %v", modelID, err)
		}
	}

	for k, v := range rec.Header() {
		w.Header()[k] = v
	}
	w.Header().Set("Content-Length", strconv.Itoa(len(out)))
	w.WriteHeader(rec.Code)
	w.Write(out)
}

// sendNoLend answers a lend that no ready lend model can serve.
func sendNoLend(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusConflict)
	w.Write([]byte(vramlend.NoLendBody))
}

// handleVramState answers the merged state of every ready lend model. The
// reads go through ServeIfReady only and run in parallel, each bounded by the
// state timeout. A model that is not ready, fails or runs out of time is left
// out, so the answer is always 200, with empty lists when no lend model is
// ready.
func (s *Server) handleVramState(w http.ResponseWriter, r *http.Request) {
	merged := vramlend.MergeStates(s.vramCards.ReadAll(r.Context()), func(modelID string, err error) {
		s.logs.ProxyLogs.Warnf("<%s> vram: state answer: %v", modelID, err)
	})

	out, err := json.Marshal(merged)
	if err != nil {
		swaputil.SendResponse(w, r, http.StatusInternalServerError, "vram: "+err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(out)))
	w.WriteHeader(http.StatusOK)
	w.Write(out)
}
