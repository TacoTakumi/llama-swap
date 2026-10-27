package server

import (
	"bytes"
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

// handleVramLend sends a lend to the ready lend model whose card list holds
// the requested card, and prefixes the lease id of a 200 answer with the
// model id. The lend goes through ServeIfReady only, so it never starts,
// queues or swaps a model. A lend that no ready lend model can serve answers
// 409 {"max_mib": 0}.
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

	modelID, ok := s.vramCards.Holder(r.Context(), card)
	if !ok {
		sendNoLend(w)
		return
	}

	// The engine gets the body exactly as the borrower sent it.
	fwd := r.Clone(r.Context())
	fwd.Body = io.NopCloser(bytes.NewReader(body))
	fwd.ContentLength = int64(len(body))
	rec := httptest.NewRecorder()
	if !s.local.ServeIfReady(modelID, rec, fwd) {
		// The model left the ready state after the lookup.
		sendNoLend(w)
		return
	}

	out := rec.Body.Bytes()
	if rec.Code == http.StatusOK {
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
}

// sendNoLend answers a lend that no ready lend model can serve.
func sendNoLend(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusConflict)
	w.Write([]byte(vramlend.NoLendBody))
}
