package gossip

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"time"

	ct "github.com/google/certificate-transparency/go"
)

var defaultNumPollinationsToReturn = flag.Int("default_num_pollinations_to_return", 10,
	"Number of randomly selected STH pollination entries to return for sth-pollination requests.")

type clock interface {
	Now() time.Time
}

type realClock struct{}

func (realClock) Now() time.Time {
	return time.Now()
}

// SignatureVerifierMap is a map of SignatureVerifier by LogID
type SignatureVerifierMap map[ct.SHA256Hash]ct.SignatureVerifier

// Handler for the gossip HTTP requests.
type Handler struct {
	storage   *Storage
	verifiers SignatureVerifierMap
	clock     clock
}

func writeWrongMethodResponse(rw *http.ResponseWriter, allowed string) {
	(*rw).Header().Add("Allow", allowed)
	(*rw).WriteHeader(http.StatusMethodNotAllowed)
}

func writeErrorResponse(rw *http.ResponseWriter, status int, body string) {
	(*rw).WriteHeader(status)
	(*rw).Write([]byte(body))
}

// HandleSCTFeedback handles requests POSTed to .../sct-feedback.
// It attempts to store the provided SCT Feedback
func (h *Handler) HandleSCTFeedback(rw http.ResponseWriter, req *http.Request) {
	if req.Method != "POST" {
		writeWrongMethodResponse(&rw, "POST")
		return
	}

	decoder := json.NewDecoder(req.Body)
	var feedback SCTFeedback
	if err := decoder.Decode(&feedback); err != nil {
		writeErrorResponse(&rw, http.StatusBadRequest, fmt.Sprintf("Invalid SCT Feedback received: %v", err))
		return
	}

	// TODO(alcutter): 5.1.1 Validate leaf chains up to a trusted root
	// TODO(alcutter): 5.1.1/2 Verify each SCT is valid and from a known log, discard those which aren't
	// TODO(alcutter): 5.1.1/3 Discard leaves for domains other than ours.
	if err := h.storage.AddSCTFeedback(feedback); err != nil {
		writeErrorResponse(&rw, http.StatusInternalServerError, fmt.Sprintf("Unable to store feedback: %v", err))
		return
	}
	rw.WriteHeader(http.StatusOK)
}

// HandleSTHPollination handles requests POSTed to .../sth-pollination.
// It attempts to store the provided pollination info, and returns a random set of
// pollination data from the last 14 days (i.e. "fresh" by the definition of the gossip RFC.)
func (h *Handler) HandleSTHPollination(rw http.ResponseWriter, req *http.Request) {
	if req.Method != "POST" {
		writeWrongMethodResponse(&rw, "POST")
		return
	}

	decoder := json.NewDecoder(req.Body)
	var p STHPollination
	if err := decoder.Decode(&p); err != nil {
		writeErrorResponse(&rw, http.StatusBadRequest, fmt.Sprintf("Invalid STH Pollination received: %v", err))
		return
	}

	sthToKeep := make([]ct.SignedTreeHead, 0, len(p.STHs))
	for _, sth := range p.STHs {
		v, found := h.verifiers[sth.LogID]
		if !found {
			log.Printf("Pollination entry for unknown logID: %s", sth.LogID.Base64String())
			continue
		}
		if err := v.VerifySTHSignature(sth); err != nil {
			log.Printf("Failed to verify STH, dropping: %v", err)
			continue
		}
		sthToKeep = append(sthToKeep, sth)
	}
	p.STHs = sthToKeep

	err := h.storage.AddSTHPollination(p)
	if err != nil {
		writeErrorResponse(&rw, http.StatusInternalServerError, fmt.Sprintf("Couldn't store pollination: %v", err))
		return
	}

	freshTime := h.clock.Now().AddDate(0, 0, -14)
	rp, err := h.storage.GetRandomSTHPollination(freshTime, *defaultNumPollinationsToReturn)
	if err != nil {
		writeErrorResponse(&rw, http.StatusInternalServerError, fmt.Sprintf("Couldn't fetch pollination to return: %v", err))
		return
	}

	json := json.NewEncoder(rw)
	if err := json.Encode(*rp); err != nil {
		writeErrorResponse(&rw, http.StatusInternalServerError, fmt.Sprintf("Couldn't encode pollination to return: %v", err))
		return
	}
}

// NewHandler creates a new Handler object, taking a pointer a Storage object to
// use for storing and retrieving feedback and pollination data, and a
// SignatureVerifierMap for verifying signatures from known logs.
func NewHandler(s *Storage, v SignatureVerifierMap) Handler {
	return Handler{
		storage:   s,
		verifiers: v,
		clock:     realClock{},
	}
}

// NewHandler creates a new Handler object, taking a pointer a Storage object to
// use for storing and retrieving feedback and pollination data, and a
// SignatureVerifierMap for verifying signatures from known logs.
func newHandlerWithClock(s *Storage, v SignatureVerifierMap, c clock) Handler {
	return Handler{
		storage:   s,
		verifiers: v,
		clock:     c,
	}
}
