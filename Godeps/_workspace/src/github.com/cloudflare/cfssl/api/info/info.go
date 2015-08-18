// Package info implements the HTTP handler for the info command.
package info

import (
	"encoding/json"
	"io/ioutil"
	"net/http"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/info"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/signer"
)

// Handler is a type that contains the root certificates for the CA,
// and serves information on them for clients that need the certificates.
type Handler struct {
	sign signer.Signer
}

// NewHandler creates a new handler to serve information on the CA's
// certificates, taking a signer to use.
func NewHandler(s signer.Signer) (http.Handler, error) {
	return &api.HTTPHandler{
		Handler: &Handler{
			sign: s,
		},
		Methods: []string{"POST"},
	}, nil
}

// Handle listens for incoming requests for CA information, and returns
// a list containing information on each root certificate.
func (h *Handler) Handle(w http.ResponseWriter, r *http.Request) error {
	req := new(info.Req)
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		log.Warningf("failed to read request body: %v", err)
		return errors.NewBadRequest(err)
	}
	err = json.Unmarshal(body, req)
	if err != nil {
		log.Warningf("failed to unmarshal request: %v", err)
		return errors.NewBadRequest(err)
	}

	resp, err := h.sign.Info(*req)
	if err != nil {
		return err
	}

	response := api.NewSuccessResponse(resp)
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	return enc.Encode(response)
}

// MultiHandler is a handler for providing the public certificates for
// a multi-root certificate authority. It takes a mapping of label to
// signer and a default label, and handles the standard information
// request as defined in the client package.
type MultiHandler struct {
	signers      map[string]signer.Signer
	defaultLabel string
}

// NewMultiHandler constructs a MultiHandler from a mapping of labels
// to signers and the default label.
func NewMultiHandler(signers map[string]signer.Signer, defaultLabel string) (http.Handler, error) {
	return &api.HTTPHandler{
		Handler: &MultiHandler{
			signers:      signers,
			defaultLabel: defaultLabel,
		},
		Methods: []string{"POST"},
	}, nil
}

// Handle accepts client information requests, and uses the label to
// look up the signer whose public certificate should be retrieved. If
// the label is empty, the default label is used.
func (h *MultiHandler) Handle(w http.ResponseWriter, r *http.Request) error {
	req := new(info.Req)
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		log.Warningf("failed to read request body: %v", err)
		return errors.NewBadRequest(err)
	}
	err = json.Unmarshal(body, req)
	if err != nil {
		log.Warningf("failed to unmarshal request: %v", err)
		return errors.NewBadRequest(err)
	}

	log.Debug("checking label")
	if req.Label == "" {
		req.Label = h.defaultLabel
	}

	if _, ok := h.signers[req.Label]; !ok {
		log.Warningf("request for invalid endpoint")
		return errors.NewBadRequestString("bad label")
	}

	log.Debug("getting info")
	resp, err := h.signers[req.Label].Info(*req)
	if err != nil {
		log.Infof("error getting certificate: %v", err)
		return err
	}

	response := api.NewSuccessResponse(resp)
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	return enc.Encode(response)
}
