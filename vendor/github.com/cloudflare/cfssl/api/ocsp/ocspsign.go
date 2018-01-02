// Package ocsp implements the HTTP handler for the ocsp commands.
package ocsp

import (
	"net/http"

	"encoding/base64"
	"encoding/json"
	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/ocsp"
	"io/ioutil"
	"time"
)

// A Handler accepts requests with a certficate parameter
// (which should be PEM-encoded) and returns a signed ocsp
// response.
type Handler struct {
	signer ocsp.Signer
}

// NewHandler returns a new http.Handler that handles a ocspsign request.
func NewHandler(s ocsp.Signer) http.Handler {
	return &api.HTTPHandler{
		Handler: &Handler{
			signer: s,
		},
		Methods: []string{"POST"},
	}
}

// This type is meant to be unmarshalled from JSON
type jsonSignRequest struct {
	Certificate string `json:"certificate"`
	Status      string `json:"status"`
	Reason      int    `json:"reason,omitempty"`
	RevokedAt   string `json:"revoked_at,omitempty"`
}

// Handle responds to requests for a ocsp signature. It creates and signs
// a ocsp response for the provided certificate and status. If the status
// is revoked then it also adds reason and revoked_at. The response is
// base64 encoded.
func (h *Handler) Handle(w http.ResponseWriter, r *http.Request) error {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return err
	}
	r.Body.Close()

	// Default the status to good so it matches the cli
	req := &jsonSignRequest{
		Status: "good",
	}
	err = json.Unmarshal(body, req)
	if err != nil {
		return errors.NewBadRequestString("Unable to parse sign request")
	}

	cert, err := helpers.ParseCertificatePEM([]byte(req.Certificate))
	if err != nil {
		log.Error("Error from ParseCertificatePEM", err)
		return errors.NewBadRequestString("Malformed certificate")
	}

	signReq := ocsp.SignRequest{
		Certificate: cert,
		Status:      req.Status,
	}
	// We need to convert the time from being a string to a time.Time
	if req.Status == "revoked" {
		signReq.Reason = req.Reason
		// "now" is accepted and the default on the cli so default that here
		if req.RevokedAt == "" || req.RevokedAt == "now" {
			signReq.RevokedAt = time.Now()
		} else {
			signReq.RevokedAt, err = time.Parse("2006-01-02", req.RevokedAt)
			if err != nil {
				return errors.NewBadRequestString("Malformed revocation time")
			}
		}
	}

	resp, err := h.signer.Sign(signReq)
	if err != nil {
		return err
	}

	b64Resp := base64.StdEncoding.EncodeToString(resp)
	result := map[string]string{"ocspResponse": b64Resp}
	return api.SendResponse(w, result)
}
