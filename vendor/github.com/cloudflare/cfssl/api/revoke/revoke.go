// Package revoke implements the HTTP handler for the revoke command
package revoke

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/certdb"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/ocsp"

	stdocsp "golang.org/x/crypto/ocsp"
)

// A Handler accepts requests with a serial number parameter
// and revokes
type Handler struct {
	dbAccessor certdb.Accessor
	Signer     ocsp.Signer
}

// NewHandler returns a new http.Handler that handles a revoke request.
func NewHandler(dbAccessor certdb.Accessor) http.Handler {
	return &api.HTTPHandler{
		Handler: &Handler{
			dbAccessor: dbAccessor,
		},
		Methods: []string{"POST"},
	}
}

// NewOCSPHandler returns a new http.Handler that handles a revoke
// request and also generates an OCSP response
func NewOCSPHandler(dbAccessor certdb.Accessor, signer ocsp.Signer) http.Handler {
	return &api.HTTPHandler{
		Handler: &Handler{
			dbAccessor: dbAccessor,
			Signer:     signer,
		},
		Methods: []string{"POST"},
	}
}

// This type is meant to be unmarshalled from JSON
type jsonRevokeRequest struct {
	Serial string `json:"serial"`
	AKI    string `json:"authority_key_id"`
	Reason string `json:"reason"`
}

// Handle responds to revocation requests. It attempts to revoke
// a certificate with a given serial number
func (h *Handler) Handle(w http.ResponseWriter, r *http.Request) error {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return err
	}
	r.Body.Close()

	// Default the status to good so it matches the cli
	var req jsonRevokeRequest
	err = json.Unmarshal(body, &req)
	if err != nil {
		return errors.NewBadRequestString("Unable to parse revocation request")
	}

	if len(req.Serial) == 0 {
		return errors.NewBadRequestString("serial number is required but not provided")
	}

	var reasonCode int
	reasonCode, err = ocsp.ReasonStringToCode(req.Reason)
	if err != nil {
		return errors.NewBadRequestString("Invalid reason code")
	}

	err = h.dbAccessor.RevokeCertificate(req.Serial, req.AKI, reasonCode)
	if err != nil {
		return err
	}

	// If we were given a signer, try and generate an OCSP
	// response indicating revocation
	if h.Signer != nil {
		// TODO: should these errors be errors?
		// Grab the certificate from the database
		cr, err := h.dbAccessor.GetCertificate(req.Serial, req.AKI)
		if err != nil {
			return err
		}
		if len(cr) != 1 {
			return errors.NewBadRequestString("No unique certificate found")
		}

		cert, err := helpers.ParseCertificatePEM([]byte(cr[0].PEM))
		if err != nil {
			return errors.NewBadRequestString("Unable to parse certificates from PEM data")
		}

		sr := ocsp.SignRequest{
			Certificate: cert,
			Status:      "revoked",
			Reason:      reasonCode,
			RevokedAt:   time.Now().UTC(),
		}

		ocspResponse, err := h.Signer.Sign(sr)
		if err != nil {
			return err
		}

		// We parse the OCSP repsonse in order to get the next
		// update time/expiry time
		ocspParsed, err := stdocsp.ParseResponse(ocspResponse, nil)
		if err != nil {
			return err
		}

		ocspRecord := certdb.OCSPRecord{
			Serial: req.Serial,
			AKI:    req.AKI,
			Body:   string(ocspResponse),
			Expiry: ocspParsed.NextUpdate,
		}

		if err = h.dbAccessor.InsertOCSP(ocspRecord); err != nil {
			return err
		}
	}

	result := map[string]string{}
	return api.SendResponse(w, result)
}
