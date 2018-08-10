// Package crl implements the HTTP handler for the crl command.
package crl

import (
	"crypto"
	"crypto/x509"
	"net/http"
	"os"
	"time"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/certdb"
	"github.com/cloudflare/cfssl/crl"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
)

// A Handler accepts requests with a serial number parameter
// and revokes
type Handler struct {
	dbAccessor certdb.Accessor
	ca         *x509.Certificate
	key        crypto.Signer
}

// NewHandler returns a new http.Handler that handles a revoke request.
func NewHandler(dbAccessor certdb.Accessor, caPath string, caKeyPath string) (http.Handler, error) {
	ca, err := helpers.ReadBytes(caPath)
	if err != nil {
		return nil, err
	}

	caKey, err := helpers.ReadBytes(caKeyPath)
	if err != nil {
		return nil, errors.Wrap(errors.PrivateKeyError, errors.ReadFailed, err)
	}

	// Parse the PEM encoded certificate
	issuerCert, err := helpers.ParseCertificatePEM(ca)
	if err != nil {
		return nil, err
	}

	strPassword := os.Getenv("CFSSL_CA_PK_PASSWORD")
	password := []byte(strPassword)
	if strPassword == "" {
		password = nil
	}

	// Parse the key given
	key, err := helpers.ParsePrivateKeyPEMWithPassword(caKey, password)
	if err != nil {
		log.Debug("malformed private key %v", err)
		return nil, err
	}

	return &api.HTTPHandler{
		Handler: &Handler{
			dbAccessor: dbAccessor,
			ca:         issuerCert,
			key:        key,
		},
		Methods: []string{"GET"},
	}, nil
}

// Handle responds to revocation requests. It attempts to revoke
// a certificate with a given serial number
func (h *Handler) Handle(w http.ResponseWriter, r *http.Request) error {
	var newExpiryTime = 7 * helpers.OneDay

	certs, err := h.dbAccessor.GetRevokedAndUnexpiredCertificates()
	if err != nil {
		return err
	}

	queryExpiryTime := r.URL.Query().Get("expiry")
	if queryExpiryTime != "" {
		log.Infof("requested expiry time of %s", queryExpiryTime)
		newExpiryTime, err = time.ParseDuration(queryExpiryTime)
		if err != nil {
			return err
		}
	}

	result, err := crl.NewCRLFromDB(certs, h.ca, h.key, newExpiryTime)
	if err != nil {
		return err
	}

	return api.SendResponse(w, result)
}
