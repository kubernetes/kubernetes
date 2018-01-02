// Package initca implements the HTTP handler for the CA initialization command
package initca

import (
	"encoding/json"
	"io/ioutil"
	"net/http"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/initca"
	"github.com/cloudflare/cfssl/log"
)

// A NewCA contains a private key and certificate suitable for serving
// as the root key for a new certificate authority.
type NewCA struct {
	Key  string `json:"private_key"`
	Cert string `json:"certificate"`
}

// initialCAHandler is an HTTP handler that accepts a JSON blob in the
// same format as the CSR endpoint; this blob should contain the
// identity information for the CA's root key. This endpoint is not
// suitable for creating intermediate certificates.
func initialCAHandler(w http.ResponseWriter, r *http.Request) error {
	log.Info("setting up initial CA handler")
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		log.Warningf("failed to read request body: %v", err)
		return errors.NewBadRequest(err)
	}

	req := new(csr.CertificateRequest)
	req.KeyRequest = csr.NewBasicKeyRequest()
	err = json.Unmarshal(body, req)
	if err != nil {
		log.Warningf("failed to unmarshal request: %v", err)
		return errors.NewBadRequest(err)
	}

	cert, _, key, err := initca.New(req)
	if err != nil {
		log.Warningf("failed to initialise new CA: %v", err)
		return err
	}

	response := api.NewSuccessResponse(&NewCA{string(key), string(cert)})

	enc := json.NewEncoder(w)
	err = enc.Encode(response)
	return err
}

// NewHandler returns a new http.Handler that handles request to
// initialize a CA.
func NewHandler() http.Handler {
	return api.HTTPHandler{Handler: api.HandlerFunc(initialCAHandler), Methods: []string{"POST"}}
}
