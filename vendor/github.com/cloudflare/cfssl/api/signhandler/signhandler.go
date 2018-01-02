// Package signhandler provides the handlers for signers.
package signhandler

import (
	"encoding/json"
	"io/ioutil"
	"math/big"
	"net/http"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/auth"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/signer"
)

// A Handler accepts requests with a hostname and certficate
// parameter (which should be PEM-encoded) and returns a new signed
// certificate. It includes upstream servers indexed by their
// profile name.
type Handler struct {
	signer signer.Signer
}

// NewHandlerFromSigner generates a new Handler directly from
// an existing signer.
func NewHandlerFromSigner(signer signer.Signer) (h *api.HTTPHandler, err error) {
	policy := signer.Policy()
	if policy == nil {
		err = errors.New(errors.PolicyError, errors.InvalidPolicy)
		return
	}

	// Sign will only respond for profiles that have no auth provider.
	// So if all of the profiles require authentication, we return an error.
	haveUnauth := (policy.Default.Provider == nil)
	for _, profile := range policy.Profiles {
		haveUnauth = haveUnauth || (profile.Provider == nil)
	}

	if !haveUnauth {
		err = errors.New(errors.PolicyError, errors.InvalidPolicy)
		return
	}

	return &api.HTTPHandler{
		Handler: &Handler{
			signer: signer,
		},
		Methods: []string{"POST"},
	}, nil
}

// This type is meant to be unmarshalled from JSON so that there can be a
// hostname field in the API
// TODO: Change the API such that the normal struct can be used.
type jsonSignRequest struct {
	Hostname string          `json:"hostname"`
	Hosts    []string        `json:"hosts"`
	Request  string          `json:"certificate_request"`
	Subject  *signer.Subject `json:"subject,omitempty"`
	Profile  string          `json:"profile"`
	Label    string          `json:"label"`
	Serial   *big.Int        `json:"serial,omitempty"`
}

func jsonReqToTrue(js jsonSignRequest) signer.SignRequest {
	sub := new(signer.Subject)
	if js.Subject == nil {
		sub = nil
	} else {
		// make a copy
		*sub = *js.Subject
	}

	if js.Hostname != "" {
		return signer.SignRequest{
			Hosts:   signer.SplitHosts(js.Hostname),
			Subject: sub,
			Request: js.Request,
			Profile: js.Profile,
			Label:   js.Label,
			Serial:  js.Serial,
		}
	}

	return signer.SignRequest{
		Hosts:   js.Hosts,
		Subject: sub,
		Request: js.Request,
		Profile: js.Profile,
		Label:   js.Label,
		Serial:  js.Serial,
	}
}

// Handle responds to requests for the CA to sign the certificate request
// present in the "certificate_request" parameter for the host named
// in the "hostname" parameter. The certificate should be PEM-encoded. If
// provided, subject information from the "subject" parameter will be used
// in place of the subject information from the CSR.
func (h *Handler) Handle(w http.ResponseWriter, r *http.Request) error {
	log.Info("signature request received")

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return err
	}
	r.Body.Close()

	var req jsonSignRequest

	err = json.Unmarshal(body, &req)
	if err != nil {
		return errors.NewBadRequestString("Unable to parse sign request")
	}

	signReq := jsonReqToTrue(req)

	if req.Request == "" {
		return errors.NewBadRequestString("missing parameter 'certificate_request'")
	}

	var cert []byte
	profile, err := signer.Profile(h.signer, req.Profile)
	if err != nil {
		return err
	}

	if profile.Provider != nil {
		log.Error("profile requires authentication")
		return errors.NewBadRequestString("authentication required")
	}

	cert, err = h.signer.Sign(signReq)
	if err != nil {
		log.Warningf("failed to sign request: %v", err)
		return err
	}

	result := map[string]string{"certificate": string(cert)}
	log.Info("wrote response")
	return api.SendResponse(w, result)
}

// An AuthHandler verifies and signs incoming signature requests.
type AuthHandler struct {
	signer signer.Signer
}

// NewAuthHandlerFromSigner creates a new AuthHandler from the signer
// that is passed in.
func NewAuthHandlerFromSigner(signer signer.Signer) (http.Handler, error) {
	policy := signer.Policy()
	if policy == nil {
		return nil, errors.New(errors.PolicyError, errors.InvalidPolicy)
	}

	if policy.Default == nil && policy.Profiles == nil {
		return nil, errors.New(errors.PolicyError, errors.InvalidPolicy)
	}

	// AuthSign will not respond for profiles that have no auth provider.
	// So if there are no profiles with auth providers in this policy,
	// we return an error.
	haveAuth := (policy.Default.Provider != nil)
	for _, profile := range policy.Profiles {
		if haveAuth {
			break
		}
		haveAuth = (profile.Provider != nil)
	}

	if !haveAuth {
		return nil, errors.New(errors.PolicyError, errors.InvalidPolicy)
	}

	return &api.HTTPHandler{
		Handler: &AuthHandler{
			signer: signer,
		},
		Methods: []string{"POST"},
	}, nil
}

// Handle receives the incoming request, validates it, and processes it.
func (h *AuthHandler) Handle(w http.ResponseWriter, r *http.Request) error {
	log.Info("signature request received")

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		log.Errorf("failed to read response body: %v", err)
		return err
	}
	r.Body.Close()

	var aReq auth.AuthenticatedRequest
	err = json.Unmarshal(body, &aReq)
	if err != nil {
		log.Errorf("failed to unmarshal authenticated request: %v", err)
		return errors.NewBadRequest(err)
	}

	var req jsonSignRequest
	err = json.Unmarshal(aReq.Request, &req)
	if err != nil {
		log.Errorf("failed to unmarshal request from authenticated request: %v", err)
		return errors.NewBadRequestString("Unable to parse authenticated sign request")
	}

	// Sanity checks to ensure that we have a valid policy. This
	// should have been checked in NewAuthHandler.
	policy := h.signer.Policy()
	if policy == nil {
		log.Critical("signer was initialised without a signing policy")
		return errors.NewBadRequestString("invalid policy")
	}

	profile, err := signer.Profile(h.signer, req.Profile)
	if err != nil {
		return err
	}

	if profile.Provider == nil {
		log.Error("profile has no authentication provider")
		return errors.NewBadRequestString("no authentication provider")
	}

	if !profile.Provider.Verify(&aReq) {
		log.Warning("received authenticated request with invalid token")
		return errors.NewBadRequestString("invalid token")
	}

	signReq := jsonReqToTrue(req)

	if signReq.Request == "" {
		return errors.NewBadRequestString("missing parameter 'certificate_request'")
	}

	cert, err := h.signer.Sign(signReq)
	if err != nil {
		log.Errorf("signature failed: %v", err)
		return err
	}

	result := map[string]string{"certificate": string(cert)}
	log.Info("wrote response")
	return api.SendResponse(w, result)
}
