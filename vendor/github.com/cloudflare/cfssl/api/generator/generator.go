// Package generator implements the HTTP handlers for certificate generation.
package generator

import (
	"crypto/md5"
	"crypto/sha1"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"net/http"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/bundler"
	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/universal"
)

const (
	// CSRNoHostMessage is used to alert the user to a certificate lacking a hosts field.
	CSRNoHostMessage = `This certificate lacks a "hosts" field. This makes it unsuitable for
websites. For more information see the Baseline Requirements for the Issuance and Management
of Publicly-Trusted Certificates, v.1.1.6, from the CA/Browser Forum (https://cabforum.org);
specifically, section 10.2.3 ("Information Requirements").`
	// NoBundlerMessage is used to alert the user that the server does not have a bundler initialized.
	NoBundlerMessage = `This request requires a bundler, but one is not initialized for the API server.`
)

// Sum contains digests for a certificate or certificate request.
type Sum struct {
	MD5  string `json:"md5"`
	SHA1 string `json:"sha-1"`
}

// Validator is a type of function that contains the logic for validating
// a certificate request.
type Validator func(*csr.CertificateRequest) error

// A CertRequest stores a PEM-encoded private key and corresponding
// CSR; this is returned from the CSR generation endpoint.
type CertRequest struct {
	Key  string         `json:"private_key"`
	CSR  string         `json:"certificate_request"`
	Sums map[string]Sum `json:"sums"`
}

// A Handler accepts JSON-encoded certificate requests and
// returns a new private key and certificate request.
type Handler struct {
	generator *csr.Generator
}

// NewHandler builds a new Handler from the
// validation function provided.
func NewHandler(validator Validator) (http.Handler, error) {
	log.Info("setting up key / CSR generator")
	return &api.HTTPHandler{
		Handler: &Handler{
			generator: &csr.Generator{Validator: validator},
		},
		Methods: []string{"POST"},
	}, nil
}

func computeSum(in []byte) (sum Sum, err error) {
	var data []byte
	p, _ := pem.Decode(in)
	if p == nil {
		err = errors.NewBadRequestString("not a CSR or certificate")
		return
	}

	switch p.Type {
	case "CERTIFICATE REQUEST":
		var req *x509.CertificateRequest
		req, err = x509.ParseCertificateRequest(p.Bytes)
		if err != nil {
			return
		}
		data = req.Raw
	case "CERTIFICATE":
		var cert *x509.Certificate
		cert, err = x509.ParseCertificate(p.Bytes)
		if err != nil {
			return
		}
		data = cert.Raw
	default:
		err = errors.NewBadRequestString("not a CSR or certificate")
		return
	}

	md5Sum := md5.Sum(data)
	sha1Sum := sha1.Sum(data)
	sum.MD5 = fmt.Sprintf("%X", md5Sum[:])
	sum.SHA1 = fmt.Sprintf("%X", sha1Sum[:])
	return
}

// Handle responds to requests for the CA to generate a new private
// key and certificate request on behalf of the client. The format for
// these requests is documented in the API documentation.
func (g *Handler) Handle(w http.ResponseWriter, r *http.Request) error {
	log.Info("request for CSR")
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		log.Warningf("failed to read request body: %v", err)
		return errors.NewBadRequest(err)
	}
	r.Body.Close()

	req := new(csr.CertificateRequest)
	req.KeyRequest = csr.NewBasicKeyRequest()
	err = json.Unmarshal(body, req)
	if err != nil {
		log.Warningf("failed to unmarshal request: %v", err)
		return errors.NewBadRequest(err)
	}

	if req.CA != nil {
		log.Warningf("request received with CA section")
		return errors.NewBadRequestString("ca section only permitted in initca")
	}

	csr, key, err := g.generator.ProcessRequest(req)
	if err != nil {
		log.Warningf("failed to process CSR: %v", err)
		// The validator returns a *cfssl/errors.HttpError
		return err
	}

	sum, err := computeSum(csr)
	if err != nil {
		return errors.NewBadRequest(err)
	}

	// Both key and csr are returned PEM-encoded.
	response := api.NewSuccessResponse(&CertRequest{
		Key:  string(key),
		CSR:  string(csr),
		Sums: map[string]Sum{"certificate_request": sum},
	})
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	err = enc.Encode(response)
	return err
}

// A CertGeneratorHandler accepts JSON-encoded certificate requests
// and returns a new private key and signed certificate; it handles
// sending the CSR to the server.
type CertGeneratorHandler struct {
	generator *csr.Generator
	bundler   *bundler.Bundler
	signer    signer.Signer
}

// NewCertGeneratorHandler builds a new handler for generating
// certificates directly from certificate requests; the validator covers
// the certificate request and the CA's key and certificate are used to
// sign the generated request. If remote is not an empty string, the
// handler will send signature requests to the CFSSL instance contained
// in remote.
func NewCertGeneratorHandler(validator Validator, caFile, caKeyFile string, policy *config.Signing) (http.Handler, error) {
	var err error
	log.Info("setting up new generator / signer")
	cg := new(CertGeneratorHandler)

	if policy == nil {
		policy = &config.Signing{
			Default:  config.DefaultConfig(),
			Profiles: nil,
		}
	}

	root := universal.Root{
		Config: map[string]string{
			"ca-file":     caFile,
			"ca-key-file": caKeyFile,
		},
	}
	if cg.signer, err = universal.NewSigner(root, policy); err != nil {
		log.Errorf("setting up signer failed: %v", err)
		return nil, err
	}

	cg.generator = &csr.Generator{Validator: validator}

	return api.HTTPHandler{Handler: cg, Methods: []string{"POST"}}, nil
}

// NewCertGeneratorHandlerFromSigner returns a handler directly from
// the signer and validation function.
func NewCertGeneratorHandlerFromSigner(validator Validator, signer signer.Signer) http.Handler {
	return api.HTTPHandler{
		Handler: &CertGeneratorHandler{
			generator: &csr.Generator{Validator: validator},
			signer:    signer,
		},
		Methods: []string{"POST"},
	}
}

// SetBundler allows injecting an optional Bundler into the CertGeneratorHandler.
func (cg *CertGeneratorHandler) SetBundler(caBundleFile, intBundleFile string) (err error) {
	cg.bundler, err = bundler.NewBundler(caBundleFile, intBundleFile)
	return err
}

type genSignRequest struct {
	Request *csr.CertificateRequest `json:"request"`
	Profile string                  `json:"profile"`
	Label   string                  `json:"label"`
	Bundle  bool                    `json:"bundle"`
}

// Handle responds to requests for the CA to generate a new private
// key and certificate on behalf of the client. The format for these
// requests is documented in the API documentation.
func (cg *CertGeneratorHandler) Handle(w http.ResponseWriter, r *http.Request) error {
	log.Info("request for CSR")

	req := new(genSignRequest)
	req.Request = csr.New()

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		log.Warningf("failed to read request body: %v", err)
		return errors.NewBadRequest(err)
	}
	r.Body.Close()

	err = json.Unmarshal(body, req)
	if err != nil {
		log.Warningf("failed to unmarshal request: %v", err)
		return errors.NewBadRequest(err)
	}

	if req.Request == nil {
		log.Warning("empty request received")
		return errors.NewBadRequestString("missing request section")
	}

	if req.Request.CA != nil {
		log.Warningf("request received with CA section")
		return errors.NewBadRequestString("ca section only permitted in initca")
	}

	csr, key, err := cg.generator.ProcessRequest(req.Request)
	if err != nil {
		log.Warningf("failed to process CSR: %v", err)
		// The validator returns a *cfssl/errors.HttpError
		return err
	}

	signReq := signer.SignRequest{
		Request: string(csr),
		Profile: req.Profile,
		Label:   req.Label,
	}

	certBytes, err := cg.signer.Sign(signReq)
	if err != nil {
		log.Warningf("failed to sign request: %v", err)
		return err
	}

	reqSum, err := computeSum(csr)
	if err != nil {
		return errors.NewBadRequest(err)
	}

	certSum, err := computeSum(certBytes)
	if err != nil {
		return errors.NewBadRequest(err)
	}

	result := map[string]interface{}{
		"private_key":         string(key),
		"certificate_request": string(csr),
		"certificate":         string(certBytes),
		"sums": map[string]Sum{
			"certificate_request": reqSum,
			"certificate":         certSum,
		},
	}

	if req.Bundle {
		if cg.bundler == nil {
			return api.SendResponseWithMessage(w, result, NoBundlerMessage,
				errors.New(errors.PolicyError, errors.InvalidRequest).ErrorCode)
		}

		bundle, err := cg.bundler.BundleFromPEMorDER(certBytes, nil, bundler.Optimal, "")
		if err != nil {
			return err
		}

		result["bundle"] = bundle
	}

	if len(req.Request.Hosts) == 0 {
		return api.SendResponseWithMessage(w, result, CSRNoHostMessage,
			errors.New(errors.PolicyError, errors.InvalidRequest).ErrorCode)
	}

	return api.SendResponse(w, result)
}

// CSRValidate does nothing and will never return an error. It exists because NewHandler
// requires a Validator as a parameter.
func CSRValidate(req *csr.CertificateRequest) error {
	return nil
}
