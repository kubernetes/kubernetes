/*

Package ocsp exposes OCSP signing functionality, much like the signer
package does for certificate signing.  It also provies a basic OCSP
responder stack for serving pre-signed OCSP responses.

*/
package ocsp

import (
	"bytes"
	"crypto"
	"crypto/x509"
	"crypto/x509/pkix"
	"io/ioutil"
	"strconv"
	"strings"
	"time"

	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
	"golang.org/x/crypto/ocsp"
)

// revocationReasonCodes is a map between string reason codes
// to integers as defined in RFC 5280
var revocationReasonCodes = map[string]int{
	"unspecified":          ocsp.Unspecified,
	"keycompromise":        ocsp.KeyCompromise,
	"cacompromise":         ocsp.CACompromise,
	"affiliationchanged":   ocsp.AffiliationChanged,
	"superseded":           ocsp.Superseded,
	"cessationofoperation": ocsp.CessationOfOperation,
	"certificatehold":      ocsp.CertificateHold,
	"removefromcrl":        ocsp.RemoveFromCRL,
	"privilegewithdrawn":   ocsp.PrivilegeWithdrawn,
	"aacompromise":         ocsp.AACompromise,
}

// StatusCode is a map between string statuses sent by cli/api
// to ocsp int statuses
var StatusCode = map[string]int{
	"good":    ocsp.Good,
	"revoked": ocsp.Revoked,
	"unknown": ocsp.Unknown,
}

// SignRequest represents the desired contents of a
// specific OCSP response.
type SignRequest struct {
	Certificate *x509.Certificate
	Status      string
	Reason      int
	RevokedAt   time.Time
	Extensions  []pkix.Extension
	// IssuerHash is the hashing function used to hash the issuer subject and public key
	// in the OCSP response. Valid values are crypto.SHA1, crypto.SHA256, crypto.SHA384,
	// and crypto.SHA512. If zero, the default is crypto.SHA1.
	IssuerHash crypto.Hash
	// If provided ThisUpdate will override the default usage of time.Now().Truncate(time.Hour)
	ThisUpdate *time.Time
	// If provided NextUpdate will override the default usage of ThisUpdate.Add(signerInterval)
	NextUpdate *time.Time
}

// Signer represents a general signer of OCSP responses.  It is
// responsible for populating all fields in the OCSP response that
// are not reflected in the SignRequest.
type Signer interface {
	Sign(req SignRequest) ([]byte, error)
}

// StandardSigner is the default concrete type of OCSP signer.
// It represents a single responder (represented by a key and certificate)
// speaking for a single issuer (certificate).  It is assumed that OCSP
// responses are issued at a regular interval, which is used to compute
// the nextUpdate value based on the current time.
type StandardSigner struct {
	issuer    *x509.Certificate
	responder *x509.Certificate
	key       crypto.Signer
	interval  time.Duration
}

// ReasonStringToCode tries to convert a reason string to an integer code
func ReasonStringToCode(reason string) (reasonCode int, err error) {
	// default to 0
	if reason == "" {
		return 0, nil
	}

	reasonCode, present := revocationReasonCodes[strings.ToLower(reason)]
	if !present {
		reasonCode, err = strconv.Atoi(reason)
		if err != nil {
			return
		}
		if reasonCode >= ocsp.AACompromise || reasonCode <= ocsp.Unspecified {
			return 0, cferr.New(cferr.OCSPError, cferr.InvalidStatus)
		}
	}

	return
}

// NewSignerFromFile reads the issuer cert, the responder cert and the responder key
// from PEM files, and takes an interval in seconds
func NewSignerFromFile(issuerFile, responderFile, keyFile string, interval time.Duration) (Signer, error) {
	log.Debug("Loading issuer cert: ", issuerFile)
	issuerBytes, err := helpers.ReadBytes(issuerFile)
	if err != nil {
		return nil, err
	}
	log.Debug("Loading responder cert: ", responderFile)
	responderBytes, err := ioutil.ReadFile(responderFile)
	if err != nil {
		return nil, err
	}
	log.Debug("Loading responder key: ", keyFile)
	keyBytes, err := ioutil.ReadFile(keyFile)
	if err != nil {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ReadFailed, err)
	}

	issuerCert, err := helpers.ParseCertificatePEM(issuerBytes)
	if err != nil {
		return nil, err
	}

	responderCert, err := helpers.ParseCertificatePEM(responderBytes)
	if err != nil {
		return nil, err
	}

	key, err := helpers.ParsePrivateKeyPEM(keyBytes)
	if err != nil {
		log.Debug("Malformed private key %v", err)
		return nil, err
	}

	return NewSigner(issuerCert, responderCert, key, interval)
}

// NewSigner simply constructs a new StandardSigner object from the inputs,
// taking the interval in seconds
func NewSigner(issuer, responder *x509.Certificate, key crypto.Signer, interval time.Duration) (Signer, error) {
	return &StandardSigner{
		issuer:    issuer,
		responder: responder,
		key:       key,
		interval:  interval,
	}, nil
}

// Sign is used with an OCSP signer to request the issuance of
// an OCSP response.
func (s StandardSigner) Sign(req SignRequest) ([]byte, error) {
	if req.Certificate == nil {
		return nil, cferr.New(cferr.OCSPError, cferr.ReadFailed)
	}

	// Verify that req.Certificate is issued under s.issuer
	if bytes.Compare(req.Certificate.RawIssuer, s.issuer.RawSubject) != 0 {
		return nil, cferr.New(cferr.OCSPError, cferr.IssuerMismatch)
	}

	err := req.Certificate.CheckSignatureFrom(s.issuer)
	if err != nil {
		return nil, cferr.Wrap(cferr.OCSPError, cferr.VerifyFailed, err)
	}

	var thisUpdate, nextUpdate time.Time
	if req.ThisUpdate != nil {
		thisUpdate = *req.ThisUpdate
	} else {
		// Round thisUpdate times down to the nearest hour
		thisUpdate = time.Now().Truncate(time.Hour)
	}
	if req.NextUpdate != nil {
		nextUpdate = *req.NextUpdate
	} else {
		nextUpdate = thisUpdate.Add(s.interval)
	}

	status, ok := StatusCode[req.Status]
	if !ok {
		return nil, cferr.New(cferr.OCSPError, cferr.InvalidStatus)
	}

	// If the OCSP responder is the same as the issuer, there is no need to
	// include any certificate in the OCSP response, which decreases the byte size
	// of OCSP responses dramatically.
	certificate := s.responder
	if s.issuer == s.responder || bytes.Equal(s.issuer.Raw, s.responder.Raw) {
		certificate = nil
	}

	template := ocsp.Response{
		Status:          status,
		SerialNumber:    req.Certificate.SerialNumber,
		ThisUpdate:      thisUpdate,
		NextUpdate:      nextUpdate,
		Certificate:     certificate,
		ExtraExtensions: req.Extensions,
		IssuerHash:      req.IssuerHash,
	}

	if status == ocsp.Revoked {
		template.RevokedAt = req.RevokedAt
		template.RevocationReason = req.Reason
	}

	return ocsp.CreateResponse(s.issuer, s.responder, template, s.key)
}
