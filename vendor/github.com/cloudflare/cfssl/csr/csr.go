// Package csr implements certificate requests for CFSSL.
package csr

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"net"
	"net/mail"
	"strings"

	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
)

const (
	curveP256 = 256
	curveP384 = 384
	curveP521 = 521
)

// A Name contains the SubjectInfo fields.
type Name struct {
	C            string // Country
	ST           string // State
	L            string // Locality
	O            string // OrganisationName
	OU           string // OrganisationalUnitName
	SerialNumber string
}

// A KeyRequest is a generic request for a new key.
type KeyRequest interface {
	Algo() string
	Size() int
	Generate() (crypto.PrivateKey, error)
	SigAlgo() x509.SignatureAlgorithm
}

// A BasicKeyRequest contains the algorithm and key size for a new private key.
type BasicKeyRequest struct {
	A string `json:"algo"`
	S int    `json:"size"`
}

// NewBasicKeyRequest returns a default BasicKeyRequest.
func NewBasicKeyRequest() *BasicKeyRequest {
	return &BasicKeyRequest{"ecdsa", curveP256}
}

// Algo returns the requested key algorithm represented as a string.
func (kr *BasicKeyRequest) Algo() string {
	return kr.A
}

// Size returns the requested key size.
func (kr *BasicKeyRequest) Size() int {
	return kr.S
}

// Generate generates a key as specified in the request. Currently,
// only ECDSA and RSA are supported.
func (kr *BasicKeyRequest) Generate() (crypto.PrivateKey, error) {
	log.Debugf("generate key from request: algo=%s, size=%d", kr.Algo(), kr.Size())
	switch kr.Algo() {
	case "rsa":
		if kr.Size() < 2048 {
			return nil, errors.New("RSA key is too weak")
		}
		if kr.Size() > 8192 {
			return nil, errors.New("RSA key size too large")
		}
		return rsa.GenerateKey(rand.Reader, kr.Size())
	case "ecdsa":
		var curve elliptic.Curve
		switch kr.Size() {
		case curveP256:
			curve = elliptic.P256()
		case curveP384:
			curve = elliptic.P384()
		case curveP521:
			curve = elliptic.P521()
		default:
			return nil, errors.New("invalid curve")
		}
		return ecdsa.GenerateKey(curve, rand.Reader)
	default:
		return nil, errors.New("invalid algorithm")
	}
}

// SigAlgo returns an appropriate X.509 signature algorithm given the
// key request's type and size.
func (kr *BasicKeyRequest) SigAlgo() x509.SignatureAlgorithm {
	switch kr.Algo() {
	case "rsa":
		switch {
		case kr.Size() >= 4096:
			return x509.SHA512WithRSA
		case kr.Size() >= 3072:
			return x509.SHA384WithRSA
		case kr.Size() >= 2048:
			return x509.SHA256WithRSA
		default:
			return x509.SHA1WithRSA
		}
	case "ecdsa":
		switch kr.Size() {
		case curveP521:
			return x509.ECDSAWithSHA512
		case curveP384:
			return x509.ECDSAWithSHA384
		case curveP256:
			return x509.ECDSAWithSHA256
		default:
			return x509.ECDSAWithSHA1
		}
	default:
		return x509.UnknownSignatureAlgorithm
	}
}

// CAConfig is a section used in the requests initialising a new CA.
type CAConfig struct {
	PathLength int    `json:"pathlen"`
	Expiry     string `json:"expiry"`
}

// A CertificateRequest encapsulates the API interface to the
// certificate request functionality.
type CertificateRequest struct {
	CN           string
	Names        []Name     `json:"names"`
	Hosts        []string   `json:"hosts"`
	KeyRequest   KeyRequest `json:"key,omitempty"`
	CA           *CAConfig  `json:"ca,omitempty"`
	SerialNumber string     `json:"serialnumber,omitempty"`
}

// New returns a new, empty CertificateRequest with a
// BasicKeyRequest.
func New() *CertificateRequest {
	return &CertificateRequest{
		KeyRequest: NewBasicKeyRequest(),
	}
}

// appendIf appends to a if s is not an empty string.
func appendIf(s string, a *[]string) {
	if s != "" {
		*a = append(*a, s)
	}
}

// Name returns the PKIX name for the request.
func (cr *CertificateRequest) Name() pkix.Name {
	var name pkix.Name
	name.CommonName = cr.CN

	for _, n := range cr.Names {
		appendIf(n.C, &name.Country)
		appendIf(n.ST, &name.Province)
		appendIf(n.L, &name.Locality)
		appendIf(n.O, &name.Organization)
		appendIf(n.OU, &name.OrganizationalUnit)
	}
	name.SerialNumber = cr.SerialNumber
	return name
}

// ParseRequest takes a certificate request and generates a key and
// CSR from it. It does no validation -- caveat emptor. It will,
// however, fail if the key request is not valid (i.e., an unsupported
// curve or RSA key size). The lack of validation was specifically
// chosen to allow the end user to define a policy and validate the
// request appropriately before calling this function.
func ParseRequest(req *CertificateRequest) (csr, key []byte, err error) {
	log.Info("received CSR")
	if req.KeyRequest == nil {
		req.KeyRequest = NewBasicKeyRequest()
	}

	log.Infof("generating key: %s-%d", req.KeyRequest.Algo(), req.KeyRequest.Size())
	priv, err := req.KeyRequest.Generate()
	if err != nil {
		err = cferr.Wrap(cferr.PrivateKeyError, cferr.GenerationFailed, err)
		return
	}

	switch priv := priv.(type) {
	case *rsa.PrivateKey:
		key = x509.MarshalPKCS1PrivateKey(priv)
		block := pem.Block{
			Type:  "RSA PRIVATE KEY",
			Bytes: key,
		}
		key = pem.EncodeToMemory(&block)
	case *ecdsa.PrivateKey:
		key, err = x509.MarshalECPrivateKey(priv)
		if err != nil {
			err = cferr.Wrap(cferr.PrivateKeyError, cferr.Unknown, err)
			return
		}
		block := pem.Block{
			Type:  "EC PRIVATE KEY",
			Bytes: key,
		}
		key = pem.EncodeToMemory(&block)
	default:
		panic("Generate should have failed to produce a valid key.")
	}

	var tpl = x509.CertificateRequest{
		Subject:            req.Name(),
		SignatureAlgorithm: req.KeyRequest.SigAlgo(),
	}

	for i := range req.Hosts {
		if ip := net.ParseIP(req.Hosts[i]); ip != nil {
			tpl.IPAddresses = append(tpl.IPAddresses, ip)
		} else if email, err := mail.ParseAddress(req.Hosts[i]); err == nil && email != nil {
			tpl.EmailAddresses = append(tpl.EmailAddresses, req.Hosts[i])
		} else {
			tpl.DNSNames = append(tpl.DNSNames, req.Hosts[i])
		}
	}

	csr, err = x509.CreateCertificateRequest(rand.Reader, &tpl, priv)
	if err != nil {
		log.Errorf("failed to generate a CSR: %v", err)
		err = cferr.Wrap(cferr.CSRError, cferr.BadRequest, err)
		return
	}
	block := pem.Block{
		Type:  "CERTIFICATE REQUEST",
		Bytes: csr,
	}

	log.Info("encoded CSR")
	csr = pem.EncodeToMemory(&block)
	return
}

// ExtractCertificateRequest extracts a CertificateRequest from
// x509.Certificate. It is aimed to used for generating a new certificate
// from an existing certificate. For a root certificate, the CA expiry
// length is calculated as the duration between cert.NotAfter and cert.NotBefore.
func ExtractCertificateRequest(cert *x509.Certificate) *CertificateRequest {
	req := New()
	req.CN = cert.Subject.CommonName
	req.Names = getNames(cert.Subject)
	req.Hosts = getHosts(cert)
	req.SerialNumber = cert.Subject.SerialNumber

	if cert.IsCA {
		req.CA = new(CAConfig)
		// CA expiry length is calculated based on the input cert
		// issue date and expiry date.
		req.CA.Expiry = cert.NotAfter.Sub(cert.NotBefore).String()
		req.CA.PathLength = cert.MaxPathLen
	}

	return req
}

func getHosts(cert *x509.Certificate) []string {
	var hosts []string
	for _, ip := range cert.IPAddresses {
		hosts = append(hosts, ip.String())
	}
	for _, dns := range cert.DNSNames {
		hosts = append(hosts, dns)
	}
	for _, email := range cert.EmailAddresses {
		hosts = append(hosts, email)
	}

	return hosts
}

// getNames returns an array of Names from the certificate
// It onnly cares about Country, Organization, OrganizationalUnit, Locality, Province
func getNames(sub pkix.Name) []Name {
	// anonymous func for finding the max of a list of interger
	max := func(v1 int, vn ...int) (max int) {
		max = v1
		for i := 0; i < len(vn); i++ {
			if vn[i] > max {
				max = vn[i]
			}
		}
		return max
	}

	nc := len(sub.Country)
	norg := len(sub.Organization)
	nou := len(sub.OrganizationalUnit)
	nl := len(sub.Locality)
	np := len(sub.Province)

	n := max(nc, norg, nou, nl, np)

	names := make([]Name, n)
	for i := range names {
		if i < nc {
			names[i].C = sub.Country[i]
		}
		if i < norg {
			names[i].O = sub.Organization[i]
		}
		if i < nou {
			names[i].OU = sub.OrganizationalUnit[i]
		}
		if i < nl {
			names[i].L = sub.Locality[i]
		}
		if i < np {
			names[i].ST = sub.Province[i]
		}
	}
	return names
}

// A Generator is responsible for validating certificate requests.
type Generator struct {
	Validator func(*CertificateRequest) error
}

// ProcessRequest validates and processes the incoming request. It is
// a wrapper around a validator and the ParseRequest function.
func (g *Generator) ProcessRequest(req *CertificateRequest) (csr, key []byte, err error) {

	log.Info("generate received request")
	err = g.Validator(req)
	if err != nil {
		log.Warningf("invalid request: %v", err)
		return
	}

	csr, key, err = ParseRequest(req)
	if err != nil {
		return nil, nil, err
	}
	return
}

// IsNameEmpty returns true if the name has no identifying information in it.
func IsNameEmpty(n Name) bool {
	empty := func(s string) bool { return strings.TrimSpace(s) == "" }

	if empty(n.C) && empty(n.ST) && empty(n.L) && empty(n.O) && empty(n.OU) {
		return true
	}
	return false
}

// Regenerate uses the provided CSR as a template for signing a new
// CSR using priv.
func Regenerate(priv crypto.Signer, csr []byte) ([]byte, error) {
	req, extra, err := helpers.ParseCSR(csr)
	if err != nil {
		return nil, err
	} else if len(extra) > 0 {
		return nil, errors.New("csr: trailing data in certificate request")
	}

	return x509.CreateCertificateRequest(rand.Reader, req, priv)
}

// Generate creates a new CSR from a CertificateRequest structure and
// an existing key. The KeyRequest field is ignored.
func Generate(priv crypto.Signer, req *CertificateRequest) (csr []byte, err error) {
	sigAlgo := helpers.SignerAlgo(priv, crypto.SHA256)
	if sigAlgo == x509.UnknownSignatureAlgorithm {
		return nil, cferr.New(cferr.PrivateKeyError, cferr.Unavailable)
	}

	var tpl = x509.CertificateRequest{
		Subject:            req.Name(),
		SignatureAlgorithm: sigAlgo,
	}

	for i := range req.Hosts {
		if ip := net.ParseIP(req.Hosts[i]); ip != nil {
			tpl.IPAddresses = append(tpl.IPAddresses, ip)
		} else if email, err := mail.ParseAddress(req.Hosts[i]); err == nil && email != nil {
			tpl.EmailAddresses = append(tpl.EmailAddresses, email.Address)
		} else {
			tpl.DNSNames = append(tpl.DNSNames, req.Hosts[i])
		}
	}

	csr, err = x509.CreateCertificateRequest(rand.Reader, &tpl, priv)
	if err != nil {
		log.Errorf("failed to generate a CSR: %v", err)
		err = cferr.Wrap(cferr.CSRError, cferr.BadRequest, err)
		return
	}
	block := pem.Block{
		Type:  "CERTIFICATE REQUEST",
		Bytes: csr,
	}

	log.Info("encoded CSR")
	csr = pem.EncodeToMemory(&block)
	return
}
