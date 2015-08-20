// Package csr implements certificate requests for CFSSL.
package csr

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"net"
	"strings"

	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/log"
)

const (
	curveP256 = 256
	curveP384 = 384
	curveP521 = 521
)

// A Name contains the SubjectInfo fields.
type Name struct {
	C  string // Country
	ST string // State
	L  string // Locality
	O  string // OrganisationName
	OU string // OrganisationalUnitName
}

// A KeyRequest contains the algorithm and key size for a new private
// key.
type KeyRequest struct {
	Algo string `json:"algo"`
	Size int    `json:"size"`
}

// The DefaultKeyRequest is used when no key request data is provided
// in the request. This should be a safe default.
var DefaultKeyRequest = KeyRequest{
	Algo: "ecdsa",
	Size: curveP256,
}

// Generate generates a key as specified in the request. Currently,
// only ECDSA and RSA are supported.
func (kr *KeyRequest) Generate() (interface{}, error) {
	log.Debugf("generate key from request: algo=%s, size=%d", kr.Algo, kr.Size)
	switch kr.Algo {
	case "rsa":
		if kr.Size < 2048 {
			return nil, errors.New("RSA key is too weak")
		}
		return rsa.GenerateKey(rand.Reader, kr.Size)
	case "ecdsa":
		var curve elliptic.Curve
		switch kr.Size {
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
func (kr *KeyRequest) SigAlgo() x509.SignatureAlgorithm {
	switch kr.Algo {
	case "rsa":
		switch {
		case kr.Size >= 4096:
			return x509.SHA512WithRSA
		case kr.Size >= 3072:
			return x509.SHA384WithRSA
		case kr.Size >= 2048:
			return x509.SHA256WithRSA
		default:
			return x509.SHA1WithRSA
		}
	case "ecdsa":
		switch {
		case kr.Size == curveP521:
			return x509.ECDSAWithSHA512
		case kr.Size == curveP384:
			return x509.ECDSAWithSHA384
		case kr.Size == curveP256:
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
	CN         string
	Names      []Name      `json:"names"`
	Hosts      []string    `json:"hosts"`
	KeyRequest *KeyRequest `json:"key,omitempty"`
	CA         *CAConfig   `json:"ca,omitempty"`
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
		req.KeyRequest = &KeyRequest{
			Algo: DefaultKeyRequest.Algo,
			Size: DefaultKeyRequest.Size,
		}
	}

	log.Infof("generating key: %s-%d", req.KeyRequest.Algo, req.KeyRequest.Size)
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
