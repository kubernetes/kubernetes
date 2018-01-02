// Package kp describes transport key providers and provides a reference
// implementation.
//
// KeyProviders are used by clients and servers as a mechanism for
// providing keys and signing CSRs. It is a mechanism designed to
// allow switching out how private keys and their associated
// certificates are managed, such as supporting PKCS #11. The
// StandardProvider provides disk-backed PEM-encoded certificates and
// private keys. DiskFallback is a provider that will attempt to
// retrieve the certificate from a CA first, falling back to a
// disk-backed pair. This is useful for test a CA while providing a
// failover solution.
package kp

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"io/ioutil"
	"strings"

	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/transport/core"
)

const (
	curveP256 = 256
	curveP384 = 384
	curveP521 = 521
)

// A KeyProvider provides some mechanism for managing private keys and
// certificates. It is not required to store the crypto.Signer itself.
type KeyProvider interface {
	// Certificate returns the associated certificate, or nil if
	// one isn't ready.
	Certificate() *x509.Certificate

	// Given some metadata about a certificate request, the
	// provider should be able to generate a new CSR.
	CertificateRequest(*csr.CertificateRequest) ([]byte, error)

	// Check returns an error if the provider has an invalid setup.
	Check() error

	// Generate should trigger the creation of a new private
	// key. This will invalidate any certificates stored in the
	// key provider.
	Generate(algo string, size int) error

	// Load causes a private key and certificate associated with
	// this provider to be loaded into memory and be prepared for
	// use.
	Load() error

	// Persistent returns true if the provider keeps state on disk.
	Persistent() bool

	// Ready returns true if the provider has a key and
	// certificate.
	Ready() bool

	// SetCertificatePEM takes a PEM-encoded certificate and
	// associates it with this key provider.
	SetCertificatePEM([]byte) error

	// SignCSR allows a templated CSR to be signed.
	SignCSR(csr *x509.CertificateRequest) ([]byte, error)

	// Store should perform whatever actions are necessary such
	// that a call to Load later will reload the key and
	// certificate associated with this provider.
	Store() error

	// X509KeyPair returns a tls.Certficate. The returns
	// tls.Certificate should have a parsed Leaf certificate.
	X509KeyPair() (tls.Certificate, error)
}

// StandardPaths contains a path to a key file and certificate file.
type StandardPaths struct {
	KeyFile  string `json:"private_key"`
	CertFile string `json:"certificate"`
}

// StandardProvider provides unencrypted PEM-encoded certificates and
// private keys. If paths are provided, the key and certificate will
// be stored on disk.
type StandardProvider struct {
	Paths    StandardPaths `json:"paths"`
	internal struct {
		priv crypto.Signer
		cert *x509.Certificate

		// The PEM-encoded private key and certificate. This
		// is stored alongside the crypto.Signer and
		// x509.Certificate for convenience in marshaling and
		// calling tls.X509KeyPair directly.
		keyPEM  []byte
		certPEM []byte
	}
}

// NewStandardProvider sets up new StandardProvider from the
// information contained in an Identity.
func NewStandardProvider(id *core.Identity) (*StandardProvider, error) {
	if id == nil {
		return nil, errors.New("transport: the identity hasn't been initialised. Has it been loaded from disk?")
	}

	paths := id.Profiles["paths"]
	if paths == nil {
		return &StandardProvider{}, nil
	}

	sp := &StandardProvider{
		Paths: StandardPaths{
			KeyFile:  paths["private_key"],
			CertFile: paths["certificate"],
		},
	}

	err := sp.Check()
	if err != nil {
		return nil, err
	}

	return sp, nil
}

func (sp *StandardProvider) resetCert() {
	sp.internal.cert = nil
	sp.internal.certPEM = nil
}

func (sp *StandardProvider) resetKey() {
	sp.internal.priv = nil
	sp.internal.keyPEM = nil
}

var (
	// ErrMissingKeyPath is returned if the StandardProvider has
	// specified a certificate path but not a key path.
	ErrMissingKeyPath = errors.New("transport: standard provider is missing a private key path to accompany the certificate path")

	// ErrMissingCertPath is returned if the StandardProvider has
	// specified a private key path but not a certificate path.
	ErrMissingCertPath = errors.New("transport: standard provider is missing a certificate path to accompany the certificate path")
)

// Check ensures that the paths are valid for the provider.
func (sp *StandardProvider) Check() error {
	if sp.Paths.KeyFile == "" && sp.Paths.CertFile == "" {
		return nil
	}

	if sp.Paths.KeyFile == "" {
		return ErrMissingKeyPath
	}

	if sp.Paths.CertFile == "" {
		return ErrMissingCertPath
	}

	return nil
}

// Persistent returns true if the key and certificate will be stored
// on disk.
func (sp *StandardProvider) Persistent() bool {
	return sp.Paths.KeyFile != "" && sp.Paths.CertFile != ""
}

// Generate generates a new private key.
func (sp *StandardProvider) Generate(algo string, size int) (err error) {
	sp.resetKey()
	sp.resetCert()

	algo = strings.ToLower(algo)
	switch algo {
	case "rsa":
		var priv *rsa.PrivateKey
		if size < 2048 {
			return errors.New("transport: RSA keys must be at least 2048 bits")
		}

		priv, err = rsa.GenerateKey(rand.Reader, size)
		if err != nil {
			return err
		}

		keyPEM := x509.MarshalPKCS1PrivateKey(priv)
		p := &pem.Block{
			Type:  "RSA PRIVATE KEY",
			Bytes: keyPEM,
		}
		sp.internal.keyPEM = pem.EncodeToMemory(p)
		sp.internal.priv = priv
	case "ecdsa":
		var priv *ecdsa.PrivateKey
		var curve elliptic.Curve
		switch size {
		case curveP256:
			curve = elliptic.P256()
		case curveP384:
			curve = elliptic.P384()
		case curveP521:
			curve = elliptic.P521()
		default:
			return errors.New("transport: invalid elliptic curve key size; only 256-, 384-, and 521-bit keys are accepted")
		}

		priv, err = ecdsa.GenerateKey(curve, rand.Reader)
		if err != nil {
			return err
		}

		var keyPEM []byte
		keyPEM, err = x509.MarshalECPrivateKey(priv)
		if err != nil {
			return err
		}

		p := &pem.Block{
			Type:  "EC PRIVATE KEY",
			Bytes: keyPEM,
		}
		sp.internal.keyPEM = pem.EncodeToMemory(p)

		sp.internal.priv = priv
	default:
		return errors.New("transport: invalid key algorithm; only RSA and ECDSA are supported")
	}

	return nil
}

// Certificate returns the associated certificate, or nil if
// one isn't ready.
func (sp *StandardProvider) Certificate() *x509.Certificate {
	return sp.internal.cert
}

// CertificateRequest takes some metadata about a certificate request,
// and attempts to produce a certificate signing request suitable for
// sending to a certificate authority.
func (sp *StandardProvider) CertificateRequest(req *csr.CertificateRequest) ([]byte, error) {
	return csr.Generate(sp.internal.priv, req)
}

// ErrCertificateUnavailable is returned when a key is available, but
// there is no accompanying certificate.
var ErrCertificateUnavailable = errors.New("transport: certificate unavailable")

// Load a private key and certificate from disk.
func (sp *StandardProvider) Load() (err error) {
	if !sp.Persistent() {
		return
	}

	var clearKey = true
	defer func() {
		if err != nil {
			if clearKey {
				sp.resetKey()
			}
			sp.resetCert()
		}
	}()

	sp.internal.keyPEM, err = ioutil.ReadFile(sp.Paths.KeyFile)
	if err != nil {
		return
	}

	sp.internal.priv, err = helpers.ParsePrivateKeyPEM(sp.internal.keyPEM)
	if err != nil {
		return
	}

	clearKey = false

	sp.internal.certPEM, err = ioutil.ReadFile(sp.Paths.CertFile)
	if err != nil {
		return ErrCertificateUnavailable
	}

	sp.internal.cert, err = helpers.ParseCertificatePEM(sp.internal.certPEM)
	if err != nil {
		err = errors.New("transport: invalid certificate")
		return
	}

	p, _ := pem.Decode(sp.internal.keyPEM)

	switch sp.internal.cert.PublicKey.(type) {
	case *rsa.PublicKey:
		if p.Type != "RSA PRIVATE KEY" {
			err = errors.New("transport: PEM type " + p.Type + " is invalid for an RSA key")
			return
		}
	case *ecdsa.PublicKey:
		if p.Type != "EC PRIVATE KEY" {
			err = errors.New("transport: PEM type " + p.Type + " is invalid for an ECDSA key")
			return
		}
	default:
		err = errors.New("transport: invalid public key type")
	}

	if err != nil {
		clearKey = true
		return
	}

	return nil
}

// Ready returns true if the provider has a key and certificate
// loaded. The certificate should be checked by the end user for
// validity.
func (sp *StandardProvider) Ready() bool {
	switch {
	case sp.internal.priv == nil:
		return false
	case sp.internal.cert == nil:
		return false
	case sp.internal.keyPEM == nil:
		return false
	case sp.internal.certPEM == nil:
		return false
	default:
		return true
	}
}

// SetCertificatePEM receives a PEM-encoded certificate and loads it
// into the provider.
func (sp *StandardProvider) SetCertificatePEM(certPEM []byte) error {
	cert, err := helpers.ParseCertificatePEM(certPEM)
	if err != nil {
		return errors.New("transport: invalid certificate")
	}

	sp.internal.certPEM = certPEM
	sp.internal.cert = cert
	return nil
}

// SignCSR takes a template certificate request and signs it.
func (sp *StandardProvider) SignCSR(tpl *x509.CertificateRequest) ([]byte, error) {
	return x509.CreateCertificateRequest(rand.Reader, tpl, sp.internal.priv)
}

// Store writes the key and certificate to disk, if necessary.
func (sp *StandardProvider) Store() error {
	if !sp.Ready() {
		return errors.New("transport: provider does not have a key and certificate")
	}

	err := ioutil.WriteFile(sp.Paths.CertFile, sp.internal.certPEM, 0644)
	if err != nil {
		return err
	}

	return ioutil.WriteFile(sp.Paths.KeyFile, sp.internal.keyPEM, 0600)
}

// X509KeyPair returns a tls.Certificate for the provider.
func (sp *StandardProvider) X509KeyPair() (tls.Certificate, error) {
	cert, err := tls.X509KeyPair(sp.internal.certPEM, sp.internal.keyPEM)
	if err != nil {
		return tls.Certificate{}, err
	}

	if cert.Leaf == nil {
		cert.Leaf, err = x509.ParseCertificate(cert.Certificate[0])
		if err != nil {
			return tls.Certificate{}, err
		}
	}
	return cert, nil
}
