package roots

import (
	"crypto/sha256"
	"crypto/x509"
	"errors"
	"io/ioutil"

	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/transport/core"
	"github.com/cloudflare/cfssl/transport/roots/system"
)

// Providers is a mapping of supported providers and the functions
// that can build them.
var Providers = map[string]func(map[string]string) ([]*x509.Certificate, error){
	"system": system.New,
	"cfssl":  NewCFSSL,
	"file":   TrustPEM,
}

// A TrustStore contains a pool of certificate that are trusted for a
// given TLS configuration.
type TrustStore struct {
	roots map[string]*x509.Certificate
}

// Pool returns a certificate pool containing the certificates
// loaded into the provider.
func (ts *TrustStore) Pool() *x509.CertPool {
	var pool = x509.NewCertPool()
	for _, cert := range ts.roots {
		pool.AddCert(cert)
	}
	return pool
}

// Certificates returns a slice of the loaded certificates.
func (ts *TrustStore) Certificates() []*x509.Certificate {
	var roots = make([]*x509.Certificate, 0, len(ts.roots))
	for _, cert := range ts.roots {
		roots = append(roots, cert)
	}
	return roots
}

func (ts *TrustStore) addCerts(certs []*x509.Certificate) {
	if ts.roots == nil {
		ts.roots = map[string]*x509.Certificate{}
	}

	for _, cert := range certs {
		digest := sha256.Sum256(cert.Raw)
		ts.roots[string(digest[:])] = cert
	}
}

// Trusted contains a store of trusted certificates.
type Trusted interface {
	// Certificates returns a slice containing the certificates
	// that are loaded into the provider.
	Certificates() []*x509.Certificate

	// AddCert adds a new certificate into the certificate pool.
	AddCert(cert *x509.Certificate)

	// AddPEM adds a one or more PEM-encoded certificates into the
	// certificate pool.
	AddPEM(cert []byte) bool
}

// New produces a new trusted root provider from a collection of
// roots. If there are no roots, the system roots will be used.
func New(rootDefs []*core.Root) (*TrustStore, error) {
	var err error

	var store = &TrustStore{}
	var roots []*x509.Certificate

	if len(rootDefs) == 0 {
		roots, err = system.New(nil)
		if err != nil {
			return nil, err
		}

		store.addCerts(roots)
		return store, nil
	}

	err = errors.New("transport: no supported root providers found")
	for _, root := range rootDefs {
		pfn, ok := Providers[root.Type]
		if ok {
			roots, err = pfn(root.Metadata)
			if err != nil {
				break
			}

			store.addCerts(roots)
		}
	}

	if err != nil {
		store = nil
	}
	return store, err
}

// TrustPEM takes a source file containing one or more certificates
// and adds them to the trust store.
func TrustPEM(metadata map[string]string) ([]*x509.Certificate, error) {
	sourceFile, ok := metadata["source"]
	if !ok {
		return nil, errors.New("transport: PEM source requires a source file")
	}

	in, err := ioutil.ReadFile(sourceFile)
	if err != nil {
		return nil, err
	}

	return helpers.ParseCertificatesPEM(in)
}
