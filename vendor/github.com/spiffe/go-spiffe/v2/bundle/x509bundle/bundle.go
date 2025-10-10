package x509bundle

import (
	"crypto/x509"
	"io"
	"os"
	"sync"

	"github.com/spiffe/go-spiffe/v2/internal/pemutil"
	"github.com/spiffe/go-spiffe/v2/internal/x509util"
	"github.com/spiffe/go-spiffe/v2/spiffeid"
	"github.com/zeebo/errs"
)

var x509bundleErr = errs.Class("x509bundle")

// Bundle is a collection of trusted X.509 authorities for a trust domain.
type Bundle struct {
	trustDomain spiffeid.TrustDomain

	mtx             sync.RWMutex
	x509Authorities []*x509.Certificate
}

// New creates a new bundle.
func New(trustDomain spiffeid.TrustDomain) *Bundle {
	return &Bundle{
		trustDomain: trustDomain,
	}
}

// FromX509Authorities creates a bundle from X.509 certificates.
func FromX509Authorities(trustDomain spiffeid.TrustDomain, authorities []*x509.Certificate) *Bundle {
	return &Bundle{
		trustDomain:     trustDomain,
		x509Authorities: x509util.CopyX509Authorities(authorities),
	}
}

// Load loads a bundle from a file on disk. The file must contain PEM-encoded
// certificate blocks.
func Load(trustDomain spiffeid.TrustDomain, path string) (*Bundle, error) {
	fileBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, x509bundleErr.New("unable to load X.509 bundle file: %w", err)
	}

	return Parse(trustDomain, fileBytes)
}

// Read decodes a bundle from a reader. The contents must be PEM-encoded
// certificate blocks.
func Read(trustDomain spiffeid.TrustDomain, r io.Reader) (*Bundle, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, x509bundleErr.New("unable to read X.509 bundle: %v", err)
	}

	return Parse(trustDomain, b)
}

// Parse parses a bundle from bytes. The data must be PEM-encoded certificate
// blocks.
func Parse(trustDomain spiffeid.TrustDomain, b []byte) (*Bundle, error) {
	bundle := New(trustDomain)
	if len(b) == 0 {
		return bundle, nil
	}

	certs, err := pemutil.ParseCertificates(b)
	if err != nil {
		return nil, x509bundleErr.New("cannot parse certificate: %v", err)
	}
	for _, cert := range certs {
		bundle.AddX509Authority(cert)
	}
	return bundle, nil
}

// ParseRaw parses a bundle from bytes. The certificate must be ASN.1 DER (concatenated
// with no intermediate padding if there are more than one certificate)
func ParseRaw(trustDomain spiffeid.TrustDomain, b []byte) (*Bundle, error) {
	bundle := New(trustDomain)
	if len(b) == 0 {
		return bundle, nil
	}

	certs, err := x509.ParseCertificates(b)
	if err != nil {
		return nil, x509bundleErr.New("cannot parse certificate: %v", err)
	}
	for _, cert := range certs {
		bundle.AddX509Authority(cert)
	}
	return bundle, nil
}

// TrustDomain returns the trust domain that the bundle belongs to.
func (b *Bundle) TrustDomain() spiffeid.TrustDomain {
	return b.trustDomain
}

// X509Authorities returns the X.509 x509Authorities in the bundle.
func (b *Bundle) X509Authorities() []*x509.Certificate {
	b.mtx.RLock()
	defer b.mtx.RUnlock()
	return x509util.CopyX509Authorities(b.x509Authorities)
}

// AddX509Authority adds an X.509 authority to the bundle. If the authority already
// exists in the bundle, the contents of the bundle will remain unchanged.
func (b *Bundle) AddX509Authority(x509Authority *x509.Certificate) {
	b.mtx.Lock()
	defer b.mtx.Unlock()

	for _, r := range b.x509Authorities {
		if r.Equal(x509Authority) {
			return
		}
	}

	b.x509Authorities = append(b.x509Authorities, x509Authority)
}

// RemoveX509Authority removes an X.509 authority from the bundle.
func (b *Bundle) RemoveX509Authority(x509Authority *x509.Certificate) {
	b.mtx.Lock()
	defer b.mtx.Unlock()

	for i, r := range b.x509Authorities {
		if r.Equal(x509Authority) {
			// remove element from slice
			b.x509Authorities = append(b.x509Authorities[:i], b.x509Authorities[i+1:]...)
			return
		}
	}
}

// HasX509Authority checks if the given X.509 authority exists in the bundle.
func (b *Bundle) HasX509Authority(x509Authority *x509.Certificate) bool {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	for _, r := range b.x509Authorities {
		if r.Equal(x509Authority) {
			return true
		}
	}
	return false
}

// SetX509Authorities sets the X.509 authorities in the bundle.
func (b *Bundle) SetX509Authorities(x509Authorities []*x509.Certificate) {
	b.mtx.Lock()
	defer b.mtx.Unlock()

	b.x509Authorities = x509util.CopyX509Authorities(x509Authorities)
}

// Empty returns true if the bundle has no X.509 x509Authorities.
func (b *Bundle) Empty() bool {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	return len(b.x509Authorities) == 0
}

// Marshal marshals the X.509 bundle into PEM-encoded certificate blocks.
func (b *Bundle) Marshal() ([]byte, error) {
	b.mtx.RLock()
	defer b.mtx.RUnlock()
	return pemutil.EncodeCertificates(b.x509Authorities), nil
}

// Equal compares the bundle for equality against the given bundle.
func (b *Bundle) Equal(other *Bundle) bool {
	if b == nil || other == nil {
		return b == other
	}

	return b.trustDomain == other.trustDomain &&
		x509util.CertsEqual(b.x509Authorities, other.x509Authorities)
}

// Clone clones the bundle.
func (b *Bundle) Clone() *Bundle {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	return FromX509Authorities(b.trustDomain, b.x509Authorities)
}

// GetX509BundleForTrustDomain returns the X.509 bundle for the given trust
// domain. It implements the Source interface. An error will be
// returned if the trust domain does not match that of the bundle.
func (b *Bundle) GetX509BundleForTrustDomain(trustDomain spiffeid.TrustDomain) (*Bundle, error) {
	if b.trustDomain != trustDomain {
		return nil, x509bundleErr.New("no X.509 bundle found for trust domain: %q", trustDomain)
	}

	return b, nil
}
