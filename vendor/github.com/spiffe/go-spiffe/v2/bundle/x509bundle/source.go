package x509bundle

import (
	"github.com/spiffe/go-spiffe/v2/spiffeid"
)

// Source represents a source of X.509 bundles keyed by trust domain.
type Source interface {
	// GetX509BundleForTrustDomain returns the X.509 bundle for the given trust
	// domain.
	GetX509BundleForTrustDomain(trustDomain spiffeid.TrustDomain) (*Bundle, error)
}
