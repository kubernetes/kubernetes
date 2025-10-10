package spiffebundle

import "github.com/spiffe/go-spiffe/v2/spiffeid"

// Source represents a source of SPIFFE bundles keyed by trust domain.
type Source interface {
	// GetBundleForTrustDomain returns the SPIFFE bundle for the given trust
	// domain.
	GetBundleForTrustDomain(trustDomain spiffeid.TrustDomain) (*Bundle, error)
}
