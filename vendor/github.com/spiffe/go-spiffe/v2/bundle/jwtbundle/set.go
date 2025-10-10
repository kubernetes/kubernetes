package jwtbundle

import (
	"sort"
	"sync"

	"github.com/spiffe/go-spiffe/v2/spiffeid"
)

// Set is a set of bundles, keyed by trust domain.
type Set struct {
	mtx     sync.RWMutex
	bundles map[spiffeid.TrustDomain]*Bundle
}

// NewSet creates a new set initialized with the given bundles.
func NewSet(bundles ...*Bundle) *Set {
	bundlesMap := make(map[spiffeid.TrustDomain]*Bundle)

	for _, b := range bundles {
		if b != nil {
			bundlesMap[b.trustDomain] = b
		}
	}

	return &Set{
		bundles: bundlesMap,
	}
}

// Add adds a new bundle into the set. If a bundle already exists for the
// trust domain, the existing bundle is replaced.
func (s *Set) Add(bundle *Bundle) {
	s.mtx.Lock()
	defer s.mtx.Unlock()

	if bundle != nil {
		s.bundles[bundle.trustDomain] = bundle
	}
}

// Remove removes the bundle for the given trust domain.
func (s *Set) Remove(trustDomain spiffeid.TrustDomain) {
	s.mtx.Lock()
	defer s.mtx.Unlock()

	delete(s.bundles, trustDomain)
}

// Has returns true if there is a bundle for the given trust domain.
func (s *Set) Has(trustDomain spiffeid.TrustDomain) bool {
	s.mtx.RLock()
	defer s.mtx.RUnlock()

	_, ok := s.bundles[trustDomain]
	return ok
}

// Get returns a bundle for the given trust domain. If the bundle is in the set
// it is returned and the boolean is true. Otherwise, the returned value is
// nil and the boolean is false.
func (s *Set) Get(trustDomain spiffeid.TrustDomain) (*Bundle, bool) {
	s.mtx.RLock()
	defer s.mtx.RUnlock()

	bundle, ok := s.bundles[trustDomain]
	return bundle, ok
}

// Bundles returns the bundles in the set sorted by trust domain.
func (s *Set) Bundles() []*Bundle {
	s.mtx.RLock()
	defer s.mtx.RUnlock()

	out := make([]*Bundle, 0, len(s.bundles))
	for _, bundle := range s.bundles {
		out = append(out, bundle)
	}
	sort.Slice(out, func(a, b int) bool {
		return out[a].TrustDomain().Compare(out[b].TrustDomain()) < 0
	})
	return out
}

// Len returns the number of bundles in the set.
func (s *Set) Len() int {
	s.mtx.RLock()
	defer s.mtx.RUnlock()

	return len(s.bundles)
}

// GetJWTBundleForTrustDomain returns the JWT bundle for the given trust
// domain. It implements the Source interface.
func (s *Set) GetJWTBundleForTrustDomain(trustDomain spiffeid.TrustDomain) (*Bundle, error) {
	s.mtx.RLock()
	defer s.mtx.RUnlock()

	bundle, ok := s.bundles[trustDomain]
	if !ok {
		return nil, jwtbundleErr.New("no JWT bundle for trust domain %q", trustDomain)
	}

	return bundle, nil
}
