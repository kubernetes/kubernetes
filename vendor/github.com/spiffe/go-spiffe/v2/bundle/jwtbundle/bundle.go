package jwtbundle

import (
	"crypto"
	"encoding/json"
	"errors"
	"io"
	"os"
	"sync"

	"github.com/go-jose/go-jose/v4"
	"github.com/spiffe/go-spiffe/v2/internal/jwtutil"
	"github.com/spiffe/go-spiffe/v2/spiffeid"
	"github.com/zeebo/errs"
)

var jwtbundleErr = errs.Class("jwtbundle")

// Bundle is a collection of trusted JWT authorities for a trust domain.
type Bundle struct {
	trustDomain spiffeid.TrustDomain

	mtx            sync.RWMutex
	jwtAuthorities map[string]crypto.PublicKey
}

// New creates a new bundle.
func New(trustDomain spiffeid.TrustDomain) *Bundle {
	return &Bundle{
		trustDomain:    trustDomain,
		jwtAuthorities: make(map[string]crypto.PublicKey),
	}
}

// FromJWTAuthorities creates a new bundle from JWT authorities
func FromJWTAuthorities(trustDomain spiffeid.TrustDomain, jwtAuthorities map[string]crypto.PublicKey) *Bundle {
	return &Bundle{
		trustDomain:    trustDomain,
		jwtAuthorities: jwtutil.CopyJWTAuthorities(jwtAuthorities),
	}
}

// Load loads a bundle from a file on disk. The file must contain a standard RFC 7517 JWKS document.
func Load(trustDomain spiffeid.TrustDomain, path string) (*Bundle, error) {
	bundleBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, jwtbundleErr.New("unable to read JWT bundle: %w", err)
	}

	return Parse(trustDomain, bundleBytes)
}

// Read decodes a bundle from a reader. The contents must contain a standard RFC 7517 JWKS document.
func Read(trustDomain spiffeid.TrustDomain, r io.Reader) (*Bundle, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, jwtbundleErr.New("unable to read: %v", err)
	}

	return Parse(trustDomain, b)
}

// Parse parses a bundle from bytes. The data must be a standard RFC 7517 JWKS document.
func Parse(trustDomain spiffeid.TrustDomain, bundleBytes []byte) (*Bundle, error) {
	jwks := new(jose.JSONWebKeySet)
	if err := json.Unmarshal(bundleBytes, jwks); err != nil {
		return nil, jwtbundleErr.New("unable to parse JWKS: %v", err)
	}

	bundle := New(trustDomain)
	for i, key := range jwks.Keys {
		if err := bundle.AddJWTAuthority(key.KeyID, key.Key); err != nil {
			return nil, jwtbundleErr.New("error adding authority %d of JWKS: %v", i, errors.Unwrap(err))
		}
	}

	return bundle, nil
}

// TrustDomain returns the trust domain that the bundle belongs to.
func (b *Bundle) TrustDomain() spiffeid.TrustDomain {
	return b.trustDomain
}

// JWTAuthorities returns the JWT authorities in the bundle, keyed by key ID.
func (b *Bundle) JWTAuthorities() map[string]crypto.PublicKey {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	return jwtutil.CopyJWTAuthorities(b.jwtAuthorities)
}

// FindJWTAuthority finds the JWT authority with the given key ID from the bundle. If the authority
// is found, it is returned and the boolean is true. Otherwise, the returned
// value is nil and the boolean is false.
func (b *Bundle) FindJWTAuthority(keyID string) (crypto.PublicKey, bool) {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	if jwtAuthority, ok := b.jwtAuthorities[keyID]; ok {
		return jwtAuthority, true
	}
	return nil, false
}

// HasJWTAuthority returns true if the bundle has a JWT authority with the given key ID.
func (b *Bundle) HasJWTAuthority(keyID string) bool {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	_, ok := b.jwtAuthorities[keyID]
	return ok
}

// AddJWTAuthority adds a JWT authority to the bundle. If a JWT authority already exists
// under the given key ID, it is replaced. A key ID must be specified.
func (b *Bundle) AddJWTAuthority(keyID string, jwtAuthority crypto.PublicKey) error {
	if keyID == "" {
		return jwtbundleErr.New("keyID cannot be empty")
	}

	b.mtx.Lock()
	defer b.mtx.Unlock()

	b.jwtAuthorities[keyID] = jwtAuthority
	return nil
}

// RemoveJWTAuthority removes the JWT authority identified by the key ID from the bundle.
func (b *Bundle) RemoveJWTAuthority(keyID string) {
	b.mtx.Lock()
	defer b.mtx.Unlock()

	delete(b.jwtAuthorities, keyID)
}

// SetJWTAuthorities sets the JWT authorities in the bundle.
func (b *Bundle) SetJWTAuthorities(jwtAuthorities map[string]crypto.PublicKey) {
	b.mtx.Lock()
	defer b.mtx.Unlock()

	b.jwtAuthorities = jwtutil.CopyJWTAuthorities(jwtAuthorities)
}

// Empty returns true if the bundle has no JWT authorities.
func (b *Bundle) Empty() bool {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	return len(b.jwtAuthorities) == 0
}

// Marshal marshals the JWT bundle into a standard RFC 7517 JWKS document. The
// JWKS does not contain any SPIFFE-specific parameters.
func (b *Bundle) Marshal() ([]byte, error) {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	jwks := jose.JSONWebKeySet{}
	for keyID, jwtAuthority := range b.jwtAuthorities {
		jwks.Keys = append(jwks.Keys, jose.JSONWebKey{
			Key:   jwtAuthority,
			KeyID: keyID,
		})
	}

	return json.Marshal(jwks)
}

// Clone clones the bundle.
func (b *Bundle) Clone() *Bundle {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	return FromJWTAuthorities(b.trustDomain, b.jwtAuthorities)
}

// Equal compares the bundle for equality against the given bundle.
func (b *Bundle) Equal(other *Bundle) bool {
	if b == nil || other == nil {
		return b == other
	}

	return b.trustDomain == other.trustDomain &&
		jwtutil.JWTAuthoritiesEqual(b.jwtAuthorities, other.jwtAuthorities)
}

// GetJWTBundleForTrustDomain returns the JWT bundle for the given trust
// domain. It implements the Source interface. An error will be returned if
// the trust domain does not match that of the bundle.
func (b *Bundle) GetJWTBundleForTrustDomain(trustDomain spiffeid.TrustDomain) (*Bundle, error) {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	if b.trustDomain != trustDomain {
		return nil, jwtbundleErr.New("no JWT bundle for trust domain %q", trustDomain)
	}

	return b, nil
}
