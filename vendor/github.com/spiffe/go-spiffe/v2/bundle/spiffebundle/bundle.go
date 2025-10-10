package spiffebundle

import (
	"crypto"
	"crypto/x509"
	"encoding/json"
	"errors"
	"io"
	"os"
	"sync"
	"time"

	"github.com/go-jose/go-jose/v4"
	"github.com/spiffe/go-spiffe/v2/bundle/jwtbundle"
	"github.com/spiffe/go-spiffe/v2/bundle/x509bundle"
	"github.com/spiffe/go-spiffe/v2/internal/jwtutil"
	"github.com/spiffe/go-spiffe/v2/internal/x509util"
	"github.com/spiffe/go-spiffe/v2/spiffeid"
	"github.com/zeebo/errs"
)

const (
	x509SVIDUse = "x509-svid"
	jwtSVIDUse  = "jwt-svid"
)

var spiffebundleErr = errs.Class("spiffebundle")

type bundleDoc struct {
	jose.JSONWebKeySet
	SequenceNumber *uint64 `json:"spiffe_sequence,omitempty"`
	RefreshHint    *int64  `json:"spiffe_refresh_hint,omitempty"`
}

// Bundle is a collection of trusted public key material for a trust domain,
// conforming to the SPIFFE Bundle Format as part of the SPIFFE Trust Domain
// and Bundle specification:
// https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE_Trust_Domain_and_Bundle.md
type Bundle struct {
	trustDomain spiffeid.TrustDomain

	mtx             sync.RWMutex
	refreshHint     *time.Duration
	sequenceNumber  *uint64
	jwtAuthorities  map[string]crypto.PublicKey
	x509Authorities []*x509.Certificate
}

// New creates a new bundle.
func New(trustDomain spiffeid.TrustDomain) *Bundle {
	return &Bundle{
		trustDomain:    trustDomain,
		jwtAuthorities: make(map[string]crypto.PublicKey),
	}
}

// Load loads a bundle from a file on disk. The file must contain a JWKS
// document following the SPIFFE Trust Domain and Bundle specification.
func Load(trustDomain spiffeid.TrustDomain, path string) (*Bundle, error) {
	bundleBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, spiffebundleErr.New("unable to read SPIFFE bundle: %w", err)
	}

	return Parse(trustDomain, bundleBytes)
}

// Read decodes a bundle from a reader. The contents must contain a JWKS
// document following the SPIFFE Trust Domain and Bundle specification.
func Read(trustDomain spiffeid.TrustDomain, r io.Reader) (*Bundle, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, spiffebundleErr.New("unable to read: %v", err)
	}

	return Parse(trustDomain, b)
}

// Parse parses a bundle from bytes. The data must be a JWKS document following
// the SPIFFE Trust Domain and Bundle specification.
func Parse(trustDomain spiffeid.TrustDomain, bundleBytes []byte) (*Bundle, error) {
	jwks := &bundleDoc{}
	if err := json.Unmarshal(bundleBytes, jwks); err != nil {
		return nil, spiffebundleErr.New("unable to parse JWKS: %v", err)
	}

	bundle := New(trustDomain)
	if jwks.RefreshHint != nil {
		bundle.SetRefreshHint(time.Second * time.Duration(*jwks.RefreshHint))
	}
	if jwks.SequenceNumber != nil {
		bundle.SetSequenceNumber(*jwks.SequenceNumber)
	}

	if jwks.Keys == nil {
		// The parameter keys MUST be present.
		// https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE_Trust_Domain_and_Bundle.md#413-keys
		return nil, spiffebundleErr.New("no authorities found")
	}
	for i, key := range jwks.Keys {
		switch key.Use {
		// Two SVID types are supported: x509-svid and jwt-svid.
		case x509SVIDUse:
			if len(key.Certificates) != 1 {
				return nil, spiffebundleErr.New("expected a single certificate in %s entry %d; got %d", x509SVIDUse, i, len(key.Certificates))
			}
			bundle.AddX509Authority(key.Certificates[0])
		case jwtSVIDUse:
			if err := bundle.AddJWTAuthority(key.KeyID, key.Key); err != nil {
				return nil, spiffebundleErr.New("error adding authority %d of JWKS: %v", i, errors.Unwrap(err))
			}
		}
	}

	return bundle, nil
}

// FromX509Bundle creates a bundle from an X.509 bundle.
// The function panics in case of a nil X.509 bundle.
func FromX509Bundle(x509Bundle *x509bundle.Bundle) *Bundle {
	bundle := New(x509Bundle.TrustDomain())
	bundle.x509Authorities = x509Bundle.X509Authorities()
	return bundle
}

// FromJWTBundle creates a bundle from a JWT bundle.
// The function panics in case of a nil JWT bundle.
func FromJWTBundle(jwtBundle *jwtbundle.Bundle) *Bundle {
	bundle := New(jwtBundle.TrustDomain())
	bundle.jwtAuthorities = jwtBundle.JWTAuthorities()
	return bundle
}

// FromX509Authorities creates a bundle from X.509 certificates.
func FromX509Authorities(trustDomain spiffeid.TrustDomain, x509Authorities []*x509.Certificate) *Bundle {
	bundle := New(trustDomain)
	bundle.x509Authorities = x509util.CopyX509Authorities(x509Authorities)
	return bundle
}

// FromJWTAuthorities creates a new bundle from JWT authorities.
func FromJWTAuthorities(trustDomain spiffeid.TrustDomain, jwtAuthorities map[string]crypto.PublicKey) *Bundle {
	bundle := New(trustDomain)
	bundle.jwtAuthorities = jwtutil.CopyJWTAuthorities(jwtAuthorities)
	return bundle
}

// TrustDomain returns the trust domain that the bundle belongs to.
func (b *Bundle) TrustDomain() spiffeid.TrustDomain {
	return b.trustDomain
}

// X509Authorities returns the X.509 authorities in the bundle.
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
func (b *Bundle) SetX509Authorities(authorities []*x509.Certificate) {
	b.mtx.Lock()
	defer b.mtx.Unlock()

	b.x509Authorities = x509util.CopyX509Authorities(authorities)
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

	jwtAuthority, ok := b.jwtAuthorities[keyID]
	return jwtAuthority, ok
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
		return spiffebundleErr.New("keyID cannot be empty")
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

// Empty returns true if the bundle has no X.509 and JWT authorities.
func (b *Bundle) Empty() bool {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	return len(b.x509Authorities) == 0 && len(b.jwtAuthorities) == 0
}

// RefreshHint returns the refresh hint. If the refresh hint is set in
// the bundle, it is returned and the boolean is true. Otherwise, the returned
// value is zero and the boolean is false.
func (b *Bundle) RefreshHint() (refreshHint time.Duration, ok bool) {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	if b.refreshHint != nil {
		return *b.refreshHint, true
	}
	return 0, false
}

// SetRefreshHint sets the refresh hint. The refresh hint value will be
// truncated to time.Second.
func (b *Bundle) SetRefreshHint(refreshHint time.Duration) {
	b.mtx.Lock()
	defer b.mtx.Unlock()

	b.refreshHint = &refreshHint
}

// ClearRefreshHint clears the refresh hint.
func (b *Bundle) ClearRefreshHint() {
	b.mtx.Lock()
	defer b.mtx.Unlock()

	b.refreshHint = nil
}

// SequenceNumber returns the sequence number. If the sequence number is set in
// the bundle, it is returned and the boolean is true. Otherwise, the returned
// value is zero and the boolean is false.
func (b *Bundle) SequenceNumber() (uint64, bool) {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	if b.sequenceNumber != nil {
		return *b.sequenceNumber, true
	}
	return 0, false
}

// SetSequenceNumber sets the sequence number.
func (b *Bundle) SetSequenceNumber(sequenceNumber uint64) {
	b.mtx.Lock()
	defer b.mtx.Unlock()

	b.sequenceNumber = &sequenceNumber
}

// ClearSequenceNumber clears the sequence number.
func (b *Bundle) ClearSequenceNumber() {
	b.mtx.Lock()
	defer b.mtx.Unlock()

	b.sequenceNumber = nil
}

// Marshal marshals the bundle according to the SPIFFE Trust Domain and Bundle
// specification. The trust domain is not marshaled as part of the bundle and
// must be conveyed separately. See the specification for details.
func (b *Bundle) Marshal() ([]byte, error) {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	jwks := bundleDoc{}
	if b.refreshHint != nil {
		tr := int64((*b.refreshHint + (time.Second - 1)) / time.Second)
		jwks.RefreshHint = &tr
	}
	jwks.SequenceNumber = b.sequenceNumber
	for _, x509Authority := range b.x509Authorities {
		jwks.Keys = append(jwks.Keys, jose.JSONWebKey{
			Key:          x509Authority.PublicKey,
			Certificates: []*x509.Certificate{x509Authority},
			Use:          x509SVIDUse,
		})
	}

	for keyID, jwtAuthority := range b.jwtAuthorities {
		jwks.Keys = append(jwks.Keys, jose.JSONWebKey{
			Key:   jwtAuthority,
			KeyID: keyID,
			Use:   jwtSVIDUse,
		})
	}

	return json.Marshal(jwks)
}

// Clone clones the bundle.
func (b *Bundle) Clone() *Bundle {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	return &Bundle{
		trustDomain:     b.trustDomain,
		refreshHint:     copyRefreshHint(b.refreshHint),
		sequenceNumber:  copySequenceNumber(b.sequenceNumber),
		x509Authorities: x509util.CopyX509Authorities(b.x509Authorities),
		jwtAuthorities:  jwtutil.CopyJWTAuthorities(b.jwtAuthorities),
	}
}

// X509Bundle returns an X.509 bundle containing the X.509 authorities in the SPIFFE
// bundle.
func (b *Bundle) X509Bundle() *x509bundle.Bundle {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	// FromX509Authorities makes a copy, so we can pass our internal slice directly.
	return x509bundle.FromX509Authorities(b.trustDomain, b.x509Authorities)
}

// JWTBundle returns a JWT bundle containing the JWT authorities in the SPIFFE bundle.
func (b *Bundle) JWTBundle() *jwtbundle.Bundle {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	// FromJWTBundle makes a copy, so we can pass our internal slice directly.
	return jwtbundle.FromJWTAuthorities(b.trustDomain, b.jwtAuthorities)
}

// GetBundleForTrustDomain returns the SPIFFE bundle for the given trust
// domain. It implements the Source interface. An error will be returned if the
// trust domain does not match that of the bundle.
func (b *Bundle) GetBundleForTrustDomain(trustDomain spiffeid.TrustDomain) (*Bundle, error) {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	if b.trustDomain != trustDomain {
		return nil, spiffebundleErr.New("no SPIFFE bundle for trust domain %q", trustDomain)
	}

	return b, nil
}

// GetX509BundleForTrustDomain returns the X.509 bundle for the given trust
// domain. It implements the x509bundle.Source interface. An error will be
// returned if the trust domain does not match that of the bundle.
func (b *Bundle) GetX509BundleForTrustDomain(trustDomain spiffeid.TrustDomain) (*x509bundle.Bundle, error) {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	if b.trustDomain != trustDomain {
		return nil, spiffebundleErr.New("no X.509 bundle for trust domain %q", trustDomain)
	}

	return b.X509Bundle(), nil
}

// GetJWTBundleForTrustDomain returns the JWT bundle of the given trust domain.
// It implements the jwtbundle.Source interface. An error will be returned if
// the trust domain does not match that of the bundle.
func (b *Bundle) GetJWTBundleForTrustDomain(trustDomain spiffeid.TrustDomain) (*jwtbundle.Bundle, error) {
	b.mtx.RLock()
	defer b.mtx.RUnlock()

	if b.trustDomain != trustDomain {
		return nil, spiffebundleErr.New("no JWT bundle for trust domain %q", trustDomain)
	}

	return b.JWTBundle(), nil
}

// Equal compares the bundle for equality against the given bundle.
func (b *Bundle) Equal(other *Bundle) bool {
	if b == nil || other == nil {
		return b == other
	}

	return b.trustDomain == other.trustDomain &&
		refreshHintEqual(b.refreshHint, other.refreshHint) &&
		sequenceNumberEqual(b.sequenceNumber, other.sequenceNumber) &&
		jwtutil.JWTAuthoritiesEqual(b.jwtAuthorities, other.jwtAuthorities) &&
		x509util.CertsEqual(b.x509Authorities, other.x509Authorities)
}

func refreshHintEqual(a, b *time.Duration) bool {
	if a == nil || b == nil {
		return a == b
	}

	return *a == *b
}

func sequenceNumberEqual(a, b *uint64) bool {
	if a == nil || b == nil {
		return a == b
	}

	return *a == *b
}

func copyRefreshHint(refreshHint *time.Duration) *time.Duration {
	if refreshHint == nil {
		return nil
	}
	copied := *refreshHint
	return &copied
}

func copySequenceNumber(sequenceNumber *uint64) *uint64 {
	if sequenceNumber == nil {
		return nil
	}
	copied := *sequenceNumber
	return &copied
}
