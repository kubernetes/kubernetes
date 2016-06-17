package oidc

import (
	"encoding/json"
	"errors"
	"net/http"
	"time"

	phttp "github.com/coreos/go-oidc/http"
	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/key"
)

// DefaultPublicKeySetTTL is the default TTL set on the PublicKeySet if no
// Cache-Control header is provided by the JWK Set document endpoint.
const DefaultPublicKeySetTTL = 24 * time.Hour

// NewRemotePublicKeyRepo is responsible for fetching the JWK Set document.
func NewRemotePublicKeyRepo(hc phttp.Client, ep string) *remotePublicKeyRepo {
	return &remotePublicKeyRepo{hc: hc, ep: ep}
}

type remotePublicKeyRepo struct {
	hc phttp.Client
	ep string
}

// Get returns a PublicKeySet fetched from the JWK Set document endpoint. A TTL
// is set on the Key Set to avoid it having to be re-retrieved for every
// encryption event. This TTL is typically controlled by the endpoint returning
// a Cache-Control header, but defaults to 24 hours if no Cache-Control header
// is found.
func (r *remotePublicKeyRepo) Get() (key.KeySet, error) {
	req, err := http.NewRequest("GET", r.ep, nil)
	if err != nil {
		return nil, err
	}

	resp, err := r.hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var d struct {
		Keys []jose.JWK `json:"keys"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&d); err != nil {
		return nil, err
	}

	if len(d.Keys) == 0 {
		return nil, errors.New("zero keys in response")
	}

	ttl, ok, err := phttp.Cacheable(resp.Header)
	if err != nil {
		return nil, err
	}
	if !ok {
		ttl = DefaultPublicKeySetTTL
	}

	exp := time.Now().UTC().Add(ttl)
	ks := key.NewPublicKeySet(d.Keys, exp)
	return ks, nil
}
