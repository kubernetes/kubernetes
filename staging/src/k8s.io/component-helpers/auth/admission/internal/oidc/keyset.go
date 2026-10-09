/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package oidc

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/rsa"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	coreosoidc "github.com/coreos/go-oidc/v3/oidc"
	jose "github.com/go-jose/go-jose/v4"
)

const (
	maxJWKSBytes             = 1 << 20
	maxJWKSKeys              = 128
	minRSAModulusBits        = 2048
	signingKeyRefreshTimeout = 10 * time.Second
)

// The local apiserver accepts precisely these ServiceAccount signing algorithms.
var serviceAccountSigningAlgs = []string{"ES256", "ES384", "ES512", "RS256"}

// A remote OIDC issuer may advertise additional asymmetric signing algorithms.
var remoteSigningAlgs = []string{"ES256", "ES384", "ES512", "RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "EdDSA"}

func boundedHTTPClient(client *http.Client) *http.Client {
	if client == nil {
		client = http.DefaultClient
	}
	bounded := *client
	if bounded.Timeout <= 0 || bounded.Timeout > signingKeyRefreshTimeout {
		bounded.Timeout = signingKeyRefreshTimeout
	}
	previousRedirectPolicy := bounded.CheckRedirect
	bounded.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if req.URL.Scheme != "https" {
			return errors.New("oidc: discovery and signing-key redirects must use HTTPS")
		}
		if previousRedirectPolicy != nil {
			return previousRedirectPolicy(req, via)
		}
		if len(via) >= 10 {
			return errors.New("oidc: stopped after 10 redirects")
		}
		return nil
	}
	return &bounded
}

type keySetReadiness struct {
	uri        string
	client     *http.Client
	algorithms []string
	allowed    map[string]bool
}

func newKeySetReadiness(uri string, client *http.Client, signingAlgs []string) (*keySetReadiness, error) {
	u, err := url.Parse(uri)
	if err != nil || u.Scheme != "https" || u.Host == "" || u.User != nil || u.Fragment != "" {
		return nil, errors.New("oidc: JWKS URI must be an HTTPS URL without userinfo or fragment")
	}
	if client == nil || len(signingAlgs) == 0 {
		return nil, errors.New("oidc: signing-key loading requires an HTTP client and asymmetric algorithms")
	}
	allowed := make(map[string]bool, len(signingAlgs))
	for _, alg := range signingAlgs {
		allowed[alg] = true
	}
	return &keySetReadiness{uri: uri, client: client, algorithms: append([]string(nil), signingAlgs...), allowed: allowed}, nil
}

// loadKeys returns exactly the validated keys used by the verifier. Probing a
// different key set would allow readiness to succeed while verification still
// trusts a removed key.
func (r *keySetReadiness) loadKeys(ctx context.Context) (*jwksKeySet, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.uri, nil)
	if err != nil {
		return nil, fmt.Errorf("oidc: create signing-key request: %w", err)
	}
	req.Header.Set("Cache-Control", "no-cache")
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("oidc: fetch signing keys: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("oidc: signing-key endpoint returned %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxJWKSBytes+1))
	if err != nil {
		return nil, fmt.Errorf("oidc: read signing keys: %w", err)
	}
	if len(body) > maxJWKSBytes {
		return nil, errors.New("oidc: signing-key response exceeds the size limit")
	}
	// Match go-oidc's per-key parser: unknown algorithms and key types do not
	// invalidate an otherwise usable JWKS, but malformed supported keys do.
	var raw struct {
		Keys []json.RawMessage `json:"keys"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("oidc: parse signing keys: %w", err)
	}
	if len(raw.Keys) > maxJWKSKeys {
		return nil, fmt.Errorf("oidc: signing-key response exceeds %d keys", maxJWKSKeys)
	}
	keys := make([]jose.JSONWebKey, 0, len(raw.Keys))
	for _, rawKey := range raw.Keys {
		var metadata struct {
			Alg string `json:"alg"`
		}
		if err := json.Unmarshal(rawKey, &metadata); err != nil {
			return nil, fmt.Errorf("oidc: parse signing-key metadata: %w", err)
		}
		if metadata.Alg != "" && !r.allowed[metadata.Alg] {
			continue
		}
		var key jose.JSONWebKey
		if err := json.Unmarshal(rawKey, &key); err != nil {
			if errors.Is(err, jose.ErrUnsupportedKeyType) {
				continue
			}
			return nil, fmt.Errorf("oidc: parse signing key: %w", err)
		}
		if !key.Valid() || !key.IsPublic() || key.Use != "" && key.Use != "sig" {
			continue
		}
		for _, alg := range r.algorithms {
			if keySupportsAlgorithm(key, alg) {
				keys = append(keys, key)
				break
			}
		}
	}
	if len(keys) == 0 {
		return nil, errors.New("oidc: signing-key endpoint has no usable public signing keys")
	}
	algorithms := make([]jose.SignatureAlgorithm, 0, len(r.algorithms))
	for _, alg := range r.algorithms {
		algorithms = append(algorithms, jose.SignatureAlgorithm(alg))
	}
	return &jwksKeySet{keys: keys, algorithms: algorithms}, nil
}

func keySupportsAlgorithm(key jose.JSONWebKey, algorithm string) bool {
	if key.Algorithm != "" && key.Algorithm != algorithm {
		return false
	}
	switch pub := key.Key.(type) {
	case *rsa.PublicKey:
		// RFC 7518 §§3.3 and 3.5 require RSA keys of at least 2048 bits for
		// RS*/PS* signatures; smaller keys are not valid JOSE signing keys.
		if pub.N == nil || pub.N.Sign() <= 0 || pub.N.BitLen() < minRSAModulusBits || pub.N.Bit(0) == 0 || pub.E < 3 || pub.E%2 == 0 {
			return false
		}
		return algorithm == "RS256" || algorithm == "RS384" || algorithm == "RS512" ||
			algorithm == "PS256" || algorithm == "PS384" || algorithm == "PS512"
	case *ecdsa.PublicKey:
		switch pub.Curve.Params().Name {
		case "P-256":
			return algorithm == "ES256"
		case "P-384":
			return algorithm == "ES384"
		case "P-521":
			return algorithm == "ES512"
		}
	case ed25519.PublicKey:
		return algorithm == "EdDSA"
	}
	return false
}

// jwksKeySet retains the JWK metadata that go-oidc's StaticKeySet drops. The
// StaticKeySet still performs the signature verification after kid/alg filtering.
type jwksKeySet struct {
	keys       []jose.JSONWebKey
	algorithms []jose.SignatureAlgorithm
}

func (s *jwksKeySet) VerifySignature(ctx context.Context, jwt string) ([]byte, error) {
	jws, err := jose.ParseSigned(jwt, s.algorithms)
	if err != nil {
		return nil, fmt.Errorf("oidc: parse signed token: %w", err)
	}
	if len(jws.Signatures) != 1 {
		return nil, errors.New("oidc: token must have exactly one signature")
	}
	header := jws.Signatures[0].Header
	publicKeys := make([]crypto.PublicKey, 0, len(s.keys))
	for _, key := range s.keys {
		if (header.KeyID == "" || key.KeyID == header.KeyID) && keySupportsAlgorithm(key, header.Algorithm) {
			publicKeys = append(publicKeys, key.Key)
		}
	}
	if len(publicKeys) == 0 {
		return nil, errors.New("oidc: token signing key is not in the current JWKS")
	}
	return (&coreosoidc.StaticKeySet{PublicKeys: publicKeys}).VerifySignature(ctx, jwt)
}
