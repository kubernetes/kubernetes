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

// Package josekeyset provides a go-jose-backed [verify.KeySet] implementation.
//
// It is deliberately a separate package from the core verifier so that all
// JOSE/JWT dependencies are confined here: the core
// k8s.io/client-go/webhook/authentication/verify package builds and tests with
// only the standard library, and this subpackage is the sole place that pulls in
// gopkg.in/go-jose/go-jose.v2. Relocating the verifier (e.g. to a dedicated
// staging module) only requires moving this file's dependency.
//
// NOTE: gopkg.in/go-jose/go-jose.v2 is not currently a dependency of the
// k8s.io/client-go module. Compiling this subpackage requires adding it via
// hack/pin-dependency.sh + hack/update-vendor.sh and sig-auth sign-off.
package josekeyset // import "k8s.io/client-go/webhook/authentication/verify/josekeyset"

import (
	"context"
	"errors"

	jose "gopkg.in/go-jose/go-jose.v2"
)

// allowedAlgorithms is the explicit allowlist of JWS "alg" values this package
// will verify. It is restricted to asymmetric signature algorithms: symmetric
// MACs (HS*) and "none" are rejected outright. Enforcing the allowlist before
// verification is defense in depth on top of go-jose's typed-key binding, and
// closes the classic key-confusion class where an RSA public key is replayed as
// an HMAC secret.
var allowedAlgorithms = map[string]bool{
	string(jose.RS256): true,
	string(jose.RS384): true,
	string(jose.RS512): true,
	string(jose.ES256): true,
	string(jose.ES384): true,
	string(jose.ES512): true,
	string(jose.PS256): true,
	string(jose.PS384): true,
	string(jose.PS512): true,
}

// errDisallowedAlgorithm reports that a JWS header advertised an algorithm
// outside allowedAlgorithms. It is intentionally coarse; the core verifier
// collapses it into a generic verification failure.
var errDisallowedAlgorithm = errors.New("josekeyset: token algorithm is not allowed")

// checkAlgorithms rejects a JWS whose any signature header advertises an
// algorithm outside the allowlist.
func checkAlgorithms(jws *jose.JSONWebSignature) error {
	if len(jws.Signatures) == 0 {
		return errDisallowedAlgorithm
	}
	for _, sig := range jws.Signatures {
		if !allowedAlgorithms[sig.Header.Algorithm] {
			return errDisallowedAlgorithm
		}
	}
	return nil
}

// filterSignatureKeys drops keys that must not be trusted for JWS verification:
// symmetric (oct) keys, whose material would enable HMAC key-confusion, and keys
// explicitly published for encryption ("use":"enc"). Keys with no "use" are
// kept, matching the JWK default that an unspecified use permits signing.
func filterSignatureKeys(keys []jose.JSONWebKey) []jose.JSONWebKey {
	out := make([]jose.JSONWebKey, 0, len(keys))
	for _, k := range keys {
		if k.Use != "" && k.Use != "sig" {
			continue
		}
		// Symmetric keys decode to a []byte key; reject them so a shared secret
		// cannot be used to verify an asymmetric token.
		if _, symmetric := k.Key.([]byte); symmetric {
			continue
		}
		out = append(out, k)
	}
	return out
}

// StaticKeySet verifies JWS signatures against a fixed set of JSON Web Keys.
// It performs no network I/O, which makes it suitable for tests and for callers
// that provision issuer keys out of band. An OIDC-discovery / JWKS-fetching
// implementation of verify.KeySet is a planned follow-up.
type StaticKeySet struct {
	keys jose.JSONWebKeySet
}

// NewStaticKeySet returns a StaticKeySet backed by the provided keys. Keys that
// are not usable for signature verification (symmetric keys, encryption-only
// keys) are filtered out so they can never verify a token.
func NewStaticKeySet(keys jose.JSONWebKeySet) *StaticKeySet {
	return &StaticKeySet{keys: jose.JSONWebKeySet{Keys: filterSignatureKeys(keys.Keys)}}
}

// VerifySignature implements verify.KeySet. It parses rawToken as a compact JWS
// and returns the payload of the first key that verifies it. The returned error
// is intentionally coarse; the core verifier collapses it into a generic
// verification failure so callers cannot distinguish failure modes.
func (s *StaticKeySet) VerifySignature(_ context.Context, rawToken string) ([]byte, error) {
	jws, err := jose.ParseSigned(rawToken)
	if err != nil {
		return nil, err
	}
	if err := checkAlgorithms(jws); err != nil {
		return nil, err
	}
	for i := range s.keys.Keys {
		if payload, err := jws.Verify(&s.keys.Keys[i]); err == nil {
			return payload, nil
		}
	}
	return nil, errors.New("josekeyset: no key verified the token signature")
}
