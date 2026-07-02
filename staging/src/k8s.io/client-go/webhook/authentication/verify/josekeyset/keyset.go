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

// StaticKeySet verifies JWS signatures against a fixed set of JSON Web Keys.
// It performs no network I/O, which makes it suitable for tests and for callers
// that provision issuer keys out of band. An OIDC-discovery / JWKS-fetching
// implementation of verify.KeySet is a planned follow-up.
type StaticKeySet struct {
	keys jose.JSONWebKeySet
}

// NewStaticKeySet returns a StaticKeySet backed by the provided keys.
func NewStaticKeySet(keys jose.JSONWebKeySet) *StaticKeySet {
	return &StaticKeySet{keys: keys}
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
	for i := range s.keys.Keys {
		if payload, err := jws.Verify(&s.keys.Keys[i]); err == nil {
			return payload, nil
		}
	}
	return nil, errors.New("josekeyset: no key verified the token signature")
}
