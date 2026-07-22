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

package oidc_test

import (
	"context"
	"errors"
	"testing"

	"k8s.io/webhookauth/verify"
)

// TestRemoteVerifier_MalformedKubernetesClaims covers the private-claims decode
// path: a token whose signature and standard claims (iss/aud/exp) are valid — so
// go-oidc accepts it — but whose "kubernetes.io" claim is not an object cannot
// decode into the webhook private-claims struct. That failure must collapse into
// the generic ErrVerificationFailed, never surfacing decode detail to the caller.
func TestRemoteVerifier_MalformedKubernetesClaims(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.newVerifier(t)

	claims := ts.baseClaims()
	// Replace the kubernetes.io object with a scalar: go-oidc's Verify still
	// passes (it only inspects iss/aud/exp/sub), but decoding into the private
	// claims struct fails.
	claims["kubernetes.io"] = "not-an-object"

	err := v.Verify(context.Background(), ts.sign(t, claims), testAPIGroup)
	if err == nil {
		t.Fatal("expected verification to fail on an undecodable kubernetes.io claim")
	}
	if !errors.Is(err, verify.ErrVerificationFailed) {
		t.Fatalf("expected generic ErrVerificationFailed, got %v", err)
	}
	// Anti-enumeration: the caller-facing message must not leak decode detail.
	if msg := err.Error(); msg != verify.ErrVerificationFailed.Error() {
		t.Errorf("caller-visible error = %q, want the generic message", msg)
	}
}
