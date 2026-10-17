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
	"testing"

	"k8s.io/component-helpers/auth/admission/internal/oidc"
)

// TestLocalKeySetVerifier_RequiresTrustedAudience ensures discovery never
// creates an accept-any-audience verifier and the chosen audience cannot be
// changed after construction.
func TestLocalKeySetVerifier_RequiresTrustedAudience(t *testing.T) {
	s := newLocalAPIServer(t)
	if _, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, "", s.Client()); err == nil {
		t.Fatal("expected construction to reject a missing webhook audience")
	}

	v, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, testAudience, s.Client())
	if err != nil {
		t.Fatalf("NewLocalKeySetVerifier: %v", err)
	}
	if err := v.HealthCheck(); err != nil {
		t.Fatalf("expected ready after construction with a trusted audience: %v", err)
	}
	if err := v.Verify(context.Background(), signWith(t, s.signer, localClaims(s.URL)), testAPIGroup); err != nil {
		t.Fatalf("verify with configured audience: %v", err)
	}
	if err := v.BindAudience(testAudience); err != nil {
		t.Fatalf("rebinding the same audience should be a no-op: %v", err)
	}
	if err := v.BindAudience("https://other.webhook.svc/validate"); err == nil {
		t.Fatal("expected an error when rebinding to a different audience")
	}
}
