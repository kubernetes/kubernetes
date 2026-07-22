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

	"k8s.io/webhookauth/internal/oidc"
)

// TestLocalKeySetVerifier_BindsAudienceAndReportsHealth exercises the in-cluster
// deferred verifier lifecycle: until an audience is bound it denies every token
// and reports unhealthy; once bound it verifies real tokens and reports ready.
// Rebinding the same audience is a no-op; a different audience is rejected so the
// frozen audience cannot be repointed.
//
// The happy path and deferred-until-bind cases are also in
// TestNewLocalKeySetVerifier_* (local_test.go); this test additionally pins
// rebind idempotency and conflicting-rebind rejection.
func TestLocalKeySetVerifier_BindsAudienceAndReportsHealth(t *testing.T) {
	s := newLocalAPIServer(t)

	v, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, oidc.WithHTTPClient(s.Client()))
	if err != nil {
		t.Fatalf("NewLocalKeySetVerifier: %v", err)
	}

	// Before binding: not ready, and every token is denied.
	if err := v.HealthCheck(); err == nil {
		t.Fatal("expected HealthCheck to report not-ready before an audience is bound")
	}
	if err := v.Verify(context.Background(), signWith(t, s.signer, localClaims(s.URL)), testAPIGroup); err == nil {
		t.Fatal("expected verification to be denied before an audience is bound")
	}

	// Bind the audience the tokens carry.
	if err := v.BindAudience(testAudience); err != nil {
		t.Fatalf("BindAudience: %v", err)
	}
	if err := v.HealthCheck(); err != nil {
		t.Fatalf("expected ready after binding, got %v", err)
	}
	if err := v.Verify(context.Background(), signWith(t, s.signer, localClaims(s.URL)), testAPIGroup); err != nil {
		t.Fatalf("expected verification to succeed after binding, got %v", err)
	}

	// Idempotent rebind; conflicting rebind rejected.
	if err := v.BindAudience(testAudience); err != nil {
		t.Fatalf("rebinding the same audience should be a no-op, got %v", err)
	}
	if err := v.BindAudience("https://other.webhook.svc/validate"); err == nil {
		t.Fatal("expected an error when rebinding to a different audience")
	}
}
