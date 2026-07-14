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

	"k8s.io/webhookauth/verify/oidc"
)

// TestDeferredVerifier_BindsAudienceAndReportsHealth exercises the in-cluster
// verifier shape: the key set is fetched at construction, but until an audience
// is bound the verifier denies every token and reports unhealthy. Once bound, it
// verifies real tokens and reports ready. Rebinding the same audience is a no-op;
// rebinding a different audience is rejected so the frozen audience cannot be
// repointed.
func TestDeferredVerifier_BindsAudienceAndReportsHealth(t *testing.T) {
	ts := newOIDCTestServer(t)

	v, err := oidc.NewDeferredVerifierForTest(context.Background(), ts.issuer, oidc.WithHTTPClient(ts.client()))
	if err != nil {
		t.Fatalf("NewDeferredVerifierForTest: %v", err)
	}

	// Before binding: not ready, and every token is denied.
	if err := v.HealthCheck(); err == nil {
		t.Fatal("expected HealthCheck to report not-ready before an audience is bound")
	}
	if err := v.Verify(context.Background(), ts.sign(t, ts.baseClaims()), testAPIGroup); err == nil {
		t.Fatal("expected verification to be denied before an audience is bound")
	}

	// Bind the audience the tokens carry.
	if err := v.BindAudience(testAudience); err != nil {
		t.Fatalf("BindAudience: %v", err)
	}
	if err := v.HealthCheck(); err != nil {
		t.Fatalf("expected ready after binding, got %v", err)
	}
	if err := v.Verify(context.Background(), ts.sign(t, ts.baseClaims()), testAPIGroup); err != nil {
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
