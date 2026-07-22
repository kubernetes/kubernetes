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

package verify

import (
	"errors"
	"testing"
)

// fakeBinder is a TokenAuthenticator that records BindAudience calls and reports
// a configurable health, so the Verifier's direct delegation of BindAudience and
// HealthCheck can be tested without crypto. It overrides the always-ready
// implementations promoted from the embedded fakeAuthenticator.
type fakeBinder struct {
	fakeAuthenticator
	bound     string
	bindErr   error
	healthErr error
}

func (f *fakeBinder) BindAudience(audience string) error {
	if f.bindErr != nil {
		return f.bindErr
	}
	f.bound = audience
	return nil
}

func (f *fakeBinder) HealthCheck() error { return f.healthErr }

// TestVerifier_BindAudience_NoopForFixedAudienceAuthenticator confirms that
// BindAudience and HealthCheck on an authenticator whose audience is fixed at
// construction (the out-of-cluster case, modelled by the plain fakeAuthenticator)
// are no-ops that report ready, so out-of-cluster callers are unaffected. The
// delegation is now direct (BindAudience/HealthCheck are part of
// TokenAuthenticator), not an optional-interface type assertion.
func TestVerifier_BindAudience_NoopForFixedAudienceAuthenticator(t *testing.T) {
	v := mustVerifier(t, fakeAuthenticator{groups: []string{testGroup}})
	if err := v.BindAudience("https://webhook.example/validate"); err != nil {
		t.Fatalf("BindAudience on a non-binder should be a no-op, got %v", err)
	}
	if err := v.HealthCheck(); err != nil {
		t.Fatalf("HealthCheck on a non-checker should be nil, got %v", err)
	}
}

// TestVerifier_BindAudience_Delegates confirms BindAudience and HealthCheck are
// forwarded to the backing authenticator's implementations.
func TestVerifier_BindAudience_Delegates(t *testing.T) {
	binder := &fakeBinder{}
	v := mustVerifier(t, binder)

	if err := v.HealthCheck(); err != nil {
		t.Fatalf("unexpected health error: %v", err)
	}
	if err := v.BindAudience("aud-1"); err != nil {
		t.Fatalf("BindAudience: %v", err)
	}
	if binder.bound != "aud-1" {
		t.Fatalf("audience not forwarded, got %q", binder.bound)
	}

	wantBind := errors.New("bind boom")
	binder.bindErr = wantBind
	if err := v.BindAudience("aud-2"); !errors.Is(err, wantBind) {
		t.Fatalf("BindAudience error = %v, want %v", err, wantBind)
	}

	wantHealth := errors.New("not ready")
	binder.healthErr = wantHealth
	if err := v.HealthCheck(); !errors.Is(err, wantHealth) {
		t.Fatalf("HealthCheck error = %v, want %v", err, wantHealth)
	}
}
