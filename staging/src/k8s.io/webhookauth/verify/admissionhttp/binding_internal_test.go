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

package admissionhttp

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/webhookauth/verify"
)

// bindSpy is a TokenAuthenticator that records BindAudience calls and reports a
// configurable health, so the handler's first-request binding can be tested
// without crypto.
type bindSpy struct {
	binds     []string
	bindErr   error
	healthErr error
}

func (b *bindSpy) AuthenticateToken(context.Context, string) ([]string, error) { return nil, nil }

func (b *bindSpy) BindAudience(audience string) error {
	if b.bindErr != nil {
		return b.bindErr
	}
	b.binds = append(b.binds, audience)
	return nil
}

func (b *bindSpy) HealthCheck() error { return b.healthErr }

func newSpyHandler(t *testing.T, spy *bindSpy, resolve AudienceResolver) *Handler {
	t.Helper()
	v, err := verify.NewVerifier(spy)
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}
	return &Handler{verifier: v, resolve: resolve}
}

// TestEnsureAudience_BindsOnce confirms the audience is derived and bound exactly
// once, from the first request, and not rebound on later requests.
func TestEnsureAudience_BindsOnce(t *testing.T) {
	spy := &bindSpy{}
	h := newSpyHandler(t, spy, func(*http.Request) (string, error) { return "aud-1", nil })

	r := httptest.NewRequest("POST", "https://my-webhook.ns.svc:443/validate", nil)
	for i := 0; i < 3; i++ {
		if err := h.ensureAudience(r); err != nil {
			t.Fatalf("ensureAudience call %d: %v", i, err)
		}
	}
	if len(spy.binds) != 1 || spy.binds[0] != "aud-1" {
		t.Fatalf("expected a single bind of %q, got %v", "aud-1", spy.binds)
	}
}

// TestEnsureAudience_ResolverErrorIsFailClosed confirms a resolver error is
// returned (request denied) and nothing is bound. The failure is NOT cached:
// resolver inputs are per-request (the Host is attacker-influenceable), so a
// failed attempt is retried rather than permanently wedging the handler deny-all;
// a permanent misconfiguration keeps failing and surfaces via HealthCheck.
func TestEnsureAudience_ResolverErrorIsFailClosed(t *testing.T) {
	spy := &bindSpy{}
	calls := 0
	h := newSpyHandler(t, spy, func(*http.Request) (string, error) {
		calls++
		return "", errors.New("no service env var")
	})

	r := httptest.NewRequest("POST", "https://my-webhook.ns.svc:443/validate", nil)
	if err := h.ensureAudience(r); err == nil {
		t.Fatal("expected ensureAudience to return the resolver error")
	}
	// A second call retries the resolver (the failure is not cached) and still
	// fails closed.
	if err := h.ensureAudience(r); err == nil {
		t.Fatal("expected the resolver error again on the second call")
	}
	if calls != 2 {
		t.Fatalf("resolver should be retried on each failure (no caching), got %d calls", calls)
	}
	if len(spy.binds) != 0 {
		t.Fatalf("nothing should be bound on resolver failure, got %v", spy.binds)
	}
}

// TestEnsureAudience_RetriesAfterFailureThenBinds confirms a first request whose
// resolve fails does NOT permanently wedge the handler: a later request with a
// resolvable audience binds and is served. The DoS fix — an unverified caller
// with a bogus Host can no longer poison the bind for the pod's lifetime (Edie
// review-3 blocker #1).
func TestEnsureAudience_RetriesAfterFailureThenBinds(t *testing.T) {
	spy := &bindSpy{}
	fail := true
	h := newSpyHandler(t, spy, func(*http.Request) (string, error) {
		if fail {
			return "", errors.New("bogus host / no service env var")
		}
		return "aud-1", nil
	})

	r := httptest.NewRequest("POST", "https://my-webhook.ns.svc:443/validate", nil)
	// First request: resolve fails, deny fail-closed, nothing bound.
	if err := h.ensureAudience(r); err == nil {
		t.Fatal("expected the first (failing) resolve to return an error")
	}
	if len(spy.binds) != 0 {
		t.Fatalf("nothing should be bound after a failed resolve, got %v", spy.binds)
	}
	// A later request can resolve: the handler retries, binds, and is served.
	fail = false
	if err := h.ensureAudience(r); err != nil {
		t.Fatalf("expected a successful bind after the resolver recovers, got %v", err)
	}
	// Once bound it stays bound and is not rebound on subsequent requests.
	if err := h.ensureAudience(r); err != nil {
		t.Fatalf("expected the bound audience to keep serving, got %v", err)
	}
	if len(spy.binds) != 1 || spy.binds[0] != "aud-1" {
		t.Fatalf("expected a single successful bind of %q, got %v", "aud-1", spy.binds)
	}
}

// TestEnsureAudience_NoResolverIsNoop confirms the out-of-cluster path (no
// resolver, audience already bound) is untouched.
func TestEnsureAudience_NoResolverIsNoop(t *testing.T) {
	spy := &bindSpy{}
	h := newSpyHandler(t, spy, nil)

	r := httptest.NewRequest("POST", "https://my-webhook.ns.svc:443/validate", nil)
	if err := h.ensureAudience(r); err != nil {
		t.Fatalf("ensureAudience with no resolver should be a no-op, got %v", err)
	}
	if len(spy.binds) != 0 {
		t.Fatalf("no resolver should bind nothing, got %v", spy.binds)
	}
}

// TestEnsureAudience_NoResolverFailsClosedWhenNotReady confirms the out-of-cluster
// path still fails closed: with no resolver, ensureAudience consults the backing
// verifier's readiness and denies when not ready, so a not-ready verifier can
// never silently serve.
func TestEnsureAudience_NoResolverFailsClosedWhenNotReady(t *testing.T) {
	spy := &bindSpy{healthErr: errors.New("verifier not ready")}
	h := newSpyHandler(t, spy, nil)

	r := httptest.NewRequest("POST", "https://my-webhook.ns.svc:443/validate", nil)
	if err := h.ensureAudience(r); err == nil {
		t.Fatal("expected ensureAudience to fail closed when the verifier is not ready")
	}
	if len(spy.binds) != 0 {
		t.Fatalf("no resolver should bind nothing, got %v", spy.binds)
	}
}

// TestHealthCheck_Delegates confirms Handler.HealthCheck surfaces the backing
// verifier's readiness.
func TestHealthCheck_Delegates(t *testing.T) {
	spy := &bindSpy{healthErr: errors.New("audience not yet derived")}
	h := newSpyHandler(t, spy, nil)

	if err := h.HealthCheck(); err == nil {
		t.Fatal("expected HealthCheck to surface the not-ready error")
	}
	spy.healthErr = nil
	if err := h.HealthCheck(); err != nil {
		t.Fatalf("expected ready, got %v", err)
	}
}
