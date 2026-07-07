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
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"k8s.io/webhook-auth/verify"
)

// deriveAudience is the single source of the PROVISIONAL webhook-audience
// derivation used by both the request-time helper (DeriveExpectedAudience) and
// the construction-time helper (AudienceFromServiceURL). It maps a webhook's own
// host and path to the audience the issuer is expected to have stamped into the
// token.
//
// PROVISIONAL — the exact audience-derivation format is an OPEN QUESTION on the
// issuer (kube-apiserver) side and is NOT finalized in KEP-6060. This
// "https://<host><path>" shape is a reasonable default so a webhook works with
// zero explicit audience configuration; it MUST be reconciled with, and match,
// the kube-apiserver-side derivation once the KEP finalizes it. Until then,
// deployments that know their audience should override it explicitly.
func deriveAudience(host, path string) string {
	if path == "" {
		path = "/"
	}
	return "https://" + host + path
}

// DeriveExpectedAudience derives the PROVISIONAL expected audience for a webhook
// request from its own r.Host and r.URL.Path. It is the request-time counterpart
// of AudienceFromServiceURL and shares the same derivation formula.
//
// PROVISIONAL — must match the issuer-side (kube-apiserver) audience derivation
// once KEP-6060 finalizes it. Prefer an explicitly configured audience where the
// expected value is known.
func DeriveExpectedAudience(r *http.Request) string {
	return deriveAudience(r.Host, r.URL.Path)
}

// AudienceFromServiceURL derives the PROVISIONAL expected audience from a
// webhook's own service URL (for example the URL registered in the
// ValidatingWebhookConfiguration clientConfig). It is the construction-time
// counterpart of DeriveExpectedAudience: feed its result into
// verify.NewVerifier (or oidckeyset.NewInClusterVerifier via WithServiceURL) so
// the strict verifier path is configured with the same value a request would
// derive.
//
// PROVISIONAL — must match the issuer-side (kube-apiserver) audience derivation
// once KEP-6060 finalizes it.
func AudienceFromServiceURL(rawURL string) (string, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", fmt.Errorf("admissionhttp: parsing service URL %q: %w", rawURL, err)
	}
	if u.Host == "" {
		return "", fmt.Errorf("admissionhttp: service URL %q has no host", rawURL)
	}
	return deriveAudience(u.Host, u.Path), nil
}

// VerifierFactory builds a Verifier bound to expectedAudiences. It exists so a
// handler can defer audience selection to request time (see
// WithTokenVerificationDerivedAudience) while still going through the strict
// verify.NewVerifier path: a factory typically closes over the KeySet and issuer
// and returns verify.NewVerifier(keySet, issuer, expectedAudiences, opts...).
type VerifierFactory func(expectedAudiences []string) (*verify.Verifier, error)

// derivedAudienceHandler builds and caches a per-endpoint Verifier on first use,
// deriving the expected audience from the request. It wraps the standard
// WithTokenVerification handler for each distinct derived audience, so all
// enforcement (single decode, generic denial, fail-closed) is unchanged — only
// the audience the Verifier is constructed with is chosen at request time.
type derivedAudienceHandler struct {
	factory VerifierFactory
	derive  func(*http.Request) []string
	next    ReviewHandler
	opts    []Option

	mu    sync.Mutex
	cache map[string]http.Handler
}

// WithTokenVerificationDerivedAudience returns an http.Handler that derives the
// expected audience from each request (via DeriveExpectedAudience, PROVISIONAL),
// builds a Verifier for that audience through factory the first time it is seen,
// caches it per derived audience, and then enforces exactly like
// WithTokenVerification.
//
// This is the "use it at request time" path for the provisional audience
// derivation. It does NOT weaken core validation: the factory goes through the
// strict verify.NewVerifier, and a factory error fails the request closed with
// the same generic denial. The explicit WithTokenVerification path (fixed,
// caller-supplied audiences) remains the recommended choice wherever the
// expected audience is known ahead of time.
func WithTokenVerificationDerivedAudience(factory VerifierFactory, next ReviewHandler, opts ...Option) http.Handler {
	return &derivedAudienceHandler{
		factory: factory,
		derive:  func(r *http.Request) []string { return []string{DeriveExpectedAudience(r)} },
		next:    next,
		opts:    opts,
		cache:   make(map[string]http.Handler),
	}
}

// ServeHTTP implements http.Handler.
func (d *derivedAudienceHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	auds := d.derive(r)
	key := strings.Join(auds, "\x00")

	d.mu.Lock()
	h := d.cache[key]
	if h == nil {
		v, err := d.factory(auds)
		if err != nil {
			d.mu.Unlock()
			// Fail closed with the uniform generic denial: a misconfigured
			// factory must never fall through to the downstream handler.
			writeDenied(w)
			return
		}
		h = WithTokenVerification(v, d.next, d.opts...)
		d.cache[key] = h
	}
	d.mu.Unlock()

	h.ServeHTTP(w, r)
}
