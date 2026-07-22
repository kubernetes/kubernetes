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

import "k8s.io/webhookauth/internal/verify"

// NewHandlerForTest exposes the internal newHandler seam so tests can assemble a
// Handler around an already-built verify.Verifier without going through
// WithTokenVerification (which constructs its own verifier via OIDC discovery).
// resolve is nil for the out-of-cluster path and an audienceResolver for the
// in-cluster deferred path. Options are applied through the same builder the
// exported constructor uses, so handler-affecting options (for example
// WithMaxBodyBytes) behave identically here.
func NewHandlerForTest(v *verify.Verifier, next AdmissionHandler, resolve audienceResolver, opts ...Option) *Handler {
	b := newBuilder()
	for _, opt := range opts {
		opt(b)
	}
	return newHandler(v, next, resolve, b.maxBody)
}

// NewVerifierForTest exposes the internal Verifier wrapper so tests can exercise
// [Verifier.Verify] against an already-built verify.Verifier without going
// through NewVerifier (which constructs its own verifier via OIDC discovery).
func NewVerifierForTest(v *verify.Verifier) *Verifier {
	return &Verifier{verifier: v}
}

// WithInClusterEndpointForTest exposes the unexported withInClusterEndpoint option
// so tests can drive the REAL WithTokenVerification in-cluster branch offline,
// redirecting it at a throwaway apiserver and HTTP client instead of
// oidc.InClusterAPIServerURL and the projected service-account CA client. It stays
// in-cluster mode (it does not select remote), closing the in-cluster e2e gap.
var WithInClusterEndpointForTest = withInClusterEndpoint
