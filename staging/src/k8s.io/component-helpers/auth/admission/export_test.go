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

package admission

import "k8s.io/component-helpers/auth/admission/internal/verify"

// NewHandlerForTest lets black-box tests use a real core verifier while the
// HTTP adapter is exercised without rebuilding its OIDC discovery fixture.
func NewHandlerForTest(v *verify.Verifier, next AdmissionHandler, opts ...Option) *Handler {
	b := newBuilder()
	for _, opt := range opts {
		opt(b)
	}
	return newHandler(v, next, b.maxBody)
}

// NewVerifierForTest exposes the internal Verifier wrapper so tests can exercise
// [Verifier.Verify] against an already-built verify.Verifier without going
// through NewVerifier (which constructs its own verifier via OIDC discovery).
func NewVerifierForTest(v *verify.Verifier) *Verifier {
	return &Verifier{verifier: v}
}

// WithInClusterEndpointForTest redirects in-cluster issuer discovery and JWKS
// calls to a TLS test server without bypassing audience or signature checks.
var WithInClusterEndpointForTest = withInClusterEndpoint

// WithMaxBodyBytesForTest tests the fixed production body limit with small inputs.
var WithMaxBodyBytesForTest = withMaxBodyBytes
