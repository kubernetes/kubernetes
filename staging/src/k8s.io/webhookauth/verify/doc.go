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

// Package verify implements offline verification of the service-account tokens
// that kube-apiserver presents to admission webhooks under KEP-6060 (API server
// authentication to admission webhooks).
//
// A webhook uses this package to validate a presented token WITHOUT calling
// TokenReview. The verifier enforces the KEP-6060 token contract:
//
//  1. The JWS signature and the standard issuer/audience/expiry claims are valid.
//     This is delegated to a [TokenAuthenticator] (see the oidc package, built on
//     OIDC discovery and go-oidc), so the core policy carries no JOSE/OIDC
//     dependency.
//  2. The namespaced allowedAPIGroup attestation claim authorizes the API group
//     of the resource under admission — its list contains that group or "*".
//
// The policy this package owns is exactly the part go-oidc has no concept of: the
// allowedAPIGroup match.
//
// Anti-enumeration: every failure surfaces the single generic
// [ErrVerificationFailed] and never echoes webhook names, UIDs, subjects, or
// group values; the specific reason is logged via the context logger only.
// Callers must not branch on why verification failed.
//
// This package omits the http.Handler adapter and controller-runtime decorator;
// callers wire [Verifier.Verify] into their own admission entry point.
package verify // import "k8s.io/webhookauth/verify"
