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
// that kube-apiserver presents to admission webhooks under KEP-6060
// (API server authentication to admission webhooks).
//
// A webhook uses this package to validate a presented token WITHOUT calling
// TokenReview. The verifier enforces the KEP-6060 token contract:
//
//  1. The JWS signature is valid against the cluster issuer's keys. Signature
//     verification is delegated to a [KeySet] so that the core logic carries no
//     third-party crypto dependency and can be backed by a static key set (for
//     tests) or an OIDC/JWKS-fetching implementation (a follow-up).
//  2. The token audience contains the webhook's expected audience.
//  3. The token is bound to exactly one of the validating- or
//     mutating-webhook configuration references (never both, never neither).
//  4. The attestation claim keyed by the fully-namespaced string
//     [AllowedAPIGroupClaimKey] carries exactly one API group.
//  5. That API group is either [WildcardAPIGroup] (matches all) or exactly
//     equals the API group of the resource in the incoming AdmissionReview.
//  6. The token is within its exp / nbf validity window.
//
// Anti-enumeration: all verification failures surface the single generic
// [ErrVerificationFailed] and never echo webhook names, UIDs, subjects, or
// API-group values, so a caller cannot probe for the existence of a specific
// webhook, group, or object. Callers must not branch on why verification
// failed; the specific reason is available only as a non-sensitive log string
// via [Reason] for operators and debugging.
//
// This first cut deliberately omits the http.Handler adapter and the
// controller-runtime decorator; callers wire [Verifier.Verify] into their own
// admission entry point. See the package README / KEP-6060 §9 for the intended
// integration points.
package verify // import "k8s.io/webhook-auth/verify"
