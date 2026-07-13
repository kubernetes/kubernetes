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

// VerifiedClaims is the signature- and standard-claim-verified view of a token
// that a TokenAuthenticator returns. The authenticator has already checked the
// signature and the standard iss/aud/exp claims (for example via OIDC discovery
// and go-oidc); the policy layer applies only the KEP-6060 checks on top.
//
// It is the single value that crosses the boundary from the (dependency-heavy)
// authenticator into this pure-stdlib policy package, so all JOSE/OIDC types
// stay out of the core verifier. The webhook-specific private claims are decoded
// under the standard "kubernetes.io" key.
type VerifiedClaims struct {
	// Issuer, Subject and Audience are the verified standard claims. The
	// authenticator populates them from its validated token, so they carry the
	// values the signature and standard-claim checks actually vouched for.
	Issuer   string
	Subject  string
	Audience []string

	// Kubernetes holds the decoded "kubernetes.io" private claims relevant to
	// webhook authentication. Its type is unexported: the only fields the policy
	// consults are reached within this package, and an authenticator populates
	// it by JSON-decoding the token payload, so it never needs to name the type.
	//
	// A bring-your-own TokenAuthenticator populates this field by JSON-decoding
	// the verified token payload into a VerifiedClaims value — for example
	// idToken.Claims(&VerifiedClaims{}) with go-oidc, or json.Unmarshal of the
	// payload — since the "kubernetes.io" JSON key routes into it automatically.
	// The unexported field type cannot be constructed by name from outside this
	// package, which is deliberate: JSON decoding is the only supported way in.
	Kubernetes kubernetesClaims `json:"kubernetes.io"`
}

// kubernetesClaims mirrors the subset of the standard "kubernetes.io" private
// claims object relevant to webhook authentication. Both webhook references use
// omitempty so the bound-object rule (exactly one) can be enforced by presence.
type kubernetesClaims struct {
	ValidatingWebhookConfiguration *objectRef `json:"validatingWebhookConfiguration,omitempty"`
	MutatingWebhookConfiguration   *objectRef `json:"mutatingWebhookConfiguration,omitempty"`

	// AttestationClaims carries the attestation values keyed by their
	// fully-namespaced claim key. For webhook authentication the only key we
	// consult is allowedAPIGroupClaimKey. The JSON wire shape is a plain
	// map[string][]string, matching the server-side map[string]AttestationClaimValue.
	AttestationClaims map[string][]string `json:"attestationClaims,omitempty"`
}

// objectRef identifies a bound Kubernetes object by name and UID, mirroring the
// { "name": ..., "uid": ... } refs the issuer places under the kubernetes.io
// private claims for the validating- or mutating-webhook configuration.
type objectRef struct {
	Name string `json:"name"`
	UID  string `json:"uid"`
}
