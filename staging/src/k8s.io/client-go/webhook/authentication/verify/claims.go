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
	"encoding/json"
	"time"
)

// ObjectRef identifies a bound Kubernetes object by name and UID, mirroring the
// { "name": ..., "uid": ... } refs the issuer places under the kubernetes.io
// private claims for the validating- or mutating-webhook configuration.
type ObjectRef struct {
	Name string `json:"name"`
	UID  string `json:"uid"`
}

// kubernetesClaims mirrors the subset of the standard "kubernetes.io" private
// claims object relevant to webhook authentication. Both webhook references use
// omitempty so the bound-object rule (exactly one) can be enforced by presence.
type kubernetesClaims struct {
	ValidatingWebhookConfiguration *ObjectRef `json:"validatingWebhookConfiguration,omitempty"`
	MutatingWebhookConfiguration   *ObjectRef `json:"mutatingWebhookConfiguration,omitempty"`

	// AttestationClaims carries the attestation values keyed by their
	// fully-namespaced claim key. For webhook authentication the only key we
	// consult is AllowedAPIGroupClaimKey. The JSON wire shape is a plain
	// map[string][]string, matching the server-side map[string]AttestationClaimValue.
	AttestationClaims map[string][]string `json:"attestationClaims,omitempty"`
}

// tokenClaims is the parsed, verified JWT payload. Only the fields this verifier
// needs are modeled; unknown claims are ignored.
type tokenClaims struct {
	Issuer    string       `json:"iss,omitempty"`
	Subject   string       `json:"sub,omitempty"`
	Audience  audience     `json:"aud,omitempty"`
	Expiry    *numericDate `json:"exp,omitempty"`
	NotBefore *numericDate `json:"nbf,omitempty"`
	IssuedAt  *numericDate `json:"iat,omitempty"`

	Kubernetes *kubernetesClaims `json:"kubernetes.io,omitempty"`
}

// audience models the JWT "aud" claim, which per RFC 7519 may be encoded either
// as a single string or as an array of strings. Both forms decode to a slice.
type audience []string

func (a *audience) UnmarshalJSON(b []byte) error {
	// Try the array form first.
	var list []string
	if err := json.Unmarshal(b, &list); err == nil {
		*a = list
		return nil
	}
	// Fall back to the single-string form.
	var single string
	if err := json.Unmarshal(b, &single); err != nil {
		return err
	}
	*a = audience{single}
	return nil
}

// contains reports whether the audience list includes want.
func (a audience) contains(want string) bool {
	for _, aud := range a {
		if aud == want {
			return true
		}
	}
	return false
}

// numericDate models a JWT NumericDate (RFC 7519 §2): seconds since the Unix
// epoch, potentially fractional. Fractions are truncated to whole seconds.
type numericDate int64

func (n *numericDate) UnmarshalJSON(b []byte) error {
	var f float64
	if err := json.Unmarshal(b, &f); err != nil {
		return err
	}
	*n = numericDate(int64(f))
	return nil
}

func (n numericDate) time() time.Time {
	return time.Unix(int64(n), 0)
}
