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
	"testing"
)

// TODO(kep-6060): rebuild the removed claims tests after review — bound-object
// decode (TestKubernetesClaims_MutatingBound) and the omitempty marshal
// round-trip were dropped with the bound-object fields. See
// kep-6060-review-2.2-actions.md ("Tests to rebuild").

// TestVerifiedClaims_DecodeKubernetesKey locks the JSON wire contract: the
// webhook private claims live under the standard "kubernetes.io" key and the
// attestation claim is keyed by the exact fully-namespaced string KEP-6060
// mandates. The claim key is hard-coded here (not via the unexported constant)
// so this test fails loudly if the constant's VALUE ever changes.
func TestVerifiedClaims_DecodeKubernetesKey(t *testing.T) {
	const payload = `{
		"iss": "https://issuer.example.com",
		"sub": "system:serviceaccount:kube-system:webhook-auth",
		"aud": ["https://webhook.example.svc/validate"],
		"kubernetes.io": {
			"attestationClaims": {
				"webhook-authentication.k8s.io/allowedAPIGroup": ["apps"]
			}
		}
	}`

	var claims VerifiedClaims
	if err := json.Unmarshal([]byte(payload), &claims); err != nil {
		t.Fatalf("decoding claims: %v", err)
	}

	groups, ok := claims.Kubernetes.AttestationClaims[allowedAPIGroupClaimKey]
	if !ok {
		t.Fatalf("attestation claim not found under the namespaced key %q; keys present: %v",
			allowedAPIGroupClaimKey, claims.Kubernetes.AttestationClaims)
	}
	if len(groups) != 1 || groups[0] != "apps" {
		t.Errorf("allowedAPIGroup = %v, want [apps]", groups)
	}
}

// TestVerifiedClaims_BareKeyDoesNotPopulateNamespacedKey proves a spec-violating
// issuer that emits the BARE "allowedAPIGroup" key does not accidentally satisfy
// the namespaced lookup: it decodes under its own literal name, so the verifier
// treats the token as missing the claim, never silently accepted.
func TestVerifiedClaims_BareKeyDoesNotPopulateNamespacedKey(t *testing.T) {
	const payload = `{
		"kubernetes.io": {
			"attestationClaims": {"allowedAPIGroup": ["apps"]}
		}
	}`

	var claims VerifiedClaims
	if err := json.Unmarshal([]byte(payload), &claims); err != nil {
		t.Fatalf("decoding claims: %v", err)
	}
	if _, ok := claims.Kubernetes.AttestationClaims[allowedAPIGroupClaimKey]; ok {
		t.Error("bare allowedAPIGroup key must NOT satisfy the namespaced lookup")
	}
	if _, ok := claims.Kubernetes.AttestationClaims["allowedAPIGroup"]; !ok {
		t.Error("expected the bare key to decode under its own literal name")
	}
}
