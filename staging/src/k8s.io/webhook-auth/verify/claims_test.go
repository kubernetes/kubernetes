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
	"strings"
	"testing"
)

// TestVerifiedClaims_DecodeKubernetesKey locks the JSON wire contract the
// authenticator relies on: the webhook private claims live under the standard
// "kubernetes.io" key, and the attestation claim is keyed by the exact
// fully-namespaced string KEP-6060 mandates. Decoding a spec-shaped payload must
// populate the bound-object ref and the namespaced allowedAPIGroup claim.
func TestVerifiedClaims_DecodeKubernetesKey(t *testing.T) {
	// The literal wire strings a spec-compliant issuer emits. The claim key is
	// hard-coded here (not via the unexported constant) so this test fails loudly
	// if the constant's VALUE is ever changed away from the KEP-mandated string.
	const payload = `{
		"iss": "https://issuer.example.com",
		"sub": "system:serviceaccount:kube-system:webhook-auth",
		"aud": ["https://webhook.example.svc/validate"],
		"kubernetes.io": {
			"validatingWebhookConfiguration": {"name": "vwc", "uid": "vwc-uid"},
			"attestationClaims": {
				"webhook-authentication.k8s.io/allowedAPIGroup": ["apps"]
			}
		}
	}`

	var claims VerifiedClaims
	if err := json.Unmarshal([]byte(payload), &claims); err != nil {
		t.Fatalf("decoding claims: %v", err)
	}

	k := claims.Kubernetes
	if k.ValidatingWebhookConfiguration == nil {
		t.Fatal("validatingWebhookConfiguration did not decode")
	}
	if k.ValidatingWebhookConfiguration.Name != "vwc" || k.ValidatingWebhookConfiguration.UID != "vwc-uid" {
		t.Errorf("bound ref = %+v, want vwc/vwc-uid", k.ValidatingWebhookConfiguration)
	}
	if k.MutatingWebhookConfiguration != nil {
		t.Errorf("mutatingWebhookConfiguration should be absent, got %+v", k.MutatingWebhookConfiguration)
	}

	groups, ok := k.AttestationClaims[allowedAPIGroupClaimKey]
	if !ok {
		t.Fatalf("attestation claim not found under the namespaced key %q; keys present: %v",
			allowedAPIGroupClaimKey, k.AttestationClaims)
	}
	if len(groups) != 1 || groups[0] != "apps" {
		t.Errorf("allowedAPIGroup = %v, want [apps]", groups)
	}
}

// TestVerifiedClaims_BareKeyDoesNotPopulateNamespacedKey proves the security
// property behind FLAG 1: a spec-violating issuer that emits the BARE
// "allowedAPIGroup" key does not accidentally satisfy the namespaced lookup. The
// bare key decodes into the map under its own name, so the namespaced lookup the
// verifier performs misses it — the token is treated as missing the claim, never
// silently accepted.
func TestVerifiedClaims_BareKeyDoesNotPopulateNamespacedKey(t *testing.T) {
	const payload = `{
		"kubernetes.io": {
			"validatingWebhookConfiguration": {"name": "vwc", "uid": "vwc-uid"},
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

// TestKubernetesClaims_MutatingBound confirms the mutating side of the
// bound-object shape decodes symmetrically and, with omitempty, leaves the
// validating ref nil so presence alone distinguishes the two.
func TestKubernetesClaims_MutatingBound(t *testing.T) {
	const payload = `{
		"kubernetes.io": {
			"mutatingWebhookConfiguration": {"name": "mwc", "uid": "mwc-uid"}
		}
	}`

	var claims VerifiedClaims
	if err := json.Unmarshal([]byte(payload), &claims); err != nil {
		t.Fatalf("decoding claims: %v", err)
	}
	if claims.Kubernetes.ValidatingWebhookConfiguration != nil {
		t.Error("validating ref should be nil when only the mutating ref is present")
	}
	if claims.Kubernetes.MutatingWebhookConfiguration == nil {
		t.Fatal("mutatingWebhookConfiguration did not decode")
	}
	if got := claims.Kubernetes.MutatingWebhookConfiguration.Name; got != "mwc" {
		t.Errorf("mutating ref name = %q, want mwc", got)
	}
}

// TestKubernetesClaims_OmitemptyMarshalRoundTrip asserts the omitempty tags on
// both webhook refs: a claim set with only the validating ref set must marshal
// WITHOUT a mutating key, so the presence-based exactly-one rule the verifier
// enforces is not defeated by an emitted-but-null mutating field.
func TestKubernetesClaims_OmitemptyMarshalRoundTrip(t *testing.T) {
	k := kubernetesClaims{
		ValidatingWebhookConfiguration: &objectRef{Name: "vwc", UID: "vwc-uid"},
	}
	b, err := json.Marshal(k)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	out := string(b)
	if want := `"validatingWebhookConfiguration"`; !strings.Contains(out, want) {
		t.Errorf("marshaled claims %q missing %s", out, want)
	}
	if bad := `"mutatingWebhookConfiguration"`; strings.Contains(out, bad) {
		t.Errorf("marshaled claims %q should omit %s", out, bad)
	}
}
