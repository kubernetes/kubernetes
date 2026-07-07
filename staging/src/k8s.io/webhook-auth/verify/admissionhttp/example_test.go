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

// This file is CONSUMER GUIDANCE, not just coverage. It is a minimal,
// copy-pasteable POC showing how a real admission webhook wires up KEP-6060
// token verification against this module's PUBLIC API — nothing more. The
// runnable Example functions below are the ~10-15 lines a webhook author copies
// to get an authenticated hook; the black-box package (admissionhttp_test)
// guarantees they use ONLY the exported surface an external consumer sees.
//
// The same two shapes cover the real ecosystem:
//   - Example_rawHTTPWebhook       — a plain net/http webhook (mount our handler)
//   - Example_controllerRuntimeStyle — a decorator over an ALREADY-decoded review
//     (controller-runtime / Gatekeeper / kube-rbac-proxy: capture the bearer
//     token in HTTP middleware, then call VerifyAdmissionReview on the review the
//     framework already decoded).
//
// The ONLY production swap is the KeySet: see exampleKeySet below.
package admissionhttp_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/webhook-auth/verify"
	"k8s.io/webhook-auth/verify/admissionhttp"
)

// The verifier's config surface is exactly three values. A real deployment sets
// these to its trusted issuer, the audience minted for this webhook, and a
// KeySet that fetches signing keys. Zero-config defaults for issuer/audience are
// still pending (KEP-6060 review-1 §6); until then they are supplied explicitly.
const (
	exampleIssuer   = "https://issuer.example.com"
	exampleAudience = "webhook.example.com"
	exampleAPIGroup = "apps" // the API group this webhook's token is authorized for
)

// exampleKeySet is the ONE piece a real deployment replaces.
//
// PRODUCTION: replace exampleKeySet with the go-oidc-backed KeySet (OIDC
// discovery + JWKS); it is the ONLY piece a real deployment swaps in. Everything
// else in these examples — the Verifier, the handler, the wiring — stays as-is.
//
// For the POC it performs no crypto: it treats the raw token string as the
// already-signature-verified JSON claims payload and hands it back verbatim.
// That lets this example compile and run with stdlib only (no JOSE/JWT
// dependency), matching the fake used by the adapter's behavior tests.
type exampleKeySet struct{}

func (exampleKeySet) VerifySignature(_ context.Context, rawToken string) ([]byte, error) {
	// The raw token IS the verified claims payload in the POC. In production the
	// go-oidc KeySet parses the JWS, checks the signature against the JWKS, and
	// returns the decoded claim bytes.
	return []byte(rawToken), nil
}

// exampleToken mints the token string the POC KeySet returns verbatim as the
// verified payload. In production this is a real signed JWT the API server issues
// to the webhook's service account; here "signing" is just JSON marshaling.
//
// The claim shape is the KEP-6060 contract: standard iss/aud/exp plus a
// kubernetes.io block carrying the bound webhook configuration and the
// fully-namespaced allowedAPIGroup attestation claim.
func exampleToken(group string) string {
	claims := map[string]interface{}{
		"iss": exampleIssuer,
		"aud": []string{exampleAudience},
		"exp": time.Now().Add(5 * time.Minute).Unix(),
		"kubernetes.io": map[string]interface{}{
			// Exactly one bound object (validating XOR mutating) identifies the
			// webhook configuration the token was minted for.
			"validatingWebhookConfiguration": map[string]string{
				"name": "my-webhook",
				"uid":  "webhook-uid",
			},
			// The namespaced key is required; the bare "allowedAPIGroup" form is a
			// known issuer bug and is rejected.
			"attestationClaims": map[string][]string{
				verify.AllowedAPIGroupClaimKey: {group},
			},
		},
	}
	payload, _ := json.Marshal(claims)
	return string(payload)
}

// exampleReview builds a minimal AdmissionReview for a resource in group. A real
// webhook receives this from the API server on the wire.
func exampleReview(group string) *admissionv1.AdmissionReview {
	return &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
		Request: &admissionv1.AdmissionRequest{
			Resource: metav1.GroupVersionResource{Group: group, Version: "v1", Resource: "deployments"},
			Name:     "my-deploy",
		},
	}
}

// Example_rawHTTPWebhook is the plain net/http consumer path. A webhook author
// copies these few lines to add KEP-6060 authentication in front of existing
// admission logic. The handler decodes the body once, enforces the token, and —
// only on success — calls the downstream ReviewHandler with the decoded review.
func Example_rawHTTPWebhook() {
	// 1. Build the verifier from the three-value config surface.
	v, err := verify.NewVerifier(exampleKeySet{}, exampleIssuer, []string{exampleAudience})
	if err != nil {
		panic(err) // misconfiguration (nil KeySet / empty issuer / empty audience)
	}

	// 2. Your existing admission logic, unchanged, as a ReviewHandler. It runs
	//    ONLY after the token is verified, and receives the already-decoded review.
	admit := func(w http.ResponseWriter, _ *http.Request, review *admissionv1.AdmissionReview) {
		resp := admissionv1.AdmissionReview{
			TypeMeta: review.TypeMeta,
			Response: &admissionv1.AdmissionResponse{UID: review.Request.UID, Allowed: true},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}

	// 3. Mount the verification handler in front of it. That is the whole wiring.
	srv := httptest.NewServer(admissionhttp.WithTokenVerification(v, admit))
	defer srv.Close()

	// --- below is just a client driving the example so it produces output ---
	body, _ := json.Marshal(exampleReview(exampleAPIGroup))
	req, _ := http.NewRequest(http.MethodPost, srv.URL, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+exampleToken(exampleAPIGroup))

	resp, err := srv.Client().Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	var out admissionv1.AdmissionReview
	_ = json.NewDecoder(resp.Body).Decode(&out)
	if resp.StatusCode == http.StatusOK && out.Response != nil && out.Response.Allowed {
		fmt.Println("admitted")
	}
	// Output: admitted
}

// Example_controllerRuntimeStyle is the decoded-input path used by a decorator
// over a framework that has ALREADY decoded the review — controller-runtime,
// Gatekeeper, kube-rbac-proxy. There is no second decode: the framework hands
// you the review, and an HTTP middleware you install captures the bearer token
// into the request context. You then call VerifyAdmissionReview and branch.
func Example_controllerRuntimeStyle() {
	v, err := verify.NewVerifier(exampleKeySet{}, exampleIssuer, []string{exampleAudience})
	if err != nil {
		panic(err)
	}

	// In controller-runtime these two come from the framework + your middleware:
	//   review — the object controller-runtime already decoded for your Handle().
	//   token  — captured from Authorization by an http.Handler wrapper and read
	//            back out of the request context here.
	ctx := context.Background()
	review := exampleReview(exampleAPIGroup)
	token := exampleToken(exampleAPIGroup)

	// One call gates your admission logic. nil == verified; any error is a single
	// generic failure (use verify.Reason(err) for a non-sensitive log line).
	if err := admissionhttp.VerifyAdmissionReview(ctx, v, review, token); err != nil {
		fmt.Println("denied") // return a 401 / deny AdmissionResponse here
		return
	}
	fmt.Println("admitted") // proceed to your real admission decision
	// Output: admitted
}

// TestExampleEndToEnd_MinimalWebhook drives the minimal hook from the raw-HTTP
// example over a real httptest.Server: a valid Bearer request reaches the
// downstream handler and returns 200; an unauthenticated request is denied with
// 401 and the downstream handler is never reached. This is the end-to-end proof
// behind Example_rawHTTPWebhook.
func TestExampleEndToEnd_MinimalWebhook(t *testing.T) {
	v, err := verify.NewVerifier(exampleKeySet{}, exampleIssuer, []string{exampleAudience})
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}

	var reached bool
	admit := func(w http.ResponseWriter, _ *http.Request, _ *admissionv1.AdmissionReview) {
		reached = true
		w.WriteHeader(http.StatusOK)
	}
	srv := httptest.NewServer(admissionhttp.WithTokenVerification(v, admit))
	defer srv.Close()

	tests := []struct {
		name            string
		bearer          string // empty means no Authorization header
		wantStatus      int
		wantNextReached bool
	}{
		{
			name:            "valid bearer token -> 200, downstream reached",
			bearer:          exampleToken(exampleAPIGroup),
			wantStatus:      http.StatusOK,
			wantNextReached: true,
		},
		{
			name:            "no bearer token -> 401, downstream NOT reached",
			bearer:          "",
			wantStatus:      http.StatusUnauthorized,
			wantNextReached: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			reached = false

			body, err := json.Marshal(exampleReview(exampleAPIGroup))
			if err != nil {
				t.Fatalf("marshal review: %v", err)
			}
			req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, srv.URL, bytes.NewReader(body))
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			req.Header.Set("Content-Type", "application/json")
			if tc.bearer != "" {
				req.Header.Set("Authorization", "Bearer "+tc.bearer)
			}

			resp, err := srv.Client().Do(req)
			if err != nil {
				t.Fatalf("do request: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != tc.wantStatus {
				t.Errorf("status = %d, want %d", resp.StatusCode, tc.wantStatus)
			}
			if reached != tc.wantNextReached {
				t.Errorf("downstream reached = %v, want %v", reached, tc.wantNextReached)
			}
		})
	}
}
