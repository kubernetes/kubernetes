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
//     token in HTTP middleware, then call VerifyAdmissionRequest on the request
//     the framework already decoded).
//
// The ONLY production change is pointing oidc.NewRemoteVerifier at the
// cluster's real OIDC issuer instead of the throwaway TLS issuer these examples
// stand up. Signatures are verified for real; there is no insecure path.
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
	"k8s.io/webhookauth/verify/admissionhttp"
	"k8s.io/webhookauth/verify/oidc"
)

// The verifier's config surface is the issuer, the audience minted for this
// webhook, and (in-cluster) an *http.Client that trusts the issuer's serving CA.
const (
	exampleAudience = "webhook.example.com"
	exampleAPIGroup = "apps" // the API group this webhook's token is authorized for
)

// exampleSignedToken mints a real RS256 JWT signed by the throwaway issuer,
// carrying the KEP-6060 claim shape: standard iss/aud/exp plus a kubernetes.io
// block with the bound webhook configuration and the fully-namespaced
// allowedAPIGroup attestation claim. In production the API server issues this
// token to the webhook's service account.
func exampleSignedToken(issuer *oidcTestServer, group string) string {
	claims := map[string]any{
		"iss": issuer.issuer,
		"aud": []string{exampleAudience},
		"exp": time.Now().Add(5 * time.Minute).Unix(),
		"nbf": time.Now().Add(-1 * time.Minute).Unix(),
		"kubernetes.io": map[string]any{
			// Exactly one bound object (validating XOR mutating) identifies the
			// webhook configuration the token was minted for.
			"validatingWebhookConfiguration": map[string]any{
				"name": "my-webhook",
				"uid":  "webhook-uid",
			},
			// The namespaced key is required; the bare "allowedAPIGroup" form is a
			// known issuer bug and is rejected.
			"attestationClaims": map[string]any{
				allowedAPIGroupClaimKey: []string{group},
			},
		},
	}
	token, err := issuer.signClaims(claims)
	if err != nil {
		panic(err)
	}
	return token
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
// only on success — calls the downstream AdmissionHandler with the decoded request.
func Example_rawHTTPWebhook() {
	// PRODUCTION: point NewRemoteVerifier at the cluster's OIDC issuer. The
	// example stands up a throwaway TLS issuer so it verifies REAL signatures.
	issuer, err := startOIDCServer()
	if err != nil {
		panic(err)
	}
	defer issuer.close()

	// 1. Build the verifier from the trusted issuer and the audience minted for
	//    this webhook. WithHTTPClient supplies a client that trusts the issuer's
	//    serving CA (in-cluster: the mounted apiserver CA bundle).
	v, err := oidc.NewRemoteVerifier(context.Background(), issuer.issuer, exampleAudience, oidc.WithHTTPClient(issuer.client()))
	if err != nil {
		panic(err) // misconfiguration (empty issuer/audience) or discovery failure
	}

	// 2. Your existing admission logic, unchanged, as an AdmissionHandler. It runs
	//    ONLY after the token is verified, receives the already-decoded request,
	//    and returns the response (the adapter wraps and writes it).
	admit := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}

	// 3. Mount the verification handler in front of it. That is the whole wiring.
	srv := httptest.NewServer(admissionhttp.WithTokenVerification(v, admit))
	defer srv.Close()

	// --- below is just a client driving the example so it produces output ---
	body, _ := json.Marshal(exampleReview(exampleAPIGroup))
	req, _ := http.NewRequest(http.MethodPost, srv.URL, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+exampleSignedToken(issuer, exampleAPIGroup))

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
// you the request, and an HTTP middleware you install captures the bearer token
// into the request context. You then call VerifyAdmissionRequest and branch.
func Example_controllerRuntimeStyle() {
	issuer, err := startOIDCServer()
	if err != nil {
		panic(err)
	}
	defer issuer.close()

	v, err := oidc.NewRemoteVerifier(context.Background(), issuer.issuer, exampleAudience, oidc.WithHTTPClient(issuer.client()))
	if err != nil {
		panic(err)
	}

	// In controller-runtime these two come from the framework + your middleware:
	//   review — the object controller-runtime already decoded for your Handle().
	//   token  — captured from Authorization by an http.Handler wrapper and read
	//            back out of the request context here.
	ctx := context.Background()
	review := exampleReview(exampleAPIGroup)
	token := exampleSignedToken(issuer, exampleAPIGroup)

	// One call gates your admission logic. nil == verified; any error is a single
	// generic failure (the reason is logged internally; do not branch on it).
	if err := admissionhttp.VerifyAdmissionRequest(ctx, v, review.Request, token); err != nil {
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
	ts := newOIDCTestServer(t)
	v, err := oidc.NewRemoteVerifier(context.Background(), ts.issuer, exampleAudience, oidc.WithHTTPClient(ts.client()))
	if err != nil {
		t.Fatalf("NewRemoteVerifier: %v", err)
	}

	var reached bool
	admit := func(_ context.Context, _ *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		reached = true
		return &admissionv1.AdmissionResponse{Allowed: true}
	}
	srv := httptest.NewServer(admissionhttp.WithTokenVerification(v, admit))
	defer srv.Close()

	validToken := exampleSignedToken(ts, exampleAPIGroup)

	tests := []struct {
		name            string
		bearer          string // empty means no Authorization header
		wantStatus      int
		wantNextReached bool
	}{
		{
			name:            "valid bearer token -> 200, downstream reached",
			bearer:          validToken,
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
