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
// The ONLY production change is pointing the verifier at the cluster's real OIDC
// issuer (RemoteConfig.Issuer for the raw-HTTP path, oidc.NewRemoteVerifier for
// the controller-runtime path) instead of the throwaway TLS issuer these examples
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
	"k8s.io/webhookauth/verify"
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
// admissionReviewAPIGroups attestation claim. In production the API server issues this
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
			// The namespaced key is required; the bare "admissionReviewAPIGroups" form is a
			// known issuer bug and is rejected.
			"attestations": map[string]any{
				admissionReviewAPIGroupsClaimKey: []string{group},
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
	// PRODUCTION: point RemoteConfig.Issuer at the cluster's OIDC issuer. The
	// example stands up a throwaway TLS issuer so it verifies REAL signatures.
	issuer, err := startOIDCServer()
	if err != nil {
		panic(err)
	}
	defer issuer.close()

	// 1. Your existing admission logic, unchanged, as an AdmissionHandler. It runs
	//    ONLY after the token is verified, receives the already-decoded request,
	//    and returns the response (the adapter wraps and writes it).
	admit := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}

	// 2. Build the verification handler in one call: it constructs the verifier
	//    from the trusted issuer and the audience minted for this webhook, and
	//    wraps admit. RemoteConfig.HTTPClient supplies a client that trusts the
	//    issuer's serving CA (in-cluster: the mounted apiserver CA bundle). Omit
	//    WithRemoteConfig entirely for the zero-config in-cluster path.
	h, err := admissionhttp.WithTokenVerification(context.Background(), admit, admissionhttp.WithRemoteConfig(admissionhttp.RemoteConfig{
		Issuer:     issuer.issuer,
		Audience:   exampleAudience,
		HTTPClient: issuer.client(),
	}))
	if err != nil {
		panic(err) // misconfiguration (empty issuer/audience) or discovery failure
	}

	// 3. Mount it. That is the whole wiring.
	srv := httptest.NewServer(h)
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

	var reached bool
	admit := func(_ context.Context, _ *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		reached = true
		return &admissionv1.AdmissionResponse{Allowed: true}
	}
	h, err := admissionhttp.WithTokenVerification(context.Background(), admit, admissionhttp.WithRemoteConfig(admissionhttp.RemoteConfig{
		Issuer:     ts.issuer,
		Audience:   exampleAudience,
		HTTPClient: ts.client(),
	}))
	if err != nil {
		t.Fatalf("WithTokenVerification: %v", err)
	}
	srv := httptest.NewServer(h)
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

// TestExampleEndToEnd_InClusterDeferredWebhook is the in-cluster counterpart to
// TestExampleEndToEnd_MinimalWebhook, proving the DEFERRED in-cluster behavior
// end-to-end through the REAL zero-config entrypoint. WithTokenVerification with
// no remote option builds the in-cluster verifier (local discovery + JWKS,
// audience UNKNOWN at startup) behind the handler with InClusterAudienceResolver.
// The handler binds the audience from the FIRST request, then verifies; readiness
// flips not-ready → ready across it, and a later unauthenticated request is denied
// fail-closed.
//
// This is the zero-config path production assembles from WithTokenVerification(ctx,
// admit): no WithRemoteConfig, so the in-cluster branch runs. WithInClusterEndpointForTest
// redirects that branch at a throwaway TLS apiserver double and its client while
// STAYING in-cluster mode (it does not select remote), so the ACTUAL entrypoint runs
// offline and verifies REAL signatures — the default path only swaps
// oidc.InClusterAPIServerURL and the projected service-account CA client.
func TestExampleEndToEnd_InClusterDeferredWebhook(t *testing.T) {
	ts := newOIDCTestServer(t)

	// The Service backing this webhook. InClusterAudienceResolver derives the
	// audience as https://<name>.<namespace>.svc:<port><path>, reading the port
	// from the kubelet-injected <NAME>_SERVICE_PORT env var (also a trust anchor:
	// only a real Service in this pod's namespace has one).
	const svcName, svcNamespace = "webhook", "default"
	const svcPort int32 = 443
	t.Setenv("WEBHOOK_SERVICE_PORT", "443")
	serviceHost := svcName + "." + svcNamespace + ".svc" // the DNS name the apiserver dials
	expectedAudience := verify.AudienceForService(svcName, svcNamespace, svcPort, "/")

	// The apiserver mints this token for the webhook's ServiceAccount; its audience
	// is the derived Service audience, and it attests the exampleAPIGroup.
	claims := ts.baseClaims()
	claims["aud"] = []string{expectedAudience}
	token := ts.sign(t, claims)

	var reached bool
	admit := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		reached = true
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}

	// The whole in-cluster wiring, through the REAL entrypoint: with no remote
	// option WithTokenVerification builds the deferred in-cluster verifier and the
	// resolver — the same assembly production gets from WithTokenVerification(ctx,
	// admit). WithInClusterEndpointForTest redirects the in-cluster branch at the
	// throwaway apiserver double instead of oidc.InClusterAPIServerURL + the SA-CA
	// client, staying in-cluster so the test runs offline.
	h, err := admissionhttp.WithTokenVerification(context.Background(), admit,
		admissionhttp.WithInClusterEndpointForTest(ts.issuer, ts.client()))
	if err != nil {
		t.Fatalf("WithTokenVerification: %v", err)
	}
	srv := httptest.NewServer(h)
	defer srv.Close()

	// Before any request the audience is unbound, so the handler is not ready —
	// the seam a controller-runtime readiness check wires into.
	if err := h.HealthCheck(); err == nil {
		t.Fatal("expected not-ready before the first request binds the audience")
	}

	do := func(bearer string) *http.Response {
		body, err := json.Marshal(exampleReview(exampleAPIGroup))
		if err != nil {
			t.Fatalf("marshal review: %v", err)
		}
		// POST to "/" so the resolver derives the same path the token audience uses.
		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, srv.URL+"/", bytes.NewReader(body))
		if err != nil {
			t.Fatalf("new request: %v", err)
		}
		req.Host = serviceHost // the apiserver dials the Service DNS name
		req.Header.Set("Content-Type", "application/json")
		if bearer != "" {
			req.Header.Set("Authorization", "Bearer "+bearer)
		}
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatalf("do request: %v", err)
		}
		return resp
	}

	// First request with a valid token: the audience is derived and bound, the
	// token is verified, the downstream is reached, and the status is 200.
	reached = false
	resp := do(token)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("valid in-cluster request status = %d, want %d", resp.StatusCode, http.StatusOK)
	}
	if !reached {
		t.Error("downstream not reached for a valid in-cluster request")
	}
	// The first request bound the audience, so the handler is now ready.
	if err := h.HealthCheck(); err != nil {
		t.Errorf("expected ready after the first request bound the audience, got %v", err)
	}

	// A subsequent unauthenticated request is denied fail-closed with 401 and
	// never reaches the downstream handler.
	reached = false
	respNoToken := do("")
	defer respNoToken.Body.Close()
	if respNoToken.StatusCode != http.StatusUnauthorized {
		t.Errorf("no-token request status = %d, want %d", respNoToken.StatusCode, http.StatusUnauthorized)
	}
	if reached {
		t.Error("downstream reached for an unauthenticated request")
	}
}
