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

// The examples use the public admission API. Remote examples verify signed
// tokens from a test issuer; in-cluster verification still needs a trusted audience.
package admission_test

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
	"k8s.io/component-helpers/auth/admission"
	"k8s.io/component-helpers/auth/admission/internal/verify"
)

const (
	exampleAudience = "webhook.example.com"
	exampleAPIGroup = "apps" // the API group this webhook's token is authorized for
)

// exampleSignedToken signs a webhook-bound token with the test issuer.
func exampleSignedToken(issuer *oidcTestServer, group string) string {
	claims := map[string]any{
		"iss": issuer.issuer,
		"aud": []string{exampleAudience},
		"exp": time.Now().Add(5 * time.Minute).Unix(),
		"nbf": time.Now().Add(-1 * time.Minute).Unix(),
		"kubernetes.io": map[string]any{
			// A token binds to one webhook configuration.
			"validatingwebhookconfiguration": map[string]any{
				"name": "my-webhook",
				"uid":  "webhook-uid",
			},
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

// Example_rawHTTPWebhook puts token verification in front of admission logic.
func Example_rawHTTPWebhook() {
	// A test TLS issuer stands in for the configured OIDC issuer.
	issuer, err := startOIDCServer()
	if err != nil {
		panic(err)
	}
	defer issuer.close()

	// Admission logic runs only after token verification succeeds.
	admit := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}

	// Bind the expected audience to the issuer, not to the inbound request.
	h, err := admission.WithTokenVerification(context.Background(), admit,
		admission.WithRemoteIssuer(issuer.issuer, exampleAudience, issuer.client()))
	if err != nil {
		panic(err)
	}

	srv := httptest.NewServer(h)
	defer srv.Close()

	// Send a signed request to exercise the mounted handler.
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

// Example_inClusterWebhook keeps issuer and key discovery automatic while
// binding the expected audience to trusted webhook configuration.
func Example_inClusterWebhook() {
	admit := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}
	// Use the webhook configuration's audience, never a value from the request.
	handler, err := admission.WithTokenVerification(context.Background(), admit,
		admission.WithAudience("https://hook.webhooks.svc:443/validate"))
	if err != nil {
		panic(err)
	}
	mux := http.NewServeMux()
	mux.Handle("/validate", handler)
	_ = mux // Serve using your webhook's existing HTTP server.
}

// Example_controllerRuntimeStyle verifies an already-decoded review. The HTTP
// layer must pass the bearer token separately from the AdmissionReview.
func Example_controllerRuntimeStyle() {
	issuer, err := startOIDCServer()
	if err != nil {
		panic(err)
	}
	defer issuer.close()

	v, err := admission.NewVerifier(context.Background(),
		admission.WithRemoteIssuer(issuer.issuer, exampleAudience, issuer.client()))
	if err != nil {
		panic(err)
	}

	// A framework supplies the decoded review; HTTP middleware supplies the token.
	ctx := context.Background()
	review := exampleReview(exampleAPIGroup)
	token := exampleSignedToken(issuer, exampleAPIGroup)

	// Treat any verification error as a deny; details remain internal.
	if err := v.Verify(ctx, review.Request, token); err != nil {
		fmt.Println("denied") // return a 401 / deny AdmissionResponse here
		return
	}
	fmt.Println("admitted") // proceed to your real admission decision
	// Output: admitted
}

// TestExampleEndToEnd_MinimalWebhook checks that a signed request reaches
// admission logic while an unauthenticated request does not.
func TestExampleEndToEnd_MinimalWebhook(t *testing.T) {
	ts := newOIDCTestServer(t)

	var reached bool
	admit := func(_ context.Context, _ *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		reached = true
		return &admissionv1.AdmissionResponse{Allowed: true}
	}
	h, err := admission.WithTokenVerification(context.Background(), admit,
		admission.WithRemoteIssuer(ts.issuer, exampleAudience, ts.client()))
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

// TestExampleEndToEnd_InClusterExplicitAudience exercises local issuer/JWKS
// discovery while binding the audience independently of an inbound request.
func TestExampleEndToEnd_InClusterExplicitAudience(t *testing.T) {
	ts := newOIDCTestServer(t)
	const svcName, svcNamespace = "webhook", "default"
	audience := verify.AudienceForService(svcName, svcNamespace, 443, "/validate")
	claims := ts.baseClaims()
	claims["aud"] = []string{audience}
	token := ts.sign(t, claims)

	var reached bool
	admit := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		reached = true
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}
	h, err := admission.WithTokenVerification(context.Background(), admit,
		admission.WithInClusterEndpointForTest(ts.issuer, ts.client()),
		admission.WithAudience(audience))
	if err != nil {
		t.Fatalf("construct in-cluster handler: %v", err)
	}
	if err := h.HealthCheck(); err != nil {
		t.Fatalf("explicit audience should be ready at construction: %v", err)
	}
	server := httptest.NewServer(h)
	defer server.Close()

	request := func(bearer string) *http.Response {
		body, err := json.Marshal(exampleReview(exampleAPIGroup))
		if err != nil {
			t.Fatalf("marshal AdmissionReview: %v", err)
		}
		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/validate", bytes.NewReader(body))
		if err != nil {
			t.Fatalf("build admission request: %v", err)
		}
		req.Host = svcName + "." + svcNamespace + ".svc"
		req.Header.Set("Content-Type", "application/json")
		if bearer != "" {
			req.Header.Set("Authorization", "Bearer "+bearer)
		}
		response, err := server.Client().Do(req)
		if err != nil {
			t.Fatalf("send admission request: %v", err)
		}
		return response
	}

	response := request(token)
	defer response.Body.Close()
	var review admissionv1.AdmissionReview
	if err := json.NewDecoder(response.Body).Decode(&review); err != nil {
		t.Fatalf("decode verified AdmissionReview: %v", err)
	}
	if response.StatusCode != http.StatusOK || review.Response == nil || !review.Response.Allowed || !reached {
		t.Fatalf("verified request: HTTP %d response %+v reached %t", response.StatusCode, review.Response, reached)
	}

	reached = false
	missing := request("")
	defer missing.Body.Close()
	if missing.StatusCode != http.StatusUnauthorized || reached {
		t.Fatalf("missing bearer: HTTP %d reached %t, want 401 without admission", missing.StatusCode, reached)
	}
}

// TestExampleEndToEnd_InClusterSpoofedHostCannotChooseAudience guards the
// receiver-owned expected audience against a first request for another Service.
func TestExampleEndToEnd_InClusterSpoofedHostCannotChooseAudience(t *testing.T) {
	ts := newOIDCTestServer(t)
	trustedAudience := verify.AudienceForService("webhook", "default", 443, "/validate")
	otherAudience := verify.AudienceForService("other", "default", 443, "/validate")

	var reached bool
	admit := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		reached = true
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}
	h, err := admission.WithTokenVerification(context.Background(), admit,
		admission.WithInClusterEndpointForTest(ts.issuer, ts.client()),
		admission.WithAudience(trustedAudience))
	if err != nil {
		t.Fatalf("construct in-cluster handler: %v", err)
	}
	server := httptest.NewServer(h)
	defer server.Close()

	verifyRequest := func(audience, host string) *admissionv1.AdmissionResponse {
		claims := ts.baseClaims()
		claims["aud"] = []string{audience}
		body, err := json.Marshal(exampleReview(exampleAPIGroup))
		if err != nil {
			t.Fatalf("marshal AdmissionReview: %v", err)
		}
		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/validate", bytes.NewReader(body))
		if err != nil {
			t.Fatalf("build admission request: %v", err)
		}
		req.Host = host
		req.Header.Set("Authorization", "Bearer "+ts.sign(t, claims))
		response, err := server.Client().Do(req)
		if err != nil {
			t.Fatalf("send admission request: %v", err)
		}
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			t.Fatalf("admission status = %d, want 200", response.StatusCode)
		}
		var review admissionv1.AdmissionReview
		if err := json.NewDecoder(response.Body).Decode(&review); err != nil {
			t.Fatalf("decode AdmissionReview: %v", err)
		}
		return review.Response
	}

	denied := verifyRequest(otherAudience, "other.default.svc")
	if denied == nil || denied.Allowed || reached {
		t.Fatalf("spoofed first request: response %+v reached %t, want denial", denied, reached)
	}
	if err := h.HealthCheck(); err != nil {
		t.Fatalf("trusted audience should remain ready: %v", err)
	}
	allowed := verifyRequest(trustedAudience, "webhook.default.svc")
	if allowed == nil || !allowed.Allowed || !reached {
		t.Fatalf("trusted follow-up: response %+v reached %t, want allow", allowed, reached)
	}
}
