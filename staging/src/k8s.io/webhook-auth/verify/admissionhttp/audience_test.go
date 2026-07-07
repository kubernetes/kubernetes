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

package admissionhttp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/webhook-auth/verify"
)

func TestDeriveExpectedAudience(t *testing.T) {
	tests := []struct {
		name string
		host string
		path string
		want string
	}{
		{name: "host and path", host: "webhook.example.svc:443", path: "/validate", want: "https://webhook.example.svc:443/validate"},
		{name: "empty path defaults to root", host: "webhook.example.svc", path: "", want: "https://webhook.example.svc/"},
		{name: "root path", host: "host", path: "/", want: "https://host/"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "http://"+tc.host+tc.path, nil)
			r.Host = tc.host
			if got := DeriveExpectedAudience(r); got != tc.want {
				t.Errorf("DeriveExpectedAudience() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestAudienceFromServiceURL(t *testing.T) {
	tests := []struct {
		name    string
		url     string
		want    string
		wantErr bool
	}{
		{name: "full URL", url: "https://webhook.example.svc:443/validate", want: "https://webhook.example.svc:443/validate"},
		{name: "no path", url: "https://webhook.example.svc", want: "https://webhook.example.svc/"},
		{name: "no host", url: "/just/a/path", wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := AudienceFromServiceURL(tc.url)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("AudienceFromServiceURL() = %q, want %q", got, tc.want)
			}
		})
	}
}

// audienceTestKeySet treats the raw token as its own verified JSON payload,
// matching the fake used elsewhere in this package's tests.
type audienceTestKeySet struct{}

func (audienceTestKeySet) VerifySignature(_ context.Context, raw string) ([]byte, error) {
	return []byte(raw), nil
}

func TestWithTokenVerificationDerivedAudience(t *testing.T) {
	const issuer = "https://issuer.test"

	t.Run("factory error fails closed with a generic 401", func(t *testing.T) {
		downstreamReached := false
		h := WithTokenVerificationDerivedAudience(
			func([]string) (*verify.Verifier, error) { return nil, errors.New("boom") },
			func(http.ResponseWriter, *http.Request, *admissionv1.AdmissionReview) { downstreamReached = true },
		)
		server := httptest.NewServer(h)
		defer server.Close()

		resp, err := http.Post(server.URL+"/validate", "application/json", strings.NewReader(`{}`))
		if err != nil {
			t.Fatalf("POST: %v", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusUnauthorized)
		}
		if downstreamReached {
			t.Error("downstream must not be reached on factory error")
		}
	})

	t.Run("request-derived audience is used to build the verifier", func(t *testing.T) {
		downstreamReached := false
		factory := func(auds []string) (*verify.Verifier, error) {
			return verify.NewVerifier(audienceTestKeySet{}, issuer, auds)
		}
		h := WithTokenVerificationDerivedAudience(
			factory,
			func(w http.ResponseWriter, _ *http.Request, _ *admissionv1.AdmissionReview) {
				downstreamReached = true
				w.WriteHeader(http.StatusOK)
			},
		)
		server := httptest.NewServer(h)
		defer server.Close()

		host := strings.TrimPrefix(server.URL, "http://")
		expectedAud := "https://" + host + "/validate"

		claims := map[string]any{
			"iss": issuer,
			"aud": []string{expectedAud},
			"exp": time.Now().Add(time.Hour).Unix(),
			"kubernetes.io": map[string]any{
				"validatingWebhookConfiguration": map[string]any{"name": "w", "uid": "u"},
				"attestationClaims":              map[string]any{verify.AllowedAPIGroupClaimKey: []string{"*"}},
			},
		}
		rawToken, err := json.Marshal(claims)
		if err != nil {
			t.Fatalf("marshaling claims: %v", err)
		}

		review := &admissionv1.AdmissionReview{
			Request: &admissionv1.AdmissionRequest{
				Resource: metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
			},
		}
		body, err := json.Marshal(review)
		if err != nil {
			t.Fatalf("marshaling review: %v", err)
		}

		req, err := http.NewRequest(http.MethodPost, server.URL+"/validate", bytes.NewReader(body))
		if err != nil {
			t.Fatalf("building request: %v", err)
		}
		req.Header.Set("Authorization", "Bearer "+string(rawToken))
		resp, err := server.Client().Do(req)
		if err != nil {
			t.Fatalf("POST: %v", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusOK)
		}
		if !downstreamReached {
			t.Error("downstream should be reached for a valid, correctly-audienced token")
		}
	})
}
