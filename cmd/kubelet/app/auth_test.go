/*
Copyright 2025 The Kubernetes Authors.

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

package app

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/apiserver/pkg/authorization/authorizer"
	authenticationv1client "k8s.io/client-go/kubernetes/typed/authentication/v1"
	authorizationv1client "k8s.io/client-go/kubernetes/typed/authorization/v1"
	"k8s.io/client-go/rest"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

func TestAuthzWebhookRequestEncoding(t *testing.T) {
	testCases := []struct {
		name                    string
		ContentType             string
		ExpectContentType       string
		ExpectRequestBodyPrefix []byte
	}{
		{
			name:                    "json",
			ContentType:             "application/json",
			ExpectContentType:       "application/json",
			ExpectRequestBodyPrefix: []byte(`{`),
		},
		{
			name:                    "empty",
			ContentType:             "",
			ExpectContentType:       "application/json",
			ExpectRequestBodyPrefix: []byte(`{`),
		},
		{
			name:                    "protobuf",
			ContentType:             "application/vnd.kubernetes.protobuf",
			ExpectContentType:       "application/vnd.kubernetes.protobuf",
			ExpectRequestBodyPrefix: []byte("\x6b\x38\x73\x00"), // k8s protobuf magic number
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			handlerInvoked := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(handlerInvoked)

				if got := r.Header.Get("Content-Type"); got != tc.ExpectContentType {
					t.Errorf("unexpected Content-Type: got %q, want %q", got, tc.ExpectContentType)
				}

				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Fatalf("failed to read request body: %v", err)
				}
				if !bytes.HasPrefix(body, tc.ExpectRequestBodyPrefix) {
					t.Errorf("request body should have prefix %q, but got %q", tc.ExpectRequestBodyPrefix, body)
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				if _, err := w.Write([]byte(`{"kind":"SubjectAccessReview","apiVersion":"authorization.k8s.io/v1","status":{"allowed":true}}`)); err != nil {
					t.Fatalf("unexpected response write failure: %v", err)
				}
			}))
			defer server.Close()

			cfg := &rest.Config{
				Host: server.URL,
				ContentConfig: rest.ContentConfig{
					ContentType: tc.ContentType,
				},
			}

			authzClient, err := authorizationv1client.NewForConfigAndClient(cfg, server.Client())
			if err != nil {
				t.Fatalf("failed to create authorization client: %v", err)
			}

			authz, err := BuildAuthz(authzClient, kubeletconfig.KubeletAuthorization{Mode: kubeletconfig.KubeletAuthorizationModeWebhook})
			if err != nil {
				t.Fatalf("failed to build authorizer: %v", err)
			}

			if _, _, err := authz.Authorize(context.Background(), &authorizer.AttributesRecord{}); err != nil {
				t.Fatalf("Authorize failed: %v", err)
			}

			select {
			case <-handlerInvoked:
			default:
				t.Fatal("webhook handler not invoked")
			}
		})
	}
}

func TestAuthnWebhookRequestEncoding(t *testing.T) {
	testCases := []struct {
		name                    string
		ContentType             string
		ExpectContentType       string
		ExpectRequestBodyPrefix []byte
	}{
		{
			name:                    "json",
			ContentType:             "application/json",
			ExpectContentType:       "application/json",
			ExpectRequestBodyPrefix: []byte(`{`),
		},
		{
			name:                    "empty",
			ContentType:             "",
			ExpectContentType:       "application/json",
			ExpectRequestBodyPrefix: []byte(`{`),
		},
		{
			name:                    "protobuf",
			ContentType:             "application/vnd.kubernetes.protobuf",
			ExpectContentType:       "application/vnd.kubernetes.protobuf",
			ExpectRequestBodyPrefix: []byte("\x6b\x38\x73\x00"), // k8s protobuf magic number
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			handlerInvoked := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(handlerInvoked)

				if got := r.Header.Get("Content-Type"); got != tc.ExpectContentType {
					t.Errorf("unexpected Content-Type: got %q, want %q", got, tc.ExpectContentType)
				}

				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Fatalf("failed to read request body: %v", err)
				}
				if !bytes.HasPrefix(body, tc.ExpectRequestBodyPrefix) {
					t.Errorf("request body should have prefix %q, but got %q", tc.ExpectRequestBodyPrefix, body)
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				if _, err := w.Write([]byte(`{"kind":"TokenReview","apiVersion":"authentication.k8s.io/v1","status":{"authenticated":true}}`)); err != nil {
					t.Fatalf("unexpected response write failure: %v", err)
				}
			}))
			defer server.Close()

			cfg := &rest.Config{
				Host: server.URL,
				ContentConfig: rest.ContentConfig{
					ContentType: tc.ContentType,
				},
			}

			authnClient, err := authenticationv1client.NewForConfigAndClient(cfg, server.Client())
			if err != nil {
				t.Fatalf("failed to create authentication client: %v", err)
			}

			authn, _, err := BuildAuthn(authnClient, kubeletconfig.KubeletAuthentication{
				Webhook: kubeletconfig.KubeletWebhookAuthentication{
					Enabled: true,
				},
			})
			if err != nil {
				t.Fatalf("failed to build authenticator: %v", err)
			}

			request, err := http.NewRequestWithContext(context.TODO(), http.MethodGet, "/fooz", nil)
			if err != nil {
				t.Fatalf("failed to build test request: %v", err)
			}
			request.Header.Set("Authorization", "Bearer foo")

			if _, _, err := authn.AuthenticateRequest(request); err != nil {
				t.Fatalf("AuthenticateToken failed: %v", err)
			}

			select {
			case <-handlerInvoked:
			default:
				t.Fatal("webhook handler not invoked")
			}
		})
	}
}
