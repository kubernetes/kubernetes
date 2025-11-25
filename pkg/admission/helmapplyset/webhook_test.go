/*
Copyright 2024 The Kubernetes Authors.

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

package helmapplyset

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2/ktesting"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/labeler"
	"k8s.io/kubernetes/pkg/controller/helmapplyset/parent"
)

func TestValidateParentLabels(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	server := NewWebhookServer(nil, logger, false)

	tests := []struct {
		name    string
		secret  *v1.Secret
		request *admissionv1.AdmissionRequest
		wantErr bool
	}{
		{
			name: "valid ApplySet parent Secret",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test",
					Namespace: "default",
					Labels: map[string]string{
						parent.ApplySetParentIDLabel: "applyset-test-id-v1",
					},
					Annotations: map[string]string{
						parent.ApplySetToolingAnnotation: "helm/v3",
						parent.ApplySetGKsAnnotation:     "Deployment.apps,Service",
					},
				},
			},
			request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Create,
			},
			wantErr: false,
		},
		{
			name: "missing labels",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test",
					Namespace: "default",
				},
			},
			request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Create,
			},
			wantErr: true,
		},
		{
			name: "missing ApplySet ID label",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test",
					Namespace: "default",
					Labels:    map[string]string{},
				},
			},
			request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Create,
			},
			wantErr: true,
		},
		{
			name: "invalid ApplySet ID format",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test",
					Namespace: "default",
					Labels: map[string]string{
						parent.ApplySetParentIDLabel: "invalid-id",
					},
				},
			},
			request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Create,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := server.validateParentLabels(tt.secret, tt.request)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateParentLabels() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateParentAnnotations(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	server := NewWebhookServer(nil, logger, false)

	tests := []struct {
		name    string
		secret  *v1.Secret
		wantErr bool
	}{
		{
			name: "valid annotations",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test",
					Namespace: "default",
					Annotations: map[string]string{
						parent.ApplySetToolingAnnotation: "helm/v3",
						parent.ApplySetGKsAnnotation:     "Deployment.apps,Service",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "missing annotations",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test",
					Namespace: "default",
				},
			},
			wantErr: true,
		},
		{
			name: "missing tooling annotation",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test",
					Namespace: "default",
					Annotations: map[string]string{
						parent.ApplySetGKsAnnotation: "Deployment.apps,Service",
					},
				},
			},
			wantErr: true,
		},
		{
			name: "invalid tooling format",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test",
					Namespace: "default",
					Annotations: map[string]string{
						parent.ApplySetToolingAnnotation: "invalid",
					},
				},
			},
			wantErr: true,
		},
		{
			name: "invalid GroupKinds format - empty after parsing",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test",
					Namespace: "default",
					Annotations: map[string]string{
						parent.ApplySetToolingAnnotation: "helm/v3",
						parent.ApplySetGKsAnnotation:     ",,,",
					},
				},
			},
			wantErr: false, // Empty string is valid (no GroupKinds)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := server.validateParentAnnotations(tt.secret)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateParentAnnotations() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCheckUnauthorizedModification(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	server := NewWebhookServer(nil, logger, false)

	validID := parent.ComputeApplySetID("test-release", "default")

	tests := []struct {
		name    string
		request *admissionv1.AdmissionRequest
		wantErr bool
	}{
		{
			name: "authorized user (controller service account)",
			request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Update,
				UserInfo: authenticationv1.UserInfo{
					Username: "system:serviceaccount:kube-system:helm-applyset-controller",
				},
				Object: runtime.RawExtension{
					Raw: marshalSecret(&v1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "applyset-test",
							Namespace: "default",
							Labels: map[string]string{
								parent.ApplySetParentIDLabel: validID,
							},
							Annotations: map[string]string{
								parent.ApplySetToolingAnnotation: "helm/v3",
							},
						},
					}),
				},
				OldObject: runtime.RawExtension{
					Raw: marshalSecret(&v1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "applyset-test",
							Namespace: "default",
							Labels: map[string]string{
								parent.ApplySetParentIDLabel: validID,
							},
							Annotations: map[string]string{
								parent.ApplySetToolingAnnotation: "helm/v3",
							},
						},
					}),
				},
			},
			wantErr: false,
		},
		{
			name: "unauthorized user modifying ApplySet ID",
			request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Update,
				UserInfo: authenticationv1.UserInfo{
					Username: "system:serviceaccount:default:other-serviceaccount",
				},
				Object: runtime.RawExtension{
					Raw: marshalSecret(&v1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "applyset-test",
							Namespace: "default",
							Labels: map[string]string{
								parent.ApplySetParentIDLabel: "applyset-different-id-v1",
							},
							Annotations: map[string]string{
								parent.ApplySetToolingAnnotation: "helm/v3",
							},
						},
					}),
				},
				OldObject: runtime.RawExtension{
					Raw: marshalSecret(&v1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "applyset-test",
							Namespace: "default",
							Labels: map[string]string{
								parent.ApplySetParentIDLabel: validID,
							},
							Annotations: map[string]string{
								parent.ApplySetToolingAnnotation: "helm/v3",
							},
						},
					}),
				},
			},
			wantErr: true,
		},
		{
			name: "unauthorized user modifying tooling annotation",
			request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Update,
				UserInfo: authenticationv1.UserInfo{
					Username: "system:serviceaccount:default:other-serviceaccount",
				},
				Object: runtime.RawExtension{
					Raw: marshalSecret(&v1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "applyset-test",
							Namespace: "default",
							Labels: map[string]string{
								parent.ApplySetParentIDLabel: validID,
							},
							Annotations: map[string]string{
								parent.ApplySetToolingAnnotation: "kubectl/v1.27",
							},
						},
					}),
				},
				OldObject: runtime.RawExtension{
					Raw: marshalSecret(&v1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "applyset-test",
							Namespace: "default",
							Labels: map[string]string{
								parent.ApplySetParentIDLabel: validID,
							},
							Annotations: map[string]string{
								parent.ApplySetToolingAnnotation: "helm/v3",
							},
						},
					}),
				},
			},
			wantErr: true,
		},
		{
			name: "authorized update to GroupKinds",
			request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Update,
				UserInfo: authenticationv1.UserInfo{
					Username: "system:serviceaccount:kube-system:helm-applyset-controller",
				},
				Object: runtime.RawExtension{
					Raw: marshalSecret(&v1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "applyset-test",
							Namespace: "default",
							Labels: map[string]string{
								parent.ApplySetParentIDLabel: validID,
							},
							Annotations: map[string]string{
								parent.ApplySetToolingAnnotation: "helm/v3",
								parent.ApplySetGKsAnnotation:     "Deployment.apps,Service,ConfigMap",
							},
						},
					}),
				},
				OldObject: runtime.RawExtension{
					Raw: marshalSecret(&v1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "applyset-test",
							Namespace: "default",
							Labels: map[string]string{
								parent.ApplySetParentIDLabel: validID,
							},
							Annotations: map[string]string{
								parent.ApplySetToolingAnnotation: "helm/v3",
								parent.ApplySetGKsAnnotation:     "Deployment.apps,Service",
							},
						},
					}),
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := server.checkUnauthorizedModification(tt.request)
			if (err != nil) != tt.wantErr {
				t.Errorf("checkUnauthorizedModification() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateObject(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	server := NewWebhookServer(nil, logger, false)

	validID := parent.ComputeApplySetID("test-release", "default")

	tests := []struct {
		name     string
		request  *admissionv1.AdmissionRequest
		allowed  bool
		hasError bool
	}{
		{
			name: "valid ApplySet parent Secret",
			request: &admissionv1.AdmissionRequest{
				Kind:      metav1.GroupVersionKind{Kind: "Secret", Version: "v1"},
				Operation: admissionv1.Create,
				Name:      "applyset-test",
				Namespace: "default",
				Object: runtime.RawExtension{
					Raw: marshalSecret(&v1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "applyset-test",
							Namespace: "default",
							Labels: map[string]string{
								parent.ApplySetParentIDLabel: validID,
							},
							Annotations: map[string]string{
								parent.ApplySetToolingAnnotation: "helm/v3",
								parent.ApplySetGKsAnnotation:     "Deployment.apps,Service",
							},
						},
					}),
				},
			},
			allowed:  true,
			hasError: false,
		},
		{
			name: "non-Secret resource",
			request: &admissionv1.AdmissionRequest{
				Kind:      metav1.GroupVersionKind{Kind: "Pod", Version: "v1"},
				Operation: admissionv1.Create,
				Object: runtime.RawExtension{
					Raw: []byte(`{"apiVersion":"v1","kind":"Pod","metadata":{"name":"test-pod"}}`),
				},
			},
			allowed:  true,
			hasError: false,
		},
		{
			name: "non-ApplySet Secret",
			request: &admissionv1.AdmissionRequest{
				Kind:      metav1.GroupVersionKind{Kind: "Secret", Version: "v1"},
				Operation: admissionv1.Create,
				Object: runtime.RawExtension{
					Raw: marshalSecret(&v1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "regular-secret",
							Namespace: "default",
						},
					}),
				},
			},
			allowed:  true,
			hasError: false,
		},
		{
			name: "invalid ApplySet parent Secret",
			request: &admissionv1.AdmissionRequest{
				Kind:      metav1.GroupVersionKind{Kind: "Secret", Version: "v1"},
				Operation: admissionv1.Create,
				Name:      "applyset-test",
				Namespace: "default",
				Object: runtime.RawExtension{
					Raw: marshalSecret(&v1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "applyset-test",
							Namespace: "default",
							Labels: map[string]string{
								parent.ApplySetParentIDLabel: "invalid-id",
							},
						},
					}),
				},
			},
			allowed:  false,
			hasError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := server.validateObject(tt.request)
			if response.Allowed != tt.allowed {
				t.Errorf("validateObject() allowed = %v, want %v", response.Allowed, tt.allowed)
			}
			if (response.Result != nil && response.Result.Message != "") != tt.hasError {
				t.Errorf("validateObject() hasError = %v, want %v", response.Result != nil, tt.hasError)
			}
		})
	}
}

func TestHandleValidate(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	server := NewWebhookServer(nil, logger, false)

	validID := parent.ComputeApplySetID("test-release", "default")

	validSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "applyset-test",
			Namespace: "default",
			Labels: map[string]string{
				parent.ApplySetParentIDLabel: validID,
			},
			Annotations: map[string]string{
				parent.ApplySetToolingAnnotation: "helm/v3",
				parent.ApplySetGKsAnnotation:     "Deployment.apps,Service",
			},
		},
	}

	review := &admissionv1.AdmissionReview{
		Request: &admissionv1.AdmissionRequest{
			UID:       "test-uid",
			Kind:      metav1.GroupVersionKind{Kind: "Secret", Version: "v1"},
			Operation: admissionv1.Create,
			Name:      "applyset-test",
			Namespace: "default",
			Object: runtime.RawExtension{
				Raw: marshalSecret(validSecret),
			},
		},
	}

	body, _ := json.Marshal(review)
	req := httptest.NewRequest(http.MethodPost, "/validate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	server.handleValidate(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("handleValidate() status = %v, want %v", w.Code, http.StatusOK)
	}

	var response admissionv1.AdmissionReview
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	if response.Response == nil {
		t.Fatal("Response is nil")
	}

	if !response.Response.Allowed {
		t.Errorf("Expected request to be allowed, got: %v", response.Response.Result)
	}

	if response.Response.UID != "test-uid" {
		t.Errorf("Response UID = %v, want test-uid", response.Response.UID)
	}
}

func TestHandleMutate(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	server := NewWebhookServer(nil, logger, false)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
		},
	}

	review := &admissionv1.AdmissionReview{
		Request: &admissionv1.AdmissionRequest{
			UID:       "test-uid",
			Kind:      metav1.GroupVersionKind{Kind: "Pod", Version: "v1"},
			Operation: admissionv1.Create,
			Name:      "test-pod",
			Namespace: "default",
			Object: runtime.RawExtension{
				Raw: marshalPod(pod),
			},
		},
	}

	body, _ := json.Marshal(review)
	req := httptest.NewRequest(http.MethodPost, "/mutate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	server.handleMutate(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("handleMutate() status = %v, want %v", w.Code, http.StatusOK)
	}

	var response admissionv1.AdmissionReview
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	if response.Response == nil {
		t.Fatal("Response is nil")
	}

	if !response.Response.Allowed {
		t.Errorf("Expected request to be allowed, got: %v", response.Response.Result)
	}
}

func TestIsApplySetParentSecret(t *testing.T) {
	validID := parent.ComputeApplySetID("test-release", "default")

	tests := []struct {
		name   string
		secret *v1.Secret
		want   bool
	}{
		{
			name: "valid ApplySet parent Secret",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						parent.ApplySetParentIDLabel: validID,
					},
				},
			},
			want: true,
		},
		{
			name: "non-ApplySet Secret",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"other-label": "value",
					},
				},
			},
			want: false,
		},
		{
			name: "Secret without labels",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isApplySetParentSecret(tt.secret); got != tt.want {
				t.Errorf("isApplySetParentSecret() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsHelmManagedResource(t *testing.T) {
	tests := []struct {
		name string
		obj  metav1.Object
		want bool
	}{
		{
			name: "Helm-managed resource",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						labeler.HelmManagedByLabel: "Helm",
						labeler.HelmInstanceLabel:  "test-release",
					},
				},
			},
			want: true,
		},
		{
			name: "non-Helm resource",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"other-label": "value",
					},
				},
			},
			want: false,
		},
		{
			name: "resource without labels",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isHelmManagedResource(tt.obj); got != tt.want {
				t.Errorf("isHelmManagedResource() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHandleHealth(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	server := NewWebhookServer(nil, logger, false)

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()

	server.handleHealth(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("handleHealth() status = %v, want %v", w.Code, http.StatusOK)
	}

	if w.Body.String() != "ok" {
		t.Errorf("handleHealth() body = %v, want ok", w.Body.String())
	}
}

func TestHandleReady(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	// Test with TLS config
	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{
			{Certificate: [][]byte{[]byte("test-cert")}},
		},
	}
	server := NewWebhookServer(tlsConfig, logger, false)

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	w := httptest.NewRecorder()

	server.handleReady(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("handleReady() status = %v, want %v", w.Code, http.StatusOK)
	}

	// Test without TLS config
	serverNoTLS := NewWebhookServer(nil, logger, false)
	w2 := httptest.NewRecorder()
	serverNoTLS.handleReady(w2, req)

	if w2.Code != http.StatusServiceUnavailable {
		t.Errorf("handleReady() without TLS status = %v, want %v", w2.Code, http.StatusServiceUnavailable)
	}
}

func TestFailOpen(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	server := NewWebhookServer(nil, logger, true) // failOpen = true

	// Send invalid request
	req := httptest.NewRequest(http.MethodPost, "/validate", bytes.NewReader([]byte("invalid json")))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	server.handleValidate(w, req)

	// Should return 200 with allow response due to fail-open
	if w.Code != http.StatusOK {
		t.Errorf("handleValidate() with fail-open status = %v, want %v", w.Code, http.StatusOK)
	}

	var response admissionv1.AdmissionReview
	if err := json.Unmarshal(w.Body.Bytes(), &response); err == nil {
		if response.Response != nil && !response.Response.Allowed {
			t.Errorf("Expected request to be allowed with fail-open, got: %v", response.Response.Result)
		}
	}
}

// Helper functions

func marshalSecret(secret *v1.Secret) []byte {
	bytes, _ := json.Marshal(secret)
	return bytes
}

func marshalPod(pod *v1.Pod) []byte {
	bytes, _ := json.Marshal(pod)
	return bytes
}
