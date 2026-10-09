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

package admission_test

import (
	"context"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/component-helpers/auth/admission"
)

func TestVerifier_SubresourceKindAndObjectGroups(t *testing.T) {
	ts := newOIDCTestServer(t)
	verifier, err := admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct token verifier: %v", err)
	}
	token := ts.sign(t, ts.baseClaims())
	appsResource := metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
	appsKind := metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}
	scaleKind := metav1.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"}
	for _, tc := range []struct {
		name     string
		kind     metav1.GroupVersionKind
		sub      string
		object   string
		old      string
		wantDeny bool
	}{
		{
			name: "legitimate deployments scale subresource",
			kind: scaleKind, sub: "scale", object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`,
		},
		{
			name: "ordinary resource cannot switch kind groups",
			kind: scaleKind, object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`, wantDeny: true,
		},
		{
			name: "forged scale subresource cannot carry a Role",
			kind: metav1.GroupVersionKind{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "Role"},
			sub:  "scale", object: `{"apiVersion":"rbac.authorization.k8s.io/v1","kind":"Role"}`, wantDeny: true,
		},
		{
			name: "unknown subresource cannot switch to autoscaling group",
			kind: scaleKind, sub: "other", object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`, wantDeny: true,
		},
		{
			name: "subresource object differs from admitted Scale kind",
			kind: scaleKind, sub: "scale", object: `{"apiVersion":"apps/v1","kind":"Deployment"}`, wantDeny: true,
		},
		{
			name: "subresource without a kind cannot claim another object group",
			sub:  "scale", object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`, wantDeny: true,
		},
		{
			name: "old object mismatches scale kind",
			kind: scaleKind, sub: "scale", object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`,
			old: `{"apiVersion":"apps/v1","kind":"Deployment"}`, wantDeny: true,
		},
		{
			name: "ordinary object Kind disagrees with AdmissionRequest Kind",
			kind: appsKind, object: `{"apiVersion":"apps/v1","kind":"Pod"}`, wantDeny: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := &admissionv1.AdmissionRequest{
				Resource: appsResource, Kind: tc.kind, SubResource: tc.sub,
				Object: runtime.RawExtension{Raw: []byte(tc.object)}, OldObject: runtime.RawExtension{Raw: []byte(tc.old)},
			}
			err := verifier.Verify(context.Background(), request, token)
			if tc.wantDeny {
				if err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("verification = %v, want generic denial", err)
				}
			} else if err != nil {
				t.Fatalf("valid subresource request denied: %v", err)
			}
		})
	}
}

func TestVerifier_DecodedObjectsMatchAdmittedKind(t *testing.T) {
	ts := newOIDCTestServer(t)
	verifier, err := admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct token verifier: %v", err)
	}
	token := ts.sign(t, ts.baseClaims())
	object := func(apiVersion, kind string) runtime.Object {
		return &unstructured.Unstructured{Object: map[string]any{"apiVersion": apiVersion, "kind": kind}}
	}
	for _, tc := range []struct {
		name     string
		current  runtime.Object
		old      runtime.Object
		wantDeny bool
	}{
		{name: "matching current and old objects", current: object("apps/v1", "Deployment"), old: object("apps/v1", "Deployment")},
		{name: "wrong current group", current: object("batch/v1", "Job"), wantDeny: true},
		{name: "wrong old group", current: object("apps/v1", "Deployment"), old: object("batch/v1", "Job"), wantDeny: true},
		{name: "missing object type metadata", current: &unstructured.Unstructured{Object: map[string]any{"metadata": map[string]any{"name": "missing"}}}, wantDeny: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := &admissionv1.AdmissionRequest{
				Kind:      metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
				Resource:  metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
				Object:    runtime.RawExtension{Object: tc.current},
				OldObject: runtime.RawExtension{Object: tc.old},
			}
			err := verifier.Verify(context.Background(), req, token)
			if tc.wantDeny {
				if err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("verification = %v, want generic denial", err)
				}
			} else if err != nil {
				t.Fatalf("matching decoded objects denied: %v", err)
			}
		})
	}
}

func TestVerifier_CoreCrossGroupSubresources(t *testing.T) {
	ts := newOIDCTestServer(t)
	verifier, err := admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct token verifier: %v", err)
	}
	claims := ts.baseClaims()
	claims["kubernetes.io"].(map[string]any)["attestations"] = map[string]any{
		admissionReviewAPIGroupsClaimKey: []string{"*"},
	}
	token := ts.sign(t, claims)
	for _, tc := range []struct {
		name, resource, subresource, group, kind, object string
		wantDeny                                         bool
	}{
		{
			name: "Pod eviction", resource: "pods", subresource: "eviction", group: "policy", kind: "Eviction",
			object: `{"apiVersion":"policy/v1","kind":"Eviction"}`,
		},
		{
			name: "ServiceAccount token", resource: "serviceaccounts", subresource: "token", group: "authentication.k8s.io", kind: "TokenRequest",
			object: `{"apiVersion":"authentication.k8s.io/v1","kind":"TokenRequest"}`,
		},
		{
			name: "Pod eviction cannot claim a TokenRequest Kind", resource: "pods", subresource: "eviction", group: "authentication.k8s.io", kind: "TokenRequest",
			object: `{"apiVersion":"authentication.k8s.io/v1","kind":"TokenRequest"}`, wantDeny: true,
		},
		{
			name: "Secret cannot spoof a Pod eviction", resource: "secrets", subresource: "eviction", group: "policy", kind: "Eviction",
			object: `{"apiVersion":"policy/v1","kind":"Eviction"}`, wantDeny: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := &admissionv1.AdmissionRequest{
				Resource:    metav1.GroupVersionResource{Version: "v1", Resource: tc.resource},
				SubResource: tc.subresource,
				Kind:        metav1.GroupVersionKind{Group: tc.group, Version: "v1", Kind: tc.kind},
				Object:      runtime.RawExtension{Raw: []byte(tc.object)},
			}
			err := verifier.Verify(context.Background(), request, token)
			if tc.wantDeny {
				if err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("cross-group spoof: verification = %v, want generic denial", err)
				}
			} else if err != nil {
				t.Fatalf("legitimate cross-group subresource denied: %v", err)
			}
		})
	}
}

func TestVerifier_RejectsMissingObjectTypeMetaForCoreGroup(t *testing.T) {
	ts := newOIDCTestServer(t)
	verifier, err := admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct token verifier: %v", err)
	}
	claims := ts.baseClaims()
	claims["kubernetes.io"].(map[string]any)["attestations"] = map[string]any{
		admissionReviewAPIGroupsClaimKey: []string{"*"},
	}
	token := ts.sign(t, claims)
	for _, tc := range []struct{ name, object string }{
		{name: "JSON null", object: `null`},
		{name: "empty object", object: `{}`},
		{name: "versionless apiVersion", object: `{"apiVersion":"/","kind":"Pod"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := &admissionv1.AdmissionRequest{
				Resource: metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
				Kind:     metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
				Object:   runtime.RawExtension{Raw: []byte(tc.object)},
			}
			if err := verifier.Verify(context.Background(), request, token); err == nil || err.Error() != "webhook token verification failed" {
				t.Fatalf("missing object type metadata: got %v, want generic denial", err)
			}
		})
	}
}
