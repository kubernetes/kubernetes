/*
Copyright 2026 The Kubernetes Authors.

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

package celtest

import (
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

func TestResolveEquivalentGVK(t *testing.T) {
	tests := []struct {
		name  string
		input *AdmissionInput
		want  schema.GroupVersionKind
	}{
		{
			name:  "from Request.Kind",
			input: &AdmissionInput{Request: &admissionv1.AdmissionRequest{Kind: metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}}},
			want:  schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
		},
		{
			name:  "from Request.RequestKind when Kind is empty",
			input: &AdmissionInput{Request: &admissionv1.AdmissionRequest{RequestKind: &metav1.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}}},
			want:  schema.GroupVersionKind{Version: "v1", Kind: "Pod"},
		},
		{
			name: "from object apiVersion/kind",
			input: &AdmissionInput{
				Object: map[string]interface{}{"apiVersion": "v1", "kind": "ConfigMap"},
			},
			want: schema.GroupVersionKind{Version: "v1", Kind: "ConfigMap"},
		},
		{
			name: "from oldObject when object has no GVK",
			input: &AdmissionInput{
				Object:    map[string]interface{}{"metadata": map[string]interface{}{"name": "x"}},
				OldObject: map[string]interface{}{"apiVersion": "v1", "kind": "Secret"},
			},
			want: schema.GroupVersionKind{Version: "v1", Kind: "Secret"},
		},
		{
			name:  "default when nothing is set",
			input: &AdmissionInput{},
			want:  defaultObjectGVK(),
		},
		{
			name:  "nil input",
			input: nil,
			want:  defaultObjectGVK(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := resolveEquivalentGVK(tt.input)
			if got != tt.want {
				t.Errorf("resolveEquivalentGVK() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestResolveOperation(t *testing.T) {
	tests := []struct {
		name      string
		request   *admissionv1.AdmissionRequest
		object    runtime.Object
		oldObject runtime.Object
		want      admission.Operation
	}{
		{
			name:    "explicit from request",
			request: &admissionv1.AdmissionRequest{Operation: admissionv1.Update},
			want:    admission.Update,
		},
		{
			name:      "infer Create when only object",
			request:   nil,
			object:    &runtime.Unknown{},
			oldObject: nil,
			want:      admission.Create,
		},
		{
			name:      "infer Update when both objects",
			request:   nil,
			object:    &runtime.Unknown{},
			oldObject: &runtime.Unknown{},
			want:      admission.Update,
		},
		{
			name:      "infer Delete when only oldObject",
			request:   nil,
			object:    nil,
			oldObject: &runtime.Unknown{},
			want:      admission.Delete,
		},
		{
			name:      "default Create when no objects",
			request:   nil,
			object:    nil,
			oldObject: nil,
			want:      admission.Create,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := resolveOperation(tt.request, tt.object, tt.oldObject)
			if got != tt.want {
				t.Errorf("resolveOperation() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDeepCopyMap(t *testing.T) {
	original := map[string]interface{}{
		"metadata": map[string]interface{}{
			"name":   "test",
			"labels": map[string]interface{}{"app": "web"},
		},
		"items": []interface{}{"a", "b"},
	}

	copied := deepCopyMap(original)

	// Mutate the copy
	copied["metadata"].(map[string]interface{})["name"] = "mutated"
	copied["metadata"].(map[string]interface{})["labels"].(map[string]interface{})["app"] = "mutated"
	copied["items"].([]interface{})[0] = "mutated"

	// Original must be unchanged
	if original["metadata"].(map[string]interface{})["name"] != "test" {
		t.Error("deepCopyMap: original metadata.name was mutated")
	}
	if original["metadata"].(map[string]interface{})["labels"].(map[string]interface{})["app"] != "web" {
		t.Error("deepCopyMap: original metadata.labels.app was mutated")
	}
	if original["items"].([]interface{})[0] != "a" {
		t.Error("deepCopyMap: original items[0] was mutated")
	}
}

func TestDeepCopyMap_Nil(t *testing.T) {
	if got := deepCopyMap(nil); got != nil {
		t.Errorf("deepCopyMap(nil) = %v, want nil", got)
	}
}

func TestResolveUserInfo(t *testing.T) {
	t.Run("nil request", func(t *testing.T) {
		if got := resolveUserInfo(nil); got != nil {
			t.Errorf("resolveUserInfo(nil) = %v, want nil", got)
		}
	})

	t.Run("empty user info returns nil", func(t *testing.T) {
		req := &admissionv1.AdmissionRequest{}
		if got := resolveUserInfo(req); got != nil {
			t.Errorf("resolveUserInfo(empty) = %v, want nil", got)
		}
	})

	t.Run("populated user info", func(t *testing.T) {
		req := &admissionv1.AdmissionRequest{}
		req.UserInfo.Username = "admin"
		req.UserInfo.UID = "123"
		req.UserInfo.Groups = []string{"system:masters"}

		info := resolveUserInfo(req)
		if info == nil {
			t.Fatal("expected non-nil user info")
		}
		if info.GetName() != "admin" {
			t.Errorf("username = %q, want %q", info.GetName(), "admin")
		}
		if info.GetUID() != "123" {
			t.Errorf("uid = %q, want %q", info.GetUID(), "123")
		}
		if len(info.GetGroups()) != 1 || info.GetGroups()[0] != "system:masters" {
			t.Errorf("groups = %v, want [system:masters]", info.GetGroups())
		}
	})
}

func TestResolveNameAndNamespace(t *testing.T) {
	tests := []struct {
		name          string
		request       *admissionv1.AdmissionRequest
		objectName    string
		oldObjectName string
		wantName      string
		wantNamespace string
	}{
		{
			name:          "from request",
			request:       &admissionv1.AdmissionRequest{Name: "req-name", Namespace: "req-ns"},
			wantName:      "req-name",
			wantNamespace: "req-ns",
		},
		{
			name:          "fallback to object",
			request:       nil,
			objectName:    "obj-name",
			wantName:      "obj-name",
			wantNamespace: "",
		},
		{
			name:          "fallback to oldObject",
			request:       nil,
			oldObjectName: "old-name",
			wantName:      "old-name",
			wantNamespace: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var object, oldObject *unstructured.Unstructured
			if tt.objectName != "" {
				object = &unstructured.Unstructured{Object: map[string]interface{}{
					"metadata": map[string]interface{}{"name": tt.objectName},
				}}
			}
			if tt.oldObjectName != "" {
				oldObject = &unstructured.Unstructured{Object: map[string]interface{}{
					"metadata": map[string]interface{}{"name": tt.oldObjectName},
				}}
			}
			gotName, gotNS := resolveNameAndNamespace(tt.request, object, oldObject)
			if gotName != tt.wantName {
				t.Errorf("name = %q, want %q", gotName, tt.wantName)
			}
			if gotNS != tt.wantNamespace {
				t.Errorf("namespace = %q, want %q", gotNS, tt.wantNamespace)
			}
		})
	}
}

func TestDefaultResourceForGVK(t *testing.T) {
	tests := []struct {
		kind    string
		wantRes string
	}{
		{kind: "Pod", wantRes: "pods"},
		{kind: "Service", wantRes: "services"},
		{kind: "Deployment", wantRes: "deployments"},
		{kind: "Ingress", wantRes: "ingresses"},
		{kind: "NetworkPolicy", wantRes: "networkpolicies"},
		{kind: "DaemonSet", wantRes: "daemonsets"},
		{kind: "ConfigMap", wantRes: "configmaps"},
		{kind: "Endpoints", wantRes: "endpointses"}, // naive but consistent for sibilant ending
		{kind: "", wantRes: "objects"},
	}
	for _, tt := range tests {
		t.Run(tt.kind, func(t *testing.T) {
			gvk := schema.GroupVersionKind{Version: "v1", Kind: tt.kind}
			gvr := defaultResourceForGVK(gvk)
			if gvr.Resource != tt.wantRes {
				t.Errorf("defaultResourceForGVK(%q) = %q, want %q", tt.kind, gvr.Resource, tt.wantRes)
			}
		})
	}
}
