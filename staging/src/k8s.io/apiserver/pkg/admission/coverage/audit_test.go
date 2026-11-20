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

package coverage

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authentication/user"
)

type annotationRecorder struct {
	admission.Attributes
	annotations map[string]string
}

func (r *annotationRecorder) AddAnnotation(key, value string) error {
	return r.AddAnnotationWithLevel(key, value, auditinternal.LevelMetadata)
}

func (r *annotationRecorder) AddAnnotationWithLevel(key, value string, level auditinternal.Level) error {
	if r.annotations == nil {
		r.annotations = make(map[string]string)
	}
	r.annotations[key] = value
	return r.Attributes.AddAnnotationWithLevel(key, value, level)
}

func newPodUpdate() *annotationRecorder {
	return &annotationRecorder{
		Attributes: admission.NewAttributesRecord(
			&corev1.Pod{},
			&corev1.Pod{},
			schema.GroupVersionKind{Version: "v1", Kind: "Pod"},
			"default",
			"my-pod",
			schema.GroupVersionResource{Version: "v1", Resource: "pods"},
			"dynamic",
			admission.Update,
			nil,
			true,
			&user.DefaultInfo{Name: "alice"},
		),
	}
}

func TestAuditAnnotationKey(t *testing.T) {
	tests := []struct {
		kind HookKind
		want string
	}{
		{WebhookValidating, "coverage.admission.k8s.io/validating-webhook"},
		{WebhookMutating, "coverage.admission.k8s.io/mutating-webhook"},
		{PolicyValidating, "coverage.admission.k8s.io/validating-admission-policy"},
		{PolicyMutating, "coverage.admission.k8s.io/mutating-admission-policy"},
	}
	for _, tc := range tests {
		if got := AuditAnnotationKey(tc.kind); got != tc.want {
			t.Errorf("AuditAnnotationKey(%q) = %q, want %q", tc.kind, got, tc.want)
		}
	}
}

func TestAddAuditAnnotation(t *testing.T) {
	t.Run("empty entries writes nothing", func(t *testing.T) {
		attr := newPodUpdate()
		if err := AddAuditAnnotation(attr, WebhookValidating, nil); err != nil {
			t.Fatalf("AddAuditAnnotation(nil): %v", err)
		}
		if err := AddAuditAnnotation(attr, WebhookValidating, []AuditAnnotationEntry{}); err != nil {
			t.Fatalf("AddAuditAnnotation([]): %v", err)
		}
		if len(attr.annotations) != 0 {
			t.Errorf("expected no annotations, got %v", attr.annotations)
		}
	})

	t.Run("webhook entry omits binding", func(t *testing.T) {
		attr := newPodUpdate()
		entries := []AuditAnnotationEntry{{
			Name:          "check.example.com",
			Configuration: "my-config",
			Source:        "CREATE pods",
			Target:        "UPDATE pods/dynamic",
		}}
		if err := AddAuditAnnotation(attr, WebhookValidating, entries); err != nil {
			t.Fatalf("AddAuditAnnotation: %v", err)
		}
		raw := attr.annotations["coverage.admission.k8s.io/validating-webhook"]
		const wantJSON = `[{"name":"check.example.com","configuration":"my-config","source":"CREATE pods","target":"UPDATE pods/dynamic"}]`
		if raw != wantJSON {
			t.Errorf("annotation = %s, want %s", raw, wantJSON)
		}
	})

	t.Run("policy entries omit configuration and empty source", func(t *testing.T) {
		attr := newPodUpdate()
		entries := []AuditAnnotationEntry{
			{
				Name:    "RequireLabel",
				Binding: "require-label-binding",
				Source:  "UPDATE pods",
				Target:  "UPDATE pods/dynamic",
			},
			{
				Name:    "BrokenCheck",
				Binding: "broken-binding",
				Target:  "UPDATE pods/dynamic",
			},
		}
		if err := AddAuditAnnotation(attr, PolicyValidating, entries); err != nil {
			t.Fatalf("AddAuditAnnotation: %v", err)
		}
		raw := attr.annotations["coverage.admission.k8s.io/validating-admission-policy"]
		const wantJSON = `[{"name":"RequireLabel","binding":"require-label-binding","source":"UPDATE pods","target":"UPDATE pods/dynamic"},{"name":"BrokenCheck","binding":"broken-binding","target":"UPDATE pods/dynamic"}]`
		if raw != wantJSON {
			t.Errorf("annotation = %s, want %s", raw, wantJSON)
		}
	})
}
