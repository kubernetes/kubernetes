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

package namespace

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/audit"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	api "k8s.io/kubernetes/pkg/apis/core"
	resourceapi "k8s.io/kubernetes/pkg/apis/resource"
)

func TestDRAAdminAccessAuditAnnotationOnCreate(t *testing.T) {
	tests := []struct {
		name      string
		labels    map[string]string
		wantAudit bool
	}{
		{
			name: "label absent",
		},
		{
			name: "label false",
			labels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "false",
			},
		},
		{
			name: "label true",
			labels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "true",
			},
			wantAudit: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := audit.WithAuditContext(genericapirequest.NewDefaultContext())

			namespace := &api.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: tc.labels,
				},
			}

			if errs := Strategy.Validate(ctx, namespace); len(errs) != 0 {
				t.Fatalf("unexpected validation errors: %v", errs)
			}

			value, found := audit.AuditContextFrom(ctx).
				GetEventAnnotation(resourceapi.DRAAdminNamespaceLabelKey)

			if found != tc.wantAudit {
				t.Fatalf("audit annotation found = %v, want %v", found, tc.wantAudit)
			}

			if tc.wantAudit && value != "true" {
				t.Fatalf("audit annotation value = %q, want %q", value, "true")
			}
		})
	}
}

func TestDRAAdminAccessAuditAnnotationOnUpdate(t *testing.T) {
	tests := []struct {
		name      string
		oldLabels map[string]string
		newLabels map[string]string
		wantAudit bool
	}{
		{
			name: "absent to true",
			newLabels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "true",
			},
			wantAudit: true,
		},
		{
			name: "false to true",
			oldLabels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "false",
			},
			newLabels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "true",
			},
			wantAudit: true,
		},
		{
			name: "true to true",
			oldLabels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "true",
			},
			newLabels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "true",
			},
		},
		{
			name: "true to false",
			oldLabels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "true",
			},
			newLabels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "false",
			},
		},
		{
			name: "true to absent",
			oldLabels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "true",
			},
		},
		{
			name: "false to false",
			oldLabels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "false",
			},
			newLabels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "false",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := audit.WithAuditContext(genericapirequest.NewDefaultContext())

			oldNamespace := &api.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "foo",
					ResourceVersion: "1",
					Labels:          tc.oldLabels,
				},
			}

			newNamespace := &api.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "foo",
					ResourceVersion: "2",
					Labels:          tc.newLabels,
				},
			}

			if errs := Strategy.ValidateUpdate(ctx, newNamespace, oldNamespace); len(errs) != 0 {
				t.Fatalf("unexpected validation errors: %v", errs)
			}

			value, found := audit.AuditContextFrom(ctx).
				GetEventAnnotation(resourceapi.DRAAdminNamespaceLabelKey)

			if found != tc.wantAudit {
				t.Fatalf("audit annotation found = %v, want %v", found, tc.wantAudit)
			}

			if tc.wantAudit && value != "true" {
				t.Fatalf("audit annotation value = %q, want %q", value, "true")
			}
		})
	}
}
