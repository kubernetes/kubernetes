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

package resource

import (
	"context"
	"testing"
	"k8s.io/utils/ptr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/client-go/kubernetes/fake"
	resourceapi "k8s.io/kubernetes/pkg/apis/resource"
)

func TestAuthorizedForAdminAuditAnnotation(t *testing.T) {
	tests := []struct {
		name            string
		adminAccess     *bool
		namespaceLabels map[string]string
		wantErr         bool
		wantAudit       bool
	}{
		{
			name:        "admin access unset",
			adminAccess: nil,
		},
		{
			name:        "admin access false",
			adminAccess: ptr.To(false),
		},
		{
			name:        "admin access true in authorized namespace",
			adminAccess: ptr.To(true),
			namespaceLabels: map[string]string{
				resourceapi.DRAAdminNamespaceLabelKey: "true",
			},
			wantAudit: true,
		},
		{
			name:        "admin access true in unauthorized namespace",
			adminAccess: ptr.To(true),
			wantErr:     true,
			wantAudit:   true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := audit.WithAuditContext(context.Background())

			namespace := &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test",
					Labels: tc.namespaceLabels,
				},
			}

			client := fake.NewSimpleClientset(namespace)

			requests := []resourceapi.DeviceRequest{
				{
					Name: "request",
					Exactly: &resourceapi.ExactDeviceRequest{
						DeviceClassName: "example.com/device",
						AdminAccess:     tc.adminAccess,
					},
				},
			}

			errs := AuthorizedForAdmin(
				ctx,
				requests,
				namespace.Name,
				client.CoreV1().Namespaces(),
			)

			if gotErr := len(errs) > 0; gotErr != tc.wantErr {
				t.Fatalf("AuthorizedForAdmin() errors = %v, wantErr = %v", errs, tc.wantErr)
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
