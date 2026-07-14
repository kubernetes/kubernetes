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

package apimachinery

import (
	"context"
	"time"

	storagemigrationv1beta1 "k8s.io/api/storagemigration/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econformance "k8s.io/kubernetes/test/e2e/framework/conformance"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("StorageVersionMigration", func() {
	f := framework.NewDefaultFramework("storage-version-migration")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("CRUD Tests", func() {
		/*
		   Release: v1.37
		   Testname: CRUD operations for storageversionmigrations
		   Description: kube-apiserver must support create/update/list/patch/delete operations for storagemigration.k8s.io/v1beta1 StorageVersionMigration.
		*/
		f.It("storagemigration.k8s.io/v1beta1 StorageVersionMigration", f.WithFeatureGate("StorageVersionMigrator"), func(ctx context.Context) {
			e2econformance.TestResource(ctx, f,
				&e2econformance.ResourceTestcase[*storagemigrationv1beta1.StorageVersionMigration]{
					GVR:        storagemigrationv1beta1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					Namespaced: ptr.To(false),
					InitialSpec: &storagemigrationv1beta1.StorageVersionMigration{
						Spec: storagemigrationv1beta1.StorageVersionMigrationSpec{
							Resource: metav1.GroupResource{
								Group:    "",
								Resource: "foo",
							},
						},
					},
					UpdateSpec: func(obj *storagemigrationv1beta1.StorageVersionMigration) *storagemigrationv1beta1.StorageVersionMigration {
						// Spec is immutable, so let's add a label instead.
						if obj.Labels == nil {
							obj.Labels = make(map[string]string)
						}
						obj.Labels["test.storagemigration.example.com"] = "test"
						return obj
					},
					UpdateStatus: func(obj *storagemigrationv1beta1.StorageVersionMigration) *storagemigrationv1beta1.StorageVersionMigration {
						obj.Status.Conditions = append(obj.Status.Conditions, metav1.Condition{
							Type:               "TestCondition",
							Status:             metav1.ConditionTrue,
							Reason:             "TestReason",
							Message:            "TestMessage",
							LastTransitionTime: metav1.NewTime(time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)),
						})
						return obj
					},

					ApplyPatchSpec:            `{"metadata": {"labels": {"test.storagemigration.example.com": "test"}}}`,
					StrategicMergePatchSpec:   `{"metadata": {"labels": {"test.storagemigration.example.com": "test"}}}`,
					JSONMergePatchSpec:        `{"metadata": {"labels": {"test.storagemigration.example.com": "test"}}}`,
					JSONPatchSpec:             `[{"op": "add", "path": "/metadata/labels/test.storagemigration.example.com", "value": "test"}]`,
					ApplyPatchStatus:          `{"status": {"conditions": [{"type": "TestCondition", "status": "True", "reason": "TestReason", "message": "TestMessage", "lastTransitionTime": "2026-01-01T00:00:00Z"}]}}`,
					StrategicMergePatchStatus: `{"status": {"conditions": [{"type": "TestCondition", "status": "True", "reason": "TestReason", "message": "TestMessage", "lastTransitionTime": "2026-01-01T00:00:00Z"}]}}`,
					JSONMergePatchStatus:      `{"status": {"conditions": [{"type": "TestCondition", "status": "True", "reason": "TestReason", "message": "TestMessage", "lastTransitionTime": "2026-01-01T00:00:00Z"}]}}`,
					JSONPatchStatus:           `[{"op": "add", "path": "/status/conditions", "value": [{"type": "TestCondition", "status": "True", "reason": "TestReason", "message": "TestMessage", "lastTransitionTime": "2026-01-01T00:00:00Z"}]}]`,
				})
		})
	})
})
