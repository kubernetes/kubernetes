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

package admission

import (
	"testing"

	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
)

func TestPodGroupAdmission(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload: true,
	})

	defaultWorkload := st.MakeWorkload().Name("test-workload").
		PodGroupTemplate(st.MakePodGroupTemplate().Name("worker").MinCount(3).Obj()).
		Obj()

	terminatingWorkload := func() *schedulingapi.Workload {
		w := defaultWorkload.DeepCopy()
		w.Finalizers = []string{"test.k8s.io/block-deletion"}
		return w
	}()

	tests := []struct {
		name           string
		workload       *schedulingapi.Workload
		deleteWorkload bool
		podGroup       func(ns string) *schedulingapi.PodGroup
		expectError    bool
	}{
		{
			name: "PodGroup referencing non-existent Workload is rejected",
			podGroup: func(ns string) *schedulingapi.PodGroup {
				return st.MakePodGroup().Name("pg1").Namespace(ns).TemplateRef("worker", "non-existent-workload").MinCount(3).Obj()
			},
			expectError: true,
		},
		{
			name:     "PodGroup referencing non-existent template is rejected",
			workload: defaultWorkload,
			podGroup: func(ns string) *schedulingapi.PodGroup {
				return st.MakePodGroup().Name("pg1").Namespace(ns).TemplateRef("non-existent-template", "test-workload").MinCount(3).Obj()
			},
			expectError: true,
		},
		{
			name:     "PodGroup referencing valid Workload and template is accepted",
			workload: defaultWorkload,
			podGroup: func(ns string) *schedulingapi.PodGroup {
				return st.MakePodGroup().Name("pg1").Namespace(ns).TemplateRef("worker", "test-workload").MinCount(3).Obj()
			},
			expectError: false,
		},
		{
			name: "PodGroup without templateRef is accepted",
			podGroup: func(ns string) *schedulingapi.PodGroup {
				return &schedulingapi.PodGroup{
					ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: ns},
					Spec: schedulingapi.PodGroupSpec{
						SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
							Gang: &schedulingapi.GangSchedulingPolicy{MinCount: 3},
						},
					},
				}
			},
			expectError: false,
		},
		{
			name:           "PodGroup referencing terminating Workload is rejected",
			workload:       terminatingWorkload,
			deleteWorkload: true,
			podGroup: func(ns string) *schedulingapi.PodGroup {
				return st.MakePodGroup().Name("pg1").Namespace(ns).TemplateRef("worker", "test-workload").MinCount(3).Obj()
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := testutils.InitTestSchedulerWithNS(t, "podgroup-admission",
				scheduler.WithPodMaxBackoffSeconds(0),
				scheduler.WithPodInitialBackoffSeconds(0))
			cs, ns := testCtx.ClientSet, testCtx.NS.Name

			if tt.workload != nil {
				tt.workload.Namespace = ns
				if _, err := cs.SchedulingV1alpha2().Workloads(ns).Create(testCtx.Ctx, tt.workload, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create Workload: %v", err)
				}
				if tt.deleteWorkload {
					if err := cs.SchedulingV1alpha2().Workloads(ns).Delete(testCtx.Ctx, tt.workload.Name, metav1.DeleteOptions{}); err != nil {
						t.Fatalf("Failed to delete Workload: %v", err)
					}
				}
			}

			_, err := cs.SchedulingV1alpha2().PodGroups(ns).Create(testCtx.Ctx, tt.podGroup(ns), metav1.CreateOptions{})
			if tt.expectError {
				if err == nil {
					t.Fatal("Expected PodGroup creation to be rejected, but it succeeded")
				}
				if !apierrors.IsForbidden(err) {
					t.Fatalf("Expected Forbidden error, got: %v", err)
				}
			} else if err != nil {
				t.Fatalf("Expected PodGroup creation to succeed, got: %v", err)
			}
		})
	}
}
