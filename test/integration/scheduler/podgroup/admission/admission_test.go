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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
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
		name  string
		steps []stepsframework.Step
	}{
		{
			name: "PodGroup referencing non-existent Workload is rejected",
			steps: []stepsframework.Step{
				{
					Name:                         "Creating podgroup referencing non-existent workload expecting failure",
					CreatePodGroupForbiddenError: st.MakePodGroup().Name("pg1").TemplateRef("worker", "non-existent-workload").MinCount(3).Obj(),
				},
			},
		},
		{
			name: "PodGroup referencing non-existent template is rejected",
			steps: []stepsframework.Step{
				{
					Name:            "Creating workload",
					CreateWorkloads: []*schedulingapi.Workload{defaultWorkload},
				},
				{
					Name:                         "Creating podgroup referencing non-existent template expecting failure",
					CreatePodGroupForbiddenError: st.MakePodGroup().Name("pg1").TemplateRef("non-existent-template", "test-workload").MinCount(3).Obj(),
				},
			},
		},
		{
			name: "PodGroup referencing valid Workload and template is accepted",
			steps: []stepsframework.Step{
				{
					Name:            "Creating workload",
					CreateWorkloads: []*schedulingapi.Workload{defaultWorkload},
				},
				{
					Name:           "Creating podgroup referencing valid workload and template",
					CreatePodGroup: st.MakePodGroup().Name("pg1").TemplateRef("worker", "test-workload").MinCount(3).Obj(),
				},
			},
		},
		{
			name: "PodGroup without templateRef is accepted",
			steps: []stepsframework.Step{
				{
					Name: "Creating podgroup without templateRef",
					CreatePodGroup: &schedulingapi.PodGroup{
						ObjectMeta: metav1.ObjectMeta{Name: "pg1"},
						Spec: schedulingapi.PodGroupSpec{
							SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
								Gang: &schedulingapi.GangSchedulingPolicy{MinCount: 3},
							},
						},
					},
				},
			},
		},
		{
			name: "PodGroup referencing terminating Workload is rejected",
			steps: []stepsframework.Step{
				{
					Name:            "Creating workload",
					CreateWorkloads: []*schedulingapi.Workload{terminatingWorkload},
				},
				{
					Name:            "Deleting workload",
					DeleteWorkloads: []*schedulingapi.Workload{terminatingWorkload},
				},
				{
					Name:                         "Creating podgroup referencing terminating workload expecting failure",
					CreatePodGroupForbiddenError: st.MakePodGroup().Name("pg1").TemplateRef("worker", "test-workload").MinCount(3).Obj(),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := testutils.InitTestSchedulerWithNS(t, "podgroup-admission",
				scheduler.WithPodMaxBackoffSeconds(0),
				scheduler.WithPodInitialBackoffSeconds(0))
			ns := testCtx.NS.Name

			if err := stepsframework.RunSteps(testCtx, ns, tt.steps); err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
		})
	}
}
