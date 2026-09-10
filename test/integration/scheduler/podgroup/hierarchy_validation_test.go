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

package podgroup

import (
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
	testutils "k8s.io/kubernetes/test/integration/util"
)

func TestHierarchyValidation(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	node := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "10"}).Obj()

	tests := []struct {
		name  string
		steps []stepsframework.Step
	}{
		{
			name: "Incomplete pod group hierarchy with cycle is marked unschedulable and scheduled after breaking cycle",
			steps: []stepsframework.Step{
				{
					Name:                    "Create cyclic composite pod group cpg1",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg1").WorkloadRef("wl", "cpg1-t").ParentCompositePodGroup("cpg2").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:                    "Create cpg2 pointing back to cpg1 forming cycle",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg2").WorkloadRef("wl", "cpg2-t").ParentCompositePodGroup("cpg1").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:           "Create pod group referencing cpg1",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("tmpl", "wl").ParentCompositePodGroup("cpg1").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:       "Create pod for pg1",
					CreatePods: makeTestPods("pg1", "1"),
				},
				{
					Name:                                "Verify pod is placed in incompletePodGroupPods",
					WaitForPodsInIncompletePodGroupPods: []string{"pg1-pod-0"},
				},
				{
					Name: "Verify queue goroutine validates hierarchy and patches pod unschedulable with cycle diagnostic",
					WaitForPodCondition: &stepsframework.PodConditionCheck{
						PodName:         "pg1-pod-0",
						ConditionType:   v1.PodScheduled,
						ConditionStatus: v1.ConditionFalse,
						Reason:          v1.PodReasonUnschedulable,
						MessageContains: "cycle detected in hierarchy",
					},
				},
				{
					Name:                    "Break cycle by deleting cpg2",
					DeleteCompositePodGroup: "cpg2",
				},
				{
					Name:                    "Recreate cpg2 as root without parent",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg2").WorkloadRef("wl", "cpg2-t").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:                 "Verify pod is promoted from incompletePodGroupPods and scheduled",
					WaitForPodsScheduled: []string{"pg1-pod-0"},
				},
			},
		},
		{
			name: "Incomplete pod group hierarchy with missing parent is marked unschedulable and scheduled after parent created",
			steps: []stepsframework.Step{
				{
					Name:                    "Create cpg1 with non-existent parent cpg-root",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg1").WorkloadRef("wl", "cpg1-t").ParentCompositePodGroup("cpg-root").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:           "Create pod group referencing cpg1",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("tmpl", "wl").ParentCompositePodGroup("cpg1").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:       "Create pod for pg1",
					CreatePods: makeTestPods("pg1", "1"),
				},
				{
					Name:                                "Verify pod is placed in incompletePodGroupPods",
					WaitForPodsInIncompletePodGroupPods: []string{"pg1-pod-0"},
				},
				{
					Name: "Verify queue goroutine validates hierarchy and patches pod unschedulable with missing parent diagnostic",
					WaitForPodCondition: &stepsframework.PodConditionCheck{
						PodName:         "pg1-pod-0",
						ConditionType:   v1.PodScheduled,
						ConditionStatus: v1.ConditionFalse,
						Reason:          v1.PodReasonUnschedulable,
						MessageContains: "cpg-root not found in workload forest",
					},
				},
				{
					Name:                    "Create missing root composite pod group",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("wl", "root-t").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:                 "Verify pod is promoted from incompletePodGroupPods and scheduled",
					WaitForPodsScheduled: []string{"pg1-pod-0"},
				},
			},
		},
		{
			name: "Pod group hierarchy exceeding WorkloadMaxTreeDepth fails validation in scheduling cycle",
			steps: []stepsframework.Step{
				{
					Name:                    "Create root composite pod group",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("wl", "root-t").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:                    "Create level 2 composite pod group",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg1").WorkloadRef("wl", "cpg1-t").ParentCompositePodGroup("cpg-root").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:                    "Create level 3 composite pod group",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg2").WorkloadRef("wl", "cpg2-t").ParentCompositePodGroup("cpg1").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:                    "Create level 4 composite pod group",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg3").WorkloadRef("wl", "cpg3-t").ParentCompositePodGroup("cpg2").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:           "Create level 5 pod group exceeding maximum depth of 4",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("tmpl", "wl").ParentCompositePodGroup("cpg3").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:       "Create pod for pg1",
					CreatePods: makeTestPods("pg1", "1"),
				},
				{
					Name: "Verify pod group condition is updated with scheduler error and depth diagnostic",
					WaitForPodGroupCondition: &stepsframework.PodGroupConditionCheck{
						PodGroupName:    "pg1",
						ConditionStatus: metav1.ConditionFalse,
						Reason:          schedulingapi.PodGroupReasonSchedulerError,
						MessageContains: "hierarchy depth 5 exceeds maximum allowed depth 4",
					},
				},
				{
					Name: "Verify pod gets scheduling error with depth diagnostic",
					WaitForPodCondition: &stepsframework.PodConditionCheck{
						PodName:         "pg1-pod-0",
						ConditionType:   v1.PodScheduled,
						ConditionStatus: v1.ConditionFalse,
						Reason:          v1.PodReasonSchedulerError,
						MessageContains: "hierarchy depth 5 exceeds maximum allowed depth 4",
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := testutils.InitTestSchedulerWithNS(t, "hierarchy-val",
				scheduler.WithPodMaxBackoffSeconds(1),
				scheduler.WithPodInitialBackoffSeconds(1),
				scheduler.WithPodMaxInIncompletePodsDuration(time.Second),
				scheduler.WithIncompletePodGroupPodsPeriod(time.Second),
			)

			commonSteps := []stepsframework.Step{
				{
					Name:        "Create Nodes",
					CreateNodes: []*v1.Node{node},
				},
			}

			if err := stepsframework.RunSteps(testCtx, t, testCtx.NS.Name, append(commonSteps, tt.steps...)); err != nil {
				t.Errorf("Test failed: %v", err)
			}
		})
	}
}
