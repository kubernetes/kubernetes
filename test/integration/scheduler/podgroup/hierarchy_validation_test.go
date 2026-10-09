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

	v1 "k8s.io/api/core/v1"
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
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("wl", "tmpl").ParentCompositePodGroup("cpg3").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:       "Create pod for pg1",
					CreatePods: makeTestPods("pg1", "1"),
				},
				{
					Name: "Verify groups have Invalid condition",
					WaitForGroupsInvalid: &stepsframework.Groups{
						CompositePodGroups: []string{"cpg-root"},
						PodGroups:          []string{"pg1"},
					},
				},
				{
					Name:                       "Verify pod gets scheduling error",
					WaitForPodsSchedulingError: []string{"pg1-pod-0"},
				},
			},
		},
		{
			name: "CompositePodGroup at depth 4 (exceeding max composite depth 3) fails validation in scheduling cycle",
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
					Name:           "Create sibling pod group under root",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("wl", "tmpl").ParentCompositePodGroup("cpg-root").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:       "Create pod for pg1",
					CreatePods: makeTestPods("pg1", "1"),
				},
				{
					Name: "Verify groups have Invalid condition",
					WaitForGroupsInvalid: &stepsframework.Groups{
						CompositePodGroups: []string{"cpg-root"},
						PodGroups:          []string{"pg1"},
					},
				},
				{
					Name:                       "Verify pod gets scheduling error",
					WaitForPodsSchedulingError: []string{"pg1-pod-0"},
				},
			},
		},
		{
			name: "Pod group hierarchy referencing multiple distinct Workloads fails validation in scheduling cycle",
			steps: []stepsframework.Step{
				{
					Name:                    "Create root composite pod group",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("wl-1", "root-t").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:           "Create pod group referencing a different workload",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("wl-2", "tmpl").ParentCompositePodGroup("cpg-root").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:       "Create pod for pg1",
					CreatePods: makeTestPods("pg1", "1"),
				},
				{
					Name: "Verify groups have Invalid condition",
					WaitForGroupsInvalid: &stepsframework.Groups{
						CompositePodGroups: []string{"cpg-root"},
						PodGroups:          []string{"pg1"},
					},
				},
				{
					Name:                       "Verify pod gets scheduling error",
					WaitForPodsSchedulingError: []string{"pg1-pod-0"},
				},
			},
		},
		{
			name: "Gang parent composite pod group with Basic child group fails validation in scheduling cycle",
			steps: []stepsframework.Step{
				{
					Name:                    "Create gang root composite pod group",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("wl", "root-t").MinGroupCount(1).Priority(100).Obj(),
				},
				{
					Name:           "Create basic child pod group",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("wl", "tmpl").ParentCompositePodGroup("cpg-root").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:       "Create pod for pg1",
					CreatePods: makeTestPods("pg1", "1"),
				},
				{
					Name: "Verify groups have Invalid condition",
					WaitForGroupsInvalid: &stepsframework.Groups{
						CompositePodGroups: []string{"cpg-root"},
						PodGroups:          []string{"pg1"},
					},
				},
				{
					Name:                       "Verify pod gets scheduling error",
					WaitForPodsSchedulingError: []string{"pg1-pod-0"},
				},
			},
		},
		{
			name: "Parent composite pod group with All disruption mode and child group with Single disruption mode fails validation in scheduling cycle",
			steps: []stepsframework.Step{
				{
					Name:                    "Create root composite pod group with All disruption mode",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("wl", "root-t").BasicPolicy().DisruptionModeAll().Priority(100).Obj(),
				},
				{
					Name:           "Create child pod group with Single disruption mode",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("wl", "tmpl").ParentCompositePodGroup("cpg-root").BasicPolicy().DisruptionModeSingle().Priority(100).Obj(),
				},
				{
					Name:       "Create pod for pg1",
					CreatePods: makeTestPods("pg1", "1"),
				},
				{
					Name: "Verify groups have Invalid condition",
					WaitForGroupsInvalid: &stepsframework.Groups{
						CompositePodGroups: []string{"cpg-root"},
						PodGroups:          []string{"pg1"},
					},
				},
				{
					Name:                       "Verify pod gets scheduling error",
					WaitForPodsSchedulingError: []string{"pg1-pod-0"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := testutils.InitTestSchedulerWithNS(t, "hierarchy-val",
				scheduler.WithPodMaxBackoffSeconds(1),
				scheduler.WithPodInitialBackoffSeconds(1),
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
