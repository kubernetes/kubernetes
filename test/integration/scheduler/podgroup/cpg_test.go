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
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// _ to avoid unused import
var _ = time.Second

func makeTestPods(pgName string, reqCPUs ...string) []*v1.Pod {
	var pods []*v1.Pod
	for i, cpu := range reqCPUs {
		pod := st.MakePod().Name(fmt.Sprintf("%s-pod-%d", pgName, i)).
			PodGroupName(pgName).Priority(100).Obj()

		pod.Spec.Containers = []v1.Container{{
			Name:  "container",
			Image: "image",
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse(cpu),
				},
			},
		}}

		pods = append(pods, pod)
	}
	return pods
}

func concatPods(podSlices ...[]*v1.Pod) []*v1.Pod {
	var pods []*v1.Pod
	for _, slice := range podSlices {
		pods = append(pods, slice...)
	}
	return pods
}

func podNames(pods []*v1.Pod) []string {
	var names []string
	for _, p := range pods {
		names = append(names, p.Name)
	}
	return names
}

func TestCPGScheduling(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	tests := []struct {
		name  string
		steps []stepsframework.Step
	}{
		{
			name: "TestCPGHierarchicalScheduling",
			// TestCPGHierarchicalScheduling verifies that readiness and feasibility propagate correctly
			// across a 3-level CPG hierarchy with mixed Gang and Basic policies.
			//
			// Tree structure:
			//
			//	                    cpg-root (Gang, Min: 2)
			//	            /                 |                 \
			//	   cpg-sub1 (Gang, Min:2)  cpg-sub2 (Basic)    cpg-sub3 (Gang, Min:2)
			//	   /    |    \              /       \             /         \
			//	 pg1   pg2   pg3          pg4       pg5         pg6        pg7
			//	(S)   (S)   (F)          (S)       (S)         (F)        (F)
			//
			// [Gang] [Gang] [Gang]     [Basic]   [Gang]      [Gang]     [Gang]
			//
			// (S) = Success (pods schedule according to policy)
			// (F) = Fail (pods cannot schedule)
			// Note: pg6 fails because its parent cpg-sub3 fails (needs 2 children but pg7 fails).
			steps: []stepsframework.Step{
				{
					Name:        "Create Node",
					CreateNodes: []*v1.Node{st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()},
				},
				{
					Name: "Create Workload",
					CreateWorkloads: []*schedulingapi.Workload{
						st.MakeWorkload().Name("workload-cpg").
							Children(
								st.MakeCompositePodGroupTemplate().Name("root-t").MinGroupCount(2).Priority(100).Children(
									st.MakeCompositePodGroupTemplate().Name("sub1-3-t").MinGroupCount(2).Priority(100).Children(
										st.MakePodGroupTemplate().Name("gang-t1").MinCount(3).Priority(100),
									),
									st.MakeCompositePodGroupTemplate().Name("sub2-t").BasicPolicy().Priority(100).Children(
										st.MakePodGroupTemplate().Name("basic-t").BasicPolicy().Priority(100),
										st.MakePodGroupTemplate().Name("gang-t2").MinCount(3).Priority(100),
									),
								),
							).Obj(),
					},
				},
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("workload-cpg", "root-t").MinGroupCount(2).Priority(100).Obj(),
				},
				{
					Name:                    "Create cpg-sub1",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-sub1").WorkloadRef("workload-cpg", "sub1-3-t").MinGroupCount(2).ParentCompositePodGroup("cpg-root").Priority(100).Obj(),
				},
				{
					Name:                    "Create cpg-sub2",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-sub2").WorkloadRef("workload-cpg", "sub2-t").BasicPolicy().ParentCompositePodGroup("cpg-root").Priority(100).Obj(),
				},
				{
					Name:                    "Create cpg-sub3",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-sub3").WorkloadRef("workload-cpg", "sub1-3-t").MinGroupCount(2).ParentCompositePodGroup("cpg-root").Priority(100).Obj(),
				},
				{
					Name:           "Create pg1",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("workload-cpg", "gang-t1").ParentCompositePodGroup("cpg-sub1").Priority(100).MinCount(3).Obj(),
				},
				{
					Name:           "Create pg2",
					CreatePodGroup: st.MakePodGroup().Name("pg2").WorkloadRef("workload-cpg", "gang-t1").ParentCompositePodGroup("cpg-sub1").Priority(100).MinCount(3).Obj(),
				},
				{
					Name:           "Create pg3",
					CreatePodGroup: st.MakePodGroup().Name("pg3").WorkloadRef("workload-cpg", "gang-t1").ParentCompositePodGroup("cpg-sub1").Priority(100).MinCount(3).Obj(),
				},
				{
					Name:           "Create pg4",
					CreatePodGroup: st.MakePodGroup().Name("pg4").WorkloadRef("workload-cpg", "basic-t").ParentCompositePodGroup("cpg-sub2").Priority(100).BasicPolicy().Obj(),
				},
				{
					Name:           "Create pg5",
					CreatePodGroup: st.MakePodGroup().Name("pg5").WorkloadRef("workload-cpg", "gang-t2").ParentCompositePodGroup("cpg-sub2").Priority(100).MinCount(3).Obj(),
				},
				{
					Name:           "Create pg6",
					CreatePodGroup: st.MakePodGroup().Name("pg6").WorkloadRef("workload-cpg", "gang-t1").ParentCompositePodGroup("cpg-sub3").Priority(100).MinCount(3).Obj(),
				},
				{
					Name:           "Create pg7",
					CreatePodGroup: st.MakePodGroup().Name("pg7").WorkloadRef("workload-cpg", "gang-t1").ParentCompositePodGroup("cpg-sub3").Priority(100).MinCount(3).Obj(),
				},
				{
					Name: "Create Pods",
					CreatePods: concatPods(
						makeTestPods("pg1", "1", "1", "1"),
						makeTestPods("pg2", "1", "1", "1"),
						makeTestPods("pg3", "100", "100", "100"), // (F) unschedulable
						makeTestPods("pg4", "1", "1", "1"),
						makeTestPods("pg5", "1", "1", "1"),
						makeTestPods("pg6", "1", "1", "1"),
						makeTestPods("pg7", "100", "100", "100"), // (F) unschedulable
					),
				},
				{
					Name: "Wait for successful pods",
					WaitForPodsScheduled: podNames(concatPods(
						makeTestPods("pg1", "1", "1", "1"),
						makeTestPods("pg2", "1", "1", "1"),
						makeTestPods("pg4", "1", "1", "1"),
						makeTestPods("pg5", "1", "1", "1"),
					)),
				},
				{
					Name: "Verify failing pods remain unschedulable",
					WaitForPodsUnschedulable: podNames(concatPods(
						makeTestPods("pg3", "1", "1", "1"),
						makeTestPods("pg6", "1", "1", "1"),
						makeTestPods("pg7", "1", "1", "1"),
					)),
				},
			},
		},
		{
			name: "TestCPGMinGroupCount",
			// TestCPGMinGroupCount verifies that a Gang CompositePodGroup schedules its children
			// if at least MinGroupCount of them are fully schedulable.
			//
			// Tree structure:
			//
			//	   cpg-root (Gang, MinGroup: 2)
			//	  /             |              \
			//	pg1            pg2            pg3
			//	(S)            (S)            (F)
			//
			// (S) = Success
			// (F) = Fail
			steps: []stepsframework.Step{
				{
					Name:        "Create Node",
					CreateNodes: []*v1.Node{st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()},
				},
				{
					Name: "Create Workload",
					CreateWorkloads: []*schedulingapi.Workload{
						st.MakeWorkload().Name("workload-cpg-min").
							Children(
								st.MakeCompositePodGroupTemplate().Name("root-t").MinGroupCount(2).Priority(100).Children(
									st.MakePodGroupTemplate().Name("gang-t").MinCount(3).Priority(100),
								),
							).Obj(),
					},
				},
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("workload-cpg-min", "root-t").MinGroupCount(2).Priority(100).Obj(),
				},
				{
					Name:           "Create pg1",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("workload-cpg-min", "gang-t").ParentCompositePodGroup("cpg-root").Priority(100).MinCount(3).Obj(),
				},
				{
					Name:           "Create pg2",
					CreatePodGroup: st.MakePodGroup().Name("pg2").WorkloadRef("workload-cpg-min", "gang-t").ParentCompositePodGroup("cpg-root").Priority(100).MinCount(3).Obj(),
				},
				{
					Name:           "Create pg3",
					CreatePodGroup: st.MakePodGroup().Name("pg3").WorkloadRef("workload-cpg-min", "gang-t").ParentCompositePodGroup("cpg-root").Priority(100).MinCount(3).Obj(),
				},
				{
					Name: "Create Pods",
					CreatePods: concatPods(
						makeTestPods("pg1", "1", "1", "1"),
						makeTestPods("pg2", "1", "1", "1"),
						makeTestPods("pg3", "100", "1", "1"),
					),
				},
				{
					Name: "Wait for successful pods",
					WaitForPodsScheduled: podNames(concatPods(
						makeTestPods("pg1", "1", "1", "1"),
						makeTestPods("pg2", "1", "1", "1"),
					)),
				},
				{
					Name: "Verify failing pods remain unschedulable",
					WaitForPodsUnschedulable: podNames(concatPods(
						makeTestPods("pg3", "1", "1", "1"),
					)),
				},
			},
		},
		{
			name: "TestCPGBasicWithGangChildren",
			// TestCPGBasicWithGangChildren verifies that a Basic CompositePodGroup allows its ready
			// child Gang groups to schedule independently of each other.
			//
			// Tree structure:
			//
			//	   cpg-root (Basic)
			//	  /            \
			//	pg1            pg2
			//	(S)            (F)
			//
			// (S) = Success
			// (F) = Fail
			steps: []stepsframework.Step{
				{
					Name:        "Create Node",
					CreateNodes: []*v1.Node{st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()},
				},
				{
					Name: "Create Workload",
					CreateWorkloads: []*schedulingapi.Workload{
						st.MakeWorkload().Name("workload-cpg-basic").
							Children(
								st.MakeCompositePodGroupTemplate().Name("root-t").BasicPolicy().Priority(100).Children(
									st.MakePodGroupTemplate().Name("gang-t").MinCount(3).Priority(100),
								),
							).Obj(),
					},
				},
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("workload-cpg-basic", "root-t").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:           "Create pg1",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("workload-cpg-basic", "gang-t").ParentCompositePodGroup("cpg-root").Priority(100).MinCount(3).Obj(),
				},
				{
					Name:           "Create pg2",
					CreatePodGroup: st.MakePodGroup().Name("pg2").WorkloadRef("workload-cpg-basic", "gang-t").ParentCompositePodGroup("cpg-root").Priority(100).MinCount(3).Obj(),
				},
				{
					Name: "Create Pods",
					CreatePods: concatPods(
						makeTestPods("pg1", "1", "1", "1"),
						makeTestPods("pg2", "100", "1", "1"),
					),
				},
				{
					Name: "Wait for successful pods",
					WaitForPodsScheduled: podNames(concatPods(
						makeTestPods("pg1", "1", "1", "1"),
					)),
				},
				{
					Name: "Verify failing pods remain unschedulable",
					WaitForPodsUnschedulable: podNames(concatPods(
						makeTestPods("pg2", "1", "1", "1"),
					)),
				},
			},
		},
		{
			name: "TestCPGBasicWithBasicChildren",
			// TestCPGBasicWithBasicChildren verifies that a Basic CompositePodGroup with Basic child
			// pod groups schedules all pods successfully since there are no gang constraints.
			//
			// Tree structure:
			//
			//	   cpg-root (Basic)
			//	  /            \
			//	pg1            pg2
			//	(S)            (S)
			//
			// (S) = Success
			steps: []stepsframework.Step{
				{
					Name:        "Create Node",
					CreateNodes: []*v1.Node{st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()},
				},
				{
					Name: "Create Workload",
					CreateWorkloads: []*schedulingapi.Workload{
						st.MakeWorkload().Name("workload-cpg-basic-basic").
							Children(
								st.MakeCompositePodGroupTemplate().Name("root-t").BasicPolicy().Priority(100).Children(
									st.MakePodGroupTemplate().Name("basic-t").BasicPolicy().Priority(100),
								),
							).Obj(),
					},
				},
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("workload-cpg-basic-basic", "root-t").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:           "Create pg1",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("workload-cpg-basic-basic", "basic-t").ParentCompositePodGroup("cpg-root").Priority(100).BasicPolicy().Obj(),
				},
				{
					Name:           "Create pg2",
					CreatePodGroup: st.MakePodGroup().Name("pg2").WorkloadRef("workload-cpg-basic-basic", "basic-t").ParentCompositePodGroup("cpg-root").Priority(100).BasicPolicy().Obj(),
				},
				{
					Name: "Create Pods",
					CreatePods: concatPods(
						makeTestPods("pg1", "1", "1"),
						makeTestPods("pg2", "1", "1"),
					),
				},
				{
					Name: "Wait for successful pods",
					WaitForPodsScheduled: podNames(concatPods(
						makeTestPods("pg1", "1", "1"),
						makeTestPods("pg2", "1", "1"),
					)),
				},
			},
		},
		{
			name: "TestCPGGangWithBasicChildren",
			// TestCPGGangWithBasicChildren verifies that a Gang CompositePodGroup with Basic children
			// enforces the gang scheduling policy (minGroupCount) at the root level.
			//
			// Tree structure:
			//
			//	   cpg-root (Gang, Min: 2)
			//	  /          |          \
			//	pg1         pg2         pg3
			//	(S)         (S)         (F)
			//
			// (S) = Success
			// (F) = Fail
			steps: []stepsframework.Step{
				{
					Name:        "Create Node",
					CreateNodes: []*v1.Node{st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "8"}).Obj()},
				},
				{
					Name: "Create Workload",
					CreateWorkloads: []*schedulingapi.Workload{
						st.MakeWorkload().Name("workload-cpg-gang-basic").
							Children(
								st.MakeCompositePodGroupTemplate().Name("root-t").MinGroupCount(2).Priority(100).Children(
									st.MakePodGroupTemplate().Name("basic-t").BasicPolicy().Priority(100),
								),
							).Obj(),
					},
				},
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("workload-cpg-gang-basic", "root-t").MinGroupCount(2).Priority(100).Obj(),
				},
				{
					Name:           "Create pg1",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("workload-cpg-gang-basic", "basic-t").ParentCompositePodGroup("cpg-root").Priority(100).BasicPolicy().Obj(),
				},
				{
					Name:           "Create pg2",
					CreatePodGroup: st.MakePodGroup().Name("pg2").WorkloadRef("workload-cpg-gang-basic", "basic-t").ParentCompositePodGroup("cpg-root").Priority(100).BasicPolicy().Obj(),
				},
				{
					Name:           "Create pg3",
					CreatePodGroup: st.MakePodGroup().Name("pg3").WorkloadRef("workload-cpg-gang-basic", "basic-t").ParentCompositePodGroup("cpg-root").Priority(100).BasicPolicy().Obj(),
				},
				{
					Name: "Create Pods",
					CreatePods: concatPods(
						makeTestPods("pg1", "1", "1", "1"),
						makeTestPods("pg2", "1", "1", "1"),
						makeTestPods("pg3", "10", "10", "10"),
					),
				},
				{
					Name: "Wait for successful pods",
					WaitForPodsScheduled: podNames(concatPods(
						makeTestPods("pg1", "1", "1", "1"),
						makeTestPods("pg2", "1", "1", "1"),
					)),
				},
				{
					Name: "Verify failing pods remain unschedulable",
					WaitForPodsUnschedulable: podNames(concatPods(
						makeTestPods("pg3", "1", "1", "1"),
					)),
				},
			},
		},
		{
			name: "TestCPGBasicWithUnschedulableBasicChildren",
			// TestCPGBasicWithUnschedulableBasicChildren verifies that when a Basic CompositePodGroup has
			// unschedulable Basic child pod groups, the pods remain unscheduled.
			//
			// Tree structure:
			//
			//	   cpg-root (Basic)
			//	  /            \
			//	pg1            pg2
			//	(F)            (F)
			//
			// (F) = Fail
			steps: []stepsframework.Step{
				{
					Name:        "Create Node",
					CreateNodes: []*v1.Node{st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()},
				},
				{
					Name: "Create Workload",
					CreateWorkloads: []*schedulingapi.Workload{
						st.MakeWorkload().Name("workload-cpg-basic-unschedulable").
							Children(
								st.MakeCompositePodGroupTemplate().Name("root-t").BasicPolicy().Priority(100).Children(
									st.MakePodGroupTemplate().Name("basic-t").BasicPolicy().Priority(100),
								),
							).Obj(),
					},
				},
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("workload-cpg-basic-unschedulable", "root-t").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:           "Create pg1",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("workload-cpg-basic-unschedulable", "basic-t").ParentCompositePodGroup("cpg-root").Priority(100).BasicPolicy().Obj(),
				},
				{
					Name:           "Create pg2",
					CreatePodGroup: st.MakePodGroup().Name("pg2").WorkloadRef("workload-cpg-basic-unschedulable", "basic-t").ParentCompositePodGroup("cpg-root").Priority(100).BasicPolicy().Obj(),
				},
				{
					Name: "Create Pods",
					CreatePods: concatPods(
						makeTestPods("pg1", "50", "50"),
						makeTestPods("pg2", "50", "50"),
					),
				},
				{
					Name: "Wait for pods to be unschedulable",
					WaitForPodsUnschedulable: podNames(concatPods(
						makeTestPods("pg1", "1", "1"),
						makeTestPods("pg2", "1", "1"),
					)),
				},
			},
		},
		{
			name: "TestCPGDynamicChildAddition",
			// TestCPGDynamicChildAddition verifies that after a CompositePodGroup (CPG) hierarchy
			// successfully schedules its initial pods, dynamically adding new child groups and sub-CPGs
			// behaves correctly under scheduling policies (e.g., Gang scheduling with minGroupCount).
			//
			// Tree structure progression:
			//
			// Stage 1:
			//
			//	cpg-root (Basic)
			//	   |
			//	  pg1 (Basic)
			//	  (Scheduled)
			//
			// Stage 2 (Dynamically added pg2):
			//
			//	 cpg-root (Basic)
			//	  /      \
			//	pg1      pg2 (Basic)
			//	(S)      (Scheduled)
			//
			// Stage 3 (Dynamically added cpg-sub tree under cpg-root):
			//
			//	 cpg-root (Basic)
			//	  /   |  \
			//	pg1  pg2  cpg-sub (Gang minGroupCount = 2)
			//	(S)  (S)     |
			//	           sub-pg1 (Basic)
			//	           (Pending - quorum of 2 not met)
			//
			// Stage 4 (Dynamically added sub-pg2 under cpg-sub):
			//
			//	 cpg-root (Basic)
			//	  /   |  \
			//	pg1  pg2  cpg-sub (Gang minGroupCount = 2)
			//	(S)  (S)   /     \
			//	        sub-pg1  sub-pg2 (Basic)
			//	        (S)      (S)
			//
			// (S) = Successfully scheduled
			steps: []stepsframework.Step{
				{
					Name:        "Create Node",
					CreateNodes: []*v1.Node{st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()},
				},
				{
					Name: "Create Workload",
					CreateWorkloads: []*schedulingapi.Workload{
						st.MakeWorkload().Name("workload-cpg-dynamic").
							Children(
								st.MakeCompositePodGroupTemplate().Name("root-t").BasicPolicy().Priority(100).Children(
									st.MakePodGroupTemplate().Name("basic-t").BasicPolicy().Priority(100),
									st.MakeCompositePodGroupTemplate().Name("sub-cpg-t").MinGroupCount(2).Priority(100).Children(
										st.MakePodGroupTemplate().Name("sub-pg-t").BasicPolicy().Priority(100),
									),
								),
							).Obj(),
					},
				},
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("workload-cpg-dynamic", "root-t").BasicPolicy().Priority(100).Obj(),
				},
				{
					Name:           "Create pg1",
					CreatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("workload-cpg-dynamic", "basic-t").ParentCompositePodGroup("cpg-root").Priority(100).BasicPolicy().Obj(),
				},
				{
					Name:       "Create pg1 pods",
					CreatePods: makeTestPods("pg1", "1", "1"),
				},
				{
					Name:                 "Wait for pg1 pods to be scheduled",
					WaitForPodsScheduled: podNames(makeTestPods("pg1", "1", "1")),
				},
				{
					Name:           "Create pg2",
					CreatePodGroup: st.MakePodGroup().Name("pg2").WorkloadRef("workload-cpg-dynamic", "basic-t").ParentCompositePodGroup("cpg-root").Priority(100).BasicPolicy().Obj(),
				},
				{
					Name:       "Create pg2 pods",
					CreatePods: makeTestPods("pg2", "1", "1"),
				},
				{
					Name:                 "Wait for pg2 pods to be scheduled",
					WaitForPodsScheduled: podNames(makeTestPods("pg2", "1", "1")),
				},
				{
					Name:                    "Create cpg-sub",
					CreateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-sub").WorkloadRef("workload-cpg-dynamic", "sub-cpg-t").MinGroupCount(2).ParentCompositePodGroup("cpg-root").Priority(100).Obj(),
				},
				{
					Name:           "Create sub-pg1",
					CreatePodGroup: st.MakePodGroup().Name("sub-pg1").WorkloadRef("workload-cpg-dynamic", "sub-pg-t").ParentCompositePodGroup("cpg-sub").Priority(100).BasicPolicy().Obj(),
				},
				{
					Name:       "Create sub-pg1 pods",
					CreatePods: makeTestPods("sub-pg1", "1", "1"),
				},
				{
					Name:                     "Wait for sub-pg1 pods to be unschedulable (quorum not met)",
					WaitForPodsUnschedulable: podNames(makeTestPods("sub-pg1", "1", "1")),
				},
				{
					Name:           "Create sub-pg2",
					CreatePodGroup: st.MakePodGroup().Name("sub-pg2").WorkloadRef("workload-cpg-dynamic", "sub-pg-t").ParentCompositePodGroup("cpg-sub").Priority(100).BasicPolicy().Obj(),
				},
				{
					Name:       "Create sub-pg2 pods",
					CreatePods: makeTestPods("sub-pg2", "1", "1"),
				},
				{
					Name: "Wait for both sub-pg1 and sub-pg2 pods to be scheduled",
					WaitForPodsScheduled: podNames(concatPods(
						makeTestPods("sub-pg1", "1", "1"),
						makeTestPods("sub-pg2", "1", "1"),
					)),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-sched",
				scheduler.WithPodMaxBackoffSeconds(1),
				scheduler.WithPodInitialBackoffSeconds(1))

			if err := stepsframework.RunSteps(testCtx, t, testCtx.NS.Name, tt.steps); err != nil {
				t.Errorf("Test failed: %v", err)
			}
		})
	}
}

func TestPodGroupDependentRequiredValidation(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	testCtx := testutils.InitTestSchedulerWithNS(t, "pg-validation",
		scheduler.WithPodMaxBackoffSeconds(1),
		scheduler.WithPodInitialBackoffSeconds(1))
	cs, ns := testCtx.ClientSet, testCtx.NS.Name

	// Case 1: ParentCompositePodGroupName set and WorkloadRef set -> Should PASS
	pgPass := &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg-pass", Namespace: ns},
		Spec: schedulingapi.PodGroupSpec{
			SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
				Basic: &schedulingapi.BasicSchedulingPolicy{},
			},
			ParentCompositePodGroupName: new("some-cpg"),
			WorkloadRef: &schedulingapi.WorkloadReference{
				WorkloadName: "some-workload",
				TemplateName: "some-template",
			},
		},
	}

	if _, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pgPass, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Expected PodGroup with ParentCompositePodGroupName and WorkloadRef to be created successfully, got error: %v", err)
	}

	// Case 2: ParentCompositePodGroupName set and WorkloadRef NOT set -> Should FAIL
	pgFail := &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg-fail", Namespace: ns},
		Spec: schedulingapi.PodGroupSpec{
			SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
				Basic: &schedulingapi.BasicSchedulingPolicy{},
			},
			ParentCompositePodGroupName: new("some-cpg"),
		},
	}

	if _, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pgFail, metav1.CreateOptions{}); err == nil {
		t.Fatalf("Expected creation of PodGroup with ParentCompositePodGroupName and no WorkloadRef to fail, but it passed")
	}
}
