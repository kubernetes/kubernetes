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

package podgrouppreemption

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// TestMultiTierCompositePodGroupPreemption_NestedDisruptionModes tests 3-level deep CompositePodGroup
// hierarchies (Root CPG -> Intermediate Child CPG -> Child PodGroup -> Pods) with various combinations
// of nested and conflicting DisruptionModes (DisruptionModeAll, DisruptionModeSingle).
// It verifies that getHighestAllAncestor correctly identifies the atomic disruption boundary,
// evicting the exact subtree required without leaking disruptions to sibling groups.
func TestMultiTierCompositePodGroupPreemption_NestedDisruptionModes(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:                 true,
		features.CompositePodGroup:               true,
		features.TopologyAwareWorkloadScheduling: true,
		features.PodGroupPreemptionPolicy:        true,
	})

	tests := []struct {
		name               string
		nodes              []*v1.Node
		compositePodGroups []*schedulingv1alpha3.CompositePodGroup
		podGroups          []*schedulingv1beta1.PodGroup
		initialPods        []*v1.Pod
		preemptorCPG       *schedulingv1alpha3.CompositePodGroup
		preemptorPG        *schedulingv1beta1.PodGroup
		preemptorPods      []*v1.Pod
		expectedScheduled  []string
		expectedPreempted  []string
		expectedRunning    []string
	}{
		{
			name: "3-Tier Hierarchy: Intermediate CPG has DisruptionModeAll, Root CPG and Leaf PGs have DisruptionModeSingle - only intermediate subtree is evicted",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Label("kubernetes.io/hostname", "node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				// Root CPG with DisruptionModeSingle
				st.MakeCompositePodGroup().Name("cpg-root").Priority(10).BasicPolicy().DisruptionModeSingle().WorkloadRef("wl-victim", "t1").Obj(),
				// Intermediate Child CPG A with DisruptionModeAll
				st.MakeCompositePodGroup().Name("cpg-mid-a").Priority(10).BasicPolicy().DisruptionModeAll().ParentCompositePodGroup("cpg-root").WorkloadRef("wl-victim", "t1").Obj(),
				// Intermediate Child CPG B (sibling) with DisruptionModeSingle
				st.MakeCompositePodGroup().Name("cpg-mid-b").Priority(10).BasicPolicy().DisruptionModeSingle().ParentCompositePodGroup("cpg-root").WorkloadRef("wl-victim", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				// Child PGs under CPG A
				st.MakePodGroup().Name("pg-a1").Priority(10).MinCount(1).DisruptionModeSingle().ParentCompositePodGroup("cpg-mid-a").WorkloadRef("wl-victim", "t1").Obj(),
				st.MakePodGroup().Name("pg-a2").Priority(10).MinCount(1).DisruptionModeSingle().ParentCompositePodGroup("cpg-mid-a").WorkloadRef("wl-victim", "t1").Obj(),
				// Child PG under sibling CPG B
				st.MakePodGroup().Name("pg-b1").Priority(10).MinCount(1).DisruptionModeSingle().ParentCompositePodGroup("cpg-mid-b").WorkloadRef("wl-victim", "t1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("victim-a1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-a1").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				st.MakePod().Name("victim-a2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-a2").ZeroTerminationGracePeriod().Priority(10).Node("node2").Obj(),
				st.MakePod().Name("victim-b1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-b1").ZeroTerminationGracePeriod().Priority(10).Node("node3").Obj(),
			},
			preemptorCPG: st.MakeCompositePodGroup().Name("cpg-preemptor").Priority(100).BasicPolicy().DisruptionModeSingle().WorkloadRef("wl-preemptor", "t1").Obj(),
			preemptorPG:  st.MakePodGroup().Name("pg-preemptor").Priority(100).MinCount(1).ParentCompositePodGroup("cpg-preemptor").WorkloadRef("wl-preemptor", "t1").Obj(),
			preemptorPods: []*v1.Pod{
				// Preemptor requires 2 CPU on node1. Evicting victim-a1 on node1 triggers atomic eviction of all pods in cpg-mid-a (victim-a1 and victim-a2).
				// Sibling cpg-mid-b pod victim-b1 on node3 is NOT evicted.
				st.MakePod().Name("preemptor-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).Obj(),
			},
			expectedScheduled: []string{"preemptor-1"},
			expectedPreempted: []string{"victim-a1", "victim-a2"},
			expectedRunning:   []string{"victim-b1"},
		},
		{
			name: "3-Tier Hierarchy: Leaf PG has DisruptionModeAll, Root CPG and Mid CPG have DisruptionModeSingle - only leaf PG pods are evicted",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Label("kubernetes.io/hostname", "node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Priority(10).BasicPolicy().DisruptionModeSingle().WorkloadRef("wl-victim", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-mid-a").Priority(10).BasicPolicy().DisruptionModeSingle().ParentCompositePodGroup("cpg-root").WorkloadRef("wl-victim", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-mid-b").Priority(10).BasicPolicy().DisruptionModeSingle().ParentCompositePodGroup("cpg-root").WorkloadRef("wl-victim", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				// pg-a1 has DisruptionModeAll
				st.MakePodGroup().Name("pg-a1").Priority(10).MinCount(2).DisruptionModeAll().ParentCompositePodGroup("cpg-mid-a").WorkloadRef("wl-victim", "t1").Obj(),
				// pg-a2 (sibling PG under cpg-mid-a) has DisruptionModeSingle
				st.MakePodGroup().Name("pg-a2").Priority(10).MinCount(1).DisruptionModeSingle().ParentCompositePodGroup("cpg-mid-a").WorkloadRef("wl-victim", "t1").Obj(),
				// pg-b1 (under cpg-mid-b) has DisruptionModeSingle
				st.MakePodGroup().Name("pg-b1").Priority(10).MinCount(1).DisruptionModeSingle().ParentCompositePodGroup("cpg-mid-b").WorkloadRef("wl-victim", "t1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("victim-a1-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-a1").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				st.MakePod().Name("victim-a1-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-a1").ZeroTerminationGracePeriod().Priority(10).Node("node2").Obj(),
				st.MakePod().Name("victim-a2-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-a2").ZeroTerminationGracePeriod().Priority(10).Node("node2").Obj(),
				st.MakePod().Name("victim-filler-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				st.MakePod().Name("victim-b1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-b1").ZeroTerminationGracePeriod().Priority(10).Node("node3").Obj(),
			},
			preemptorCPG: st.MakeCompositePodGroup().Name("cpg-preemptor").Priority(100).BasicPolicy().DisruptionModeSingle().WorkloadRef("wl-preemptor", "t1").Obj(),
			preemptorPG:  st.MakePodGroup().Name("pg-preemptor").Priority(100).MinCount(1).ParentCompositePodGroup("cpg-preemptor").WorkloadRef("wl-preemptor", "t1").Obj(),
			preemptorPods: []*v1.Pod{
				// Needs 2 CPU on node1. Evicting victim-a1-1 causes victim-a1-2 to be evicted (DisruptionModeAll on pg-a1).
				// But victim-a2-1 (sibling in cpg-mid-a) and victim-b1 (in cpg-mid-b) stay running.
				st.MakePod().Name("preemptor-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).Obj(),
			},
			expectedScheduled: []string{"preemptor-1"},
			expectedPreempted: []string{"victim-a1-1", "victim-a1-2", "victim-filler-1"},
			expectedRunning:   []string{"victim-a2-1", "victim-b1"},
		},
		{
			name: "3-Tier Hierarchy: Root CPG has DisruptionModeAll, Mid CPGs and Leaf PGs have DisruptionModeSingle - entire 3-tier tree is evicted atomically",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Priority(10).BasicPolicy().DisruptionModeAll().WorkloadRef("wl-victim", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-mid-a").Priority(10).BasicPolicy().DisruptionModeSingle().ParentCompositePodGroup("cpg-root").WorkloadRef("wl-victim", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-mid-b").Priority(10).BasicPolicy().DisruptionModeSingle().ParentCompositePodGroup("cpg-root").WorkloadRef("wl-victim", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-a1").Priority(10).MinCount(1).DisruptionModeSingle().ParentCompositePodGroup("cpg-mid-a").WorkloadRef("wl-victim", "t1").Obj(),
				st.MakePodGroup().Name("pg-b1").Priority(10).MinCount(1).DisruptionModeSingle().ParentCompositePodGroup("cpg-mid-b").WorkloadRef("wl-victim", "t1").Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("victim-a1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-a1").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				st.MakePod().Name("victim-b1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-b1").ZeroTerminationGracePeriod().Priority(10).Node("node2").Obj(),
			},
			preemptorCPG: st.MakeCompositePodGroup().Name("cpg-preemptor").Priority(100).BasicPolicy().DisruptionModeSingle().WorkloadRef("wl-preemptor", "t1").Obj(),
			preemptorPG:  st.MakePodGroup().Name("pg-preemptor").Priority(100).MinCount(1).ParentCompositePodGroup("cpg-preemptor").WorkloadRef("wl-preemptor", "t1").Obj(),
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("preemptor-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-preemptor").ZeroTerminationGracePeriod().Priority(100).NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).Obj(),
			},
			expectedScheduled: []string{"preemptor-1"},
			expectedPreempted: []string{"victim-a1", "victim-b1"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-nested-disruption")
			cs, ns := testCtx.ClientSet, testCtx.NS.Name

			for _, n := range tt.nodes {
				if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, n, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create node %s: %v", n.Name, err)
				}
			}

			for _, cpg := range tt.compositePodGroups {
				cpgCopy := cpg.DeepCopy()
				cpgCopy.Namespace = ns
				if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create CompositePodGroup %s: %v", cpgCopy.Name, err)
				}
			}

			for _, pg := range tt.podGroups {
				pgCopy := pg.DeepCopy()
				pgCopy.Namespace = ns
				if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pgCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create PodGroup %s: %v", pgCopy.Name, err)
				}
			}

			for _, p := range tt.initialPods {
				pCopy := p.DeepCopy()
				pCopy.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create initial pod %s: %v", pCopy.Name, err)
				}
			}
			for _, p := range tt.initialPods {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodScheduled(cs, ns, p.Name)); err != nil {
					t.Fatalf("Failed to wait for initial pod %s to be scheduled: %v", p.Name, err)
				}
			}

			if tt.preemptorCPG != nil {
				cpgCopy := tt.preemptorCPG.DeepCopy()
				cpgCopy.Namespace = ns
				if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create preemptor CompositePodGroup %s: %v", cpgCopy.Name, err)
				}
			}
			if tt.preemptorPG != nil {
				pgCopy := tt.preemptorPG.DeepCopy()
				pgCopy.Namespace = ns
				if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pgCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create preemptor PodGroup %s: %v", pgCopy.Name, err)
				}
			}

			for _, p := range tt.preemptorPods {
				pCopy := p.DeepCopy()
				pCopy.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create preemptor pod %s: %v", pCopy.Name, err)
				}
			}

			for _, podName := range tt.expectedScheduled {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodScheduled(cs, ns, podName)); err != nil {
					t.Errorf("Expected pod %s to be scheduled: %v", podName, err)
				}
			}

			for _, podName := range tt.expectedPreempted {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					func(ctx context.Context) (bool, error) {
						pod, err := cs.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
						if err != nil {
							return apierrors.IsNotFound(err), nil
						}
						if pod.DeletionTimestamp != nil {
							return true, nil
						}
						_, cond := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget)
						return cond != nil, nil
					}); err != nil {
					t.Errorf("Expected pod %s to be preempted but wasn't: %v", podName, err)
				}
			}

			for _, podName := range tt.expectedRunning {
				pod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
				if err != nil {
					t.Errorf("Expected pod %s to be running, but failed to get: %v", podName, err)
					continue
				}
				if pod.DeletionTimestamp != nil {
					t.Errorf("Expected pod %s to stay running, but it has DeletionTimestamp set", podName)
				}
			}
		})
	}
}

// TestPartialGangPreemption_PodGroupAndCompositePodGroup tests partial gang preemption
// when the preemptor PodGroup specifies MinCount < TotalPods or a CompositePodGroup
// specifies MinGroupCount < TotalGroupCount.
// It verifies that the scheduler preemption evaluator only evicts victims necessary to satisfy
// MinCount / MinGroupCount, reprieving victims that would only serve optional unplaceable pods.
func TestPartialGangPreemption_PodGroupAndCompositePodGroup(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:                 true,
		features.CompositePodGroup:               true,
		features.TopologyAwareWorkloadScheduling: true,
		features.PodGroupPreemptionPolicy:        true,
	})

	tests := []struct {
		name               string
		nodes              []*v1.Node
		compositePodGroups []*schedulingv1alpha3.CompositePodGroup
		podGroups          []*schedulingv1beta1.PodGroup
		initialPods        []*v1.Pod
		preemptorCPG       *schedulingv1alpha3.CompositePodGroup
		preemptorPGs       []*schedulingv1beta1.PodGroup
		preemptorPods      []*v1.Pod
		expectedScheduled  []string
		expectedPreempted  []string
		expectedRunning    []string
	}{
		{
			name: "Partial PodGroup Gang Preemption: MinCount=2 with 3 preemptor pods, 3rd pod cannot fit - only victims for MinCount are evicted",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Label("kubernetes.io/hostname", "node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-victim-1").Priority(10).MinCount(1).Obj(),
				st.MakePodGroup().Name("pg-victim-2").Priority(10).MinCount(1).Obj(),
				st.MakePodGroup().Name("pg-victim-3").Priority(10).MinCount(1).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("victim-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-victim-1").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				st.MakePod().Name("victim-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-victim-2").ZeroTerminationGracePeriod().Priority(10).Node("node2").Obj(),
				st.MakePod().Name("victim-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-victim-3").ZeroTerminationGracePeriod().Priority(10).Node("node3").Obj(),
			},
			preemptorPGs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-preemptor-gang").Priority(100).MinCount(2).Obj(),
			},
			preemptorPods: []*v1.Pod{
				// preemptor-1 fits on node1 (2 CPU)
				st.MakePod().Name("preemptor-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-preemptor-gang").ZeroTerminationGracePeriod().Priority(100).NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).Obj(),
				// preemptor-2 fits on node2 (2 CPU)
				st.MakePod().Name("preemptor-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-preemptor-gang").ZeroTerminationGracePeriod().Priority(100).NodeSelector(map[string]string{"kubernetes.io/hostname": "node2"}).Obj(),
				// preemptor-3 requires 2 CPU on node1, but node1 only has 2 CPU total and preemptor-1 already occupies it.
				// Since MinCount=2 is already satisfied by preemptor-1 and preemptor-2, preemptor-3 cannot be scheduled.
				// Preemption evaluator should NOT disrupt victim-3 on node3.
				st.MakePod().Name("preemptor-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-preemptor-gang").ZeroTerminationGracePeriod().Priority(100).NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).Obj(),
			},
			expectedScheduled: []string{"preemptor-1", "preemptor-2"},
			expectedPreempted: []string{"victim-1", "victim-2"},
			expectedRunning:   []string{"victim-3"},
		},
		{
			name: "Partial CompositePodGroup Gang Preemption: MinGroupCount=1 with 2 child PGs, 2nd child PG cannot fit - only victims for MinGroupCount are evicted",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Label("kubernetes.io/hostname", "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-victim-1").Priority(10).MinCount(1).Obj(),
				st.MakePodGroup().Name("pg-victim-2").Priority(10).MinCount(1).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("victim-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-victim-1").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				st.MakePod().Name("victim-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-victim-2").ZeroTerminationGracePeriod().Priority(10).Node("node2").Obj(),
			},
			preemptorCPG: st.MakeCompositePodGroup().Name("cpg-preemptor-cgang").Priority(100).MinGroupCount(1).WorkloadRef("wl-preemptor-cg", "t1").Obj(),
			preemptorPGs: []*schedulingv1beta1.PodGroup{
				// Child PG 1: MinCount 1, fits on node1
				st.MakePodGroup().Name("pg-child-1").Priority(100).MinCount(1).ParentCompositePodGroup("cpg-preemptor-cgang").WorkloadRef("wl-preemptor-cg", "t1").Obj(),
				// Child PG 2: MinCount 2, needs 4 CPU on node2 (which only has 2 CPU), cannot fit
				st.MakePodGroup().Name("pg-child-2").Priority(100).MinCount(2).ParentCompositePodGroup("cpg-preemptor-cgang").WorkloadRef("wl-preemptor-cg", "t1").Obj(),
			},
			preemptorPods: []*v1.Pod{
				// Child PG 1 pod
				st.MakePod().Name("c-preemptor-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-child-1").ZeroTerminationGracePeriod().Priority(100).NodeSelector(map[string]string{"kubernetes.io/hostname": "node1"}).Obj(),
				// Child PG 2 pods (need 2x2 = 4 CPU on node2, but node2 only has 2 CPU)
				st.MakePod().Name("c-preemptor-2a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-child-2").ZeroTerminationGracePeriod().Priority(100).NodeSelector(map[string]string{"kubernetes.io/hostname": "node2"}).Obj(),
				st.MakePod().Name("c-preemptor-2b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-child-2").ZeroTerminationGracePeriod().Priority(100).NodeSelector(map[string]string{"kubernetes.io/hostname": "node2"}).Obj(),
			},
			expectedScheduled: []string{"c-preemptor-1"},
			expectedPreempted: []string{"victim-1"},
			expectedRunning:   []string{"victim-2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := testutils.InitTestSchedulerWithNS(t, "partial-gang-preemption")
			cs, ns := testCtx.ClientSet, testCtx.NS.Name

			for _, n := range tt.nodes {
				if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, n, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create node %s: %v", n.Name, err)
				}
			}

			for _, cpg := range tt.compositePodGroups {
				cpgCopy := cpg.DeepCopy()
				cpgCopy.Namespace = ns
				if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create CompositePodGroup %s: %v", cpgCopy.Name, err)
				}
			}

			for _, pg := range tt.podGroups {
				pgCopy := pg.DeepCopy()
				pgCopy.Namespace = ns
				if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pgCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create PodGroup %s: %v", pgCopy.Name, err)
				}
			}

			for _, p := range tt.initialPods {
				pCopy := p.DeepCopy()
				pCopy.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create initial pod %s: %v", pCopy.Name, err)
				}
			}
			for _, p := range tt.initialPods {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodScheduled(cs, ns, p.Name)); err != nil {
					t.Fatalf("Failed to wait for initial pod %s to be scheduled: %v", p.Name, err)
				}
			}

			if tt.preemptorCPG != nil {
				cpgCopy := tt.preemptorCPG.DeepCopy()
				cpgCopy.Namespace = ns
				if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create preemptor CompositePodGroup %s: %v", cpgCopy.Name, err)
				}
			}
			for _, pg := range tt.preemptorPGs {
				pgCopy := pg.DeepCopy()
				pgCopy.Namespace = ns
				if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pgCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create preemptor PodGroup %s: %v", pgCopy.Name, err)
				}
			}

			for _, p := range tt.preemptorPods {
				pCopy := p.DeepCopy()
				pCopy.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create preemptor pod %s: %v", pCopy.Name, err)
				}
			}

			for _, podName := range tt.expectedScheduled {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodScheduled(cs, ns, podName)); err != nil {
					t.Errorf("Expected pod %s to be scheduled: %v", podName, err)
				}
			}

			for _, podName := range tt.expectedPreempted {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					func(ctx context.Context) (bool, error) {
						pod, err := cs.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
						if err != nil {
							return apierrors.IsNotFound(err), nil
						}
						if pod.DeletionTimestamp != nil {
							return true, nil
						}
						_, cond := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget)
						return cond != nil, nil
					}); err != nil {
					t.Errorf("Expected pod %s to be preempted but wasn't: %v", podName, err)
				}
			}

			for _, podName := range tt.expectedRunning {
				pod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
				if err != nil {
					t.Errorf("Expected pod %s to be running, but failed to get: %v", podName, err)
					continue
				}
				if pod.DeletionTimestamp != nil {
					t.Errorf("Expected pod %s to stay running, but it has DeletionTimestamp set", podName)
				}
			}
		})
	}
}

// TestCompositePodGroupPreemption_CyclicReferencesAndEdgeCases tests scheduler robustness
// when invalid or cyclic composite group references, self-referencing loops, or dangling parent
// references are present during preemption simulation and dry-run ancestor traversal.
func TestCompositePodGroupPreemption_CyclicReferencesAndEdgeCases(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:                 true,
		features.CompositePodGroup:               true,
		features.TopologyAwareWorkloadScheduling: true,
		features.PodGroupPreemptionPolicy:        true,
	})

	t.Run("Cyclic CompositePodGroup references during preemption dry-run ancestor traversal", func(t *testing.T) {
		testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-cycle-defense")
		cs, ns := testCtx.ClientSet, testCtx.NS.Name

		node := st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// Create cyclic CompositePodGroups: CPG A -> CPG B -> CPG A
		cpgA := st.MakeCompositePodGroup().Name("cpg-cycle-a").Namespace(ns).Priority(10).BasicPolicy().ParentCompositePodGroup("cpg-cycle-b").WorkloadRef("wl-cycle", "t1").Obj()
		cpgB := st.MakeCompositePodGroup().Name("cpg-cycle-b").Namespace(ns).Priority(10).BasicPolicy().ParentCompositePodGroup("cpg-cycle-a").WorkloadRef("wl-cycle", "t1").Obj()
		if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgA, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create CPG A: %v", err)
		}
		if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgB, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create CPG B: %v", err)
		}

		pgCyclic := st.MakePodGroup().Name("pg-cyclic").Namespace(ns).Priority(10).MinCount(1).ParentCompositePodGroup("cpg-cycle-a").WorkloadRef("wl-cycle", "t1").Obj()
		if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pgCyclic, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PodGroup: %v", err)
		}

		// Create victim pod in pg-cyclic
		victimPod := st.MakePod().Name("victim-cyclic").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).
			Container("image").PodGroupName("pg-cyclic").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj()
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, victimPod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create victim pod: %v", err)
		}
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
			testutils.PodScheduled(cs, ns, victimPod.Name)); err != nil {
			t.Fatalf("Failed to schedule initial victim pod: %v", err)
		}

		// Create high-priority preemptor pod
		preemptorPod := st.MakePod().Name("preemptor-cyclic-test").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).
			Container("image").ZeroTerminationGracePeriod().Priority(100).Obj()
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptorPod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create preemptor pod: %v", err)
		}

		// Verify preemptor is scheduled and victim is preempted despite cyclic reference in victim's ancestor chain
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
			testutils.PodScheduled(cs, ns, preemptorPod.Name)); err != nil {
			t.Fatalf("Preemptor failed to schedule: %v", err)
		}

		// Verify scheduler did not deadlock and can schedule subsequent independent pods
		independentPod := st.MakePod().Name("independent-pod-1").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "0"}).
			Container("image").ZeroTerminationGracePeriod().Priority(50).Obj()
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, independentPod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create independent pod: %v", err)
		}
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
			testutils.PodScheduled(cs, ns, independentPod.Name)); err != nil {
			t.Fatalf("Scheduler appeared deadlocked after cyclic preemption evaluation: %v", err)
		}
	})

	t.Run("Multi-level loop reference in CompositePodGroup hierarchy", func(t *testing.T) {
		testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-multilevel-loop")
		cs, ns := testCtx.ClientSet, testCtx.NS.Name

		node := st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// 3-node loop: CPG1 -> CPG2 -> CPG3 -> CPG1
		cpg1 := st.MakeCompositePodGroup().Name("cpg-loop-1").Namespace(ns).Priority(10).BasicPolicy().ParentCompositePodGroup("cpg-loop-2").WorkloadRef("wl-loop", "t1").Obj()
		cpg2 := st.MakeCompositePodGroup().Name("cpg-loop-2").Namespace(ns).Priority(10).BasicPolicy().ParentCompositePodGroup("cpg-loop-3").WorkloadRef("wl-loop", "t1").Obj()
		cpg3 := st.MakeCompositePodGroup().Name("cpg-loop-3").Namespace(ns).Priority(10).BasicPolicy().ParentCompositePodGroup("cpg-loop-1").WorkloadRef("wl-loop", "t1").Obj()
		if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpg1, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create CPG 1: %v", err)
		}
		if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpg2, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create CPG 2: %v", err)
		}
		if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpg3, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create CPG 3: %v", err)
		}

		pgLoop := st.MakePodGroup().Name("pg-loop").Namespace(ns).Priority(10).MinCount(1).ParentCompositePodGroup("cpg-loop-1").WorkloadRef("wl-loop", "t1").Obj()
		if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pgLoop, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PodGroup: %v", err)
		}

		victimPod := st.MakePod().Name("victim-loop").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).
			Container("image").PodGroupName("pg-loop").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj()
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, victimPod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create victim pod: %v", err)
		}
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
			testutils.PodScheduled(cs, ns, victimPod.Name)); err != nil {
			t.Fatalf("Failed to schedule initial victim pod: %v", err)
		}

		preemptorPod := st.MakePod().Name("preemptor-loop-test").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).
			Container("image").ZeroTerminationGracePeriod().Priority(100).Obj()
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptorPod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create preemptor pod: %v", err)
		}

		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
			testutils.PodScheduled(cs, ns, preemptorPod.Name)); err != nil {
			t.Fatalf("Preemptor failed to schedule: %v", err)
		}
	})

	t.Run("Self-referencing CompositePodGroup handling", func(t *testing.T) {
		testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-self-loop")
		cs, ns := testCtx.ClientSet, testCtx.NS.Name

		node := st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		cpgSelf := st.MakeCompositePodGroup().Name("cpg-self").Namespace(ns).Priority(10).BasicPolicy().ParentCompositePodGroup("cpg-self").WorkloadRef("wl-self", "t1").Obj()
		if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgSelf, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create self-referencing CPG: %v", err)
		}

		pgSelf := st.MakePodGroup().Name("pg-self").Namespace(ns).Priority(10).MinCount(1).ParentCompositePodGroup("cpg-self").WorkloadRef("wl-self", "t1").Obj()
		if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pgSelf, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PodGroup: %v", err)
		}

		victimPod := st.MakePod().Name("victim-self").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).
			Container("image").PodGroupName("pg-self").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj()
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, victimPod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create victim pod: %v", err)
		}
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
			testutils.PodScheduled(cs, ns, victimPod.Name)); err != nil {
			t.Fatalf("Failed to schedule initial victim pod: %v", err)
		}

		preemptorPod := st.MakePod().Name("preemptor-self-test").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).
			Container("image").ZeroTerminationGracePeriod().Priority(100).Obj()
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptorPod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create preemptor pod: %v", err)
		}

		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
			testutils.PodScheduled(cs, ns, preemptorPod.Name)); err != nil {
			t.Fatalf("Preemptor failed to schedule: %v", err)
		}
	})

	t.Run("Dangling non-existent parent reference during preemption dry-run", func(t *testing.T) {
		testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-dangling-parent")
		cs, ns := testCtx.ClientSet, testCtx.NS.Name

		node := st.MakeNode().Name("node1").Label("kubernetes.io/hostname", "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// PG referencing non-existent parent CPG
		pgDangling := st.MakePodGroup().Name("pg-dangling").Namespace(ns).Priority(10).MinCount(1).ParentCompositePodGroup("cpg-nonexistent").WorkloadRef("wl-dangling", "t1").Obj()
		if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pgDangling, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PodGroup: %v", err)
		}

		victimPod := st.MakePod().Name("victim-dangling").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).
			Container("image").PodGroupName("pg-dangling").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj()
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, victimPod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create victim pod: %v", err)
		}
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
			testutils.PodScheduled(cs, ns, victimPod.Name)); err != nil {
			t.Fatalf("Failed to schedule initial victim pod: %v", err)
		}

		preemptorPod := st.MakePod().Name("preemptor-dangling-test").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).
			Container("image").ZeroTerminationGracePeriod().Priority(100).Obj()
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptorPod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create preemptor pod: %v", err)
		}

		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
			testutils.PodScheduled(cs, ns, preemptorPod.Name)); err != nil {
			t.Fatalf("Preemptor failed to schedule: %v", err)
		}
	})
}
