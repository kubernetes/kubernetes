/*
Copyright 2025 The Kubernetes Authors.

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
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
	testutils "k8s.io/kubernetes/test/integration/util"
)

const hostnameKey = "kubernetes.io/hostname"

// TestPodGroupSchedulingWithPodAntiAffinity is a regression test for the snapshot
// AssumePod/ForgetPod bug. During a PodGroup scheduling cycle the pods of a group
// are assumed into the scheduler snapshot one by one, so the snapshot must keep
// its havePodsWithRequiredAntiAffinityNodeInfoList consistent: otherwise the
// InterPodAffinity plugin does not see the required anti-affinity terms of pods
// assumed earlier in the same cycle and silently co-locates pods that should be
// kept apart.
//
// The anti-affinity pod is created and enqueued first so the pod group cycle
// (which orders pods of equal priority by enqueue time) always assumes it before
// the plain pod, and it is pinned to node-1 so the assignment is deterministic.
// The plain pod declares no anti-affinity of its own, so it can only be kept off
// node-1 via the existing-pods anti-affinity term recorded in the snapshot when
// the anti-affinity pod was assumed - exactly the path the fix repairs.
//
// The single-topology case is the deterministic regression signal: without the
// fix the plain pod ignores the assumed anti-affinity pod, the group is silently
// co-located on the only node and the "unschedulable" assertion fails. The
// multi-topology case additionally checks that the plain pod lands on node-2.
func TestPodGroupSchedulingWithPodAntiAffinity(t *testing.T) {
	nodeCapacity := map[v1.ResourceName]string{v1.ResourceCPU: "2"}
	podRequest := map[v1.ResourceName]string{v1.ResourceCPU: "1"}

	// antiAffinityPod repels other pods labeled app=foo from its hostname. It is
	// assumed first (created and enqueued before the plain pod) and pinned to
	// node-1 for a deterministic assignment.
	antiAffinityPod := st.MakePod().Name("anti-pod").Label("app", "foo").
		Req(podRequest).Container("image").
		NodeSelector(map[string]string{hostnameKey: "node-1"}).
		PodAntiAffinity(hostnameKey, &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}, st.PodAntiAffinityWithRequiredReq).
		PodGroupName("pg").Priority(200).Obj()
	// plainPod matches the anti-affinity pod's selector but declares no
	// anti-affinity itself, so it can only be kept off node-1 via the existing-pods
	// anti-affinity path that relies on the snapshot.
	plainPod := st.MakePod().Name("plain-pod").Label("app", "foo").
		Req(podRequest).Container("image").
		PodGroupName("pg").Priority(200).Obj()

	workload := st.MakeWorkload().Name("workload").
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t").MinCount(2).Obj()).Obj()
	podGroup := st.MakePodGroup().Name("pg").WorkloadRef("t", "workload").Priority(200).MinCount(2).Obj()

	tests := []struct {
		name  string
		nodes []*v1.Node
		steps []stepsframework.Step
	}{
		{
			name: "anti-affinity honored across the cycle on a multi-topology cluster",
			nodes: []*v1.Node{
				st.MakeNode().Name("node-1").Label(hostnameKey, "node-1").Capacity(nodeCapacity).Obj(),
				st.MakeNode().Name("node-2").Label(hostnameKey, "node-2").Capacity(nodeCapacity).Obj(),
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: podGroup,
				},
				{
					Name:              "Create both pods belonging to the group",
					CreatePodsInOrder: []*v1.Pod{antiAffinityPod, plainPod},
				},
				{
					Name:                 "Verify both pods are scheduled",
					WaitForPodsScheduled: []string{"anti-pod", "plain-pod"},
				},
				{
					Name: "Verify the anti-affinity pod landed on node-1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"anti-pod"},
						Nodes: sets.New("node-1"),
					},
				},
				{
					Name: "Verify the plain pod was kept off node-1 and landed on node-2",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"plain-pod"},
						Nodes: sets.New("node-2"),
					},
				},
			},
		},
		{
			name: "anti-affinity keeps the group unschedulable on a single-topology cluster",
			nodes: []*v1.Node{
				st.MakeNode().Name("node-1").Label(hostnameKey, "node-1").Capacity(nodeCapacity).Obj(),
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: podGroup,
				},
				{
					Name:              "Create both pods belonging to the group",
					CreatePodsInOrder: []*v1.Pod{antiAffinityPod, plainPod},
				},
				{
					Name:                     "Verify the group becomes unschedulable instead of co-locating",
					WaitForPodsUnschedulable: []string{"anti-pod", "plain-pod"},
				},
				{
					Name: "Verify PodGroup condition is set to Unschedulable",
					WaitForPodGroupCondition: &stepsframework.PodGroupConditionCheck{
						PodGroupName:    "pg",
						ConditionStatus: metav1.ConditionFalse,
						Reason:          schedulingapi.PodGroupReasonUnschedulable,
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			testCtx := testutils.InitTestSchedulerWithNS(t, "podgroup-anti-affinity",
				// disable backoff
				scheduler.WithPodMaxBackoffSeconds(0),
				scheduler.WithPodInitialBackoffSeconds(0))
			ns := testCtx.NS.Name

			commonSteps := []stepsframework.Step{
				{
					Name:        "Create Nodes",
					CreateNodes: tt.nodes,
				},
				{
					Name:            "Create workloads",
					CreateWorkloads: []*schedulingapi.Workload{workload},
				},
			}

			if err := stepsframework.RunSteps(testCtx, t, ns, append(commonSteps, tt.steps...)); err != nil {
				t.Fatal(err)
			}
		})
	}
}
