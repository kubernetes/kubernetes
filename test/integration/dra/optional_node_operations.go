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

package dra

import (
	"fmt"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/ktesting"
	"k8s.io/component-helpers/nodedeclaredfeatures/features/draoptionalnodeoperations"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// testOptionalNodeOperations verifies DRA optional node operations.
// Note: This test relies on the shared test setup (in dra.go) having configured
// worker-0 to declare support for DRAOptionalNodeOperations when this feature is enabled.
// All other nodes (worker-1 through worker-7) lack this declared feature.
func testOptionalNodeOperations(tCtx ktesting.TContext, enabled bool) {
	if !enabled {
		return
	}
	// Adds a node, so it must not run in parallel with other tests of the same
	// configuration: some of them depend on the set of nodes.
	tCtx.Run("pod-scheduled-after-node-starts-declaring-feature", testOptionalNodeOperationsRequeue)
	tCtx.Parallel()

	tests := map[string]struct {
		usePodGroup     bool
		nodeSelector    map[string]string
		expectScheduled bool
		expectedNode    string
	}{
		"pod-scheduled-on-supported-node": {
			usePodGroup:     false,
			expectScheduled: true,
			expectedNode:    "worker-0",
		},
		"pod-unschedulable-on-unsupported-node": {
			usePodGroup:     false,
			nodeSelector:    map[string]string{"kubernetes.io/hostname": "worker-1"},
			expectScheduled: false,
		},
		"podgroup-scheduled-on-supported-node": {
			usePodGroup:     true,
			expectScheduled: true,
			expectedNode:    "worker-0",
		},
		"podgroup-unschedulable-on-unsupported-node": {
			usePodGroup:     true,
			nodeSelector:    map[string]string{"kubernetes.io/hostname": "worker-1"},
			expectScheduled: false,
		},
	}

	for name, tt := range tests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			if tt.usePodGroup && !utilfeature.DefaultFeatureGate.Enabled(features.DRAWorkloadResourceClaims) {
				tCtx.Skip("PodGroup ResourceClaims need the DRAWorkloadResourceClaims feature gate")
			}
			tCtx.Parallel()
			namespace := createTestNamespace(tCtx, nil)
			startScheduler(tCtx)
			class, driverName := createTestClass(tCtx, namespace)

			slice := st.MakeResourceSliceWithAllNodes("shared-slice", driverName).
				Devices(device1).
				SkipNodeOperations(resourceapi.SkipNodeOperationAll).
				Obj()
			createSlice(tCtx, slice)

			claimObj := createClaim(tCtx, namespace, "", class, claim)

			var podGroup *schedulingapi.PodGroup
			if tt.usePodGroup {
				pg := &schedulingapi.PodGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "podgroup",
						Namespace: namespace,
					},
					Spec: schedulingapi.PodGroupSpec{
						SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
							Basic: &schedulingapi.BasicSchedulingPolicy{},
						},
						ResourceClaims: []schedulingapi.PodGroupResourceClaim{
							{
								Name:              claimObj.Name,
								ResourceClaimName: &claimObj.Name,
							},
						},
					},
				}
				podGroup = must(tCtx, tCtx.Client().SchedulingV1beta1().PodGroups(namespace).Create, pg, metav1.CreateOptions{})
			}

			numPods := 1
			if tt.usePodGroup {
				numPods = 2
			}

			var pods []*v1.Pod
			for i := range numPods {
				p := podWithClaimName.DeepCopy()
				if tt.usePodGroup {
					p.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
						PodGroupName: &podGroup.Name,
					}
				}
				// Apply node selector to the last pod if specified to test rejection.
				if i == numPods-1 && tt.nodeSelector != nil {
					p.Spec.NodeSelector = tt.nodeSelector
				}
				pod := createPod(tCtx, namespace, fmt.Sprintf("-%d", i), p, claimObj)
				pods = append(pods, pod)
			}

			if tt.expectScheduled {
				// The PodScheduled condition checked by [waitForPodScheduled] could
				// converge to either a True or False status due to a race
				// condition, but Pods should still ultimately be scheduled:
				// https://github.com/kubernetes/kubernetes/issues/139417#issuecomment-4651436886
				for _, pod := range pods {
					tCtx.Eventually(func(tCtx ktesting.TContext) (string, error) {
						p, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
						if err != nil {
							return "", err
						}
						return p.Spec.NodeName, nil
					}).WithTimeout(schedulingTimeout).Should(
						gomega.Equal(tt.expectedNode),
						"pod %s should be scheduled to %s", pod.Name, tt.expectedNode,
					)
				}
			} else {
				// Verify that the pod with node selector (the last one) is unschedulable.
				unschedulablePod := pods[numPods-1]
				expectPodUnschedulable(tCtx, unschedulablePod, "")
			}
		})
	}
}

// testOptionalNodeOperationsRequeue verifies that a pod which is pending because no node
// declares DRAOptionalNodeOperations gets scheduled as soon as a node starts declaring it,
// instead of waiting for the periodic retry of unschedulable pods (5 minutes).
func testOptionalNodeOperationsRequeue(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	startScheduler(tCtx)
	class, driverName := createTestClass(tCtx, namespace)
	createSlice(tCtx, st.MakeResourceSliceWithAllNodes("shared-slice", driverName).
		Devices(device1).
		SkipNodeOperations(resourceapi.SkipNodeOperationAll).
		Obj())

	// The taint keeps pods of other tests off this node while it exists.
	taint := v1.Taint{Key: namespace + ".example.com/dedicated", Effect: v1.TaintEffectNoSchedule}
	node := createDedicatedNode(tCtx, namespace+"-node", taint)

	claimObj := createClaim(tCtx, namespace, "", class, claim)
	pod := podWithClaimName.DeepCopy()
	pod.Spec.NodeSelector = map[string]string{"kubernetes.io/hostname": node.Name}
	pod.Spec.Tolerations = []v1.Toleration{{Key: taint.Key, Operator: v1.TolerationOpExists, Effect: taint.Effect}}
	pod = createPod(tCtx, namespace, "", pod, claimObj)
	// Other rejections, for example before the scheduler knows about the new node,
	// would not exercise the requeueing for declared features.
	tCtx.Eventually(func(tCtx ktesting.TContext) (string, error) {
		p, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
		if err != nil {
			return "", err
		}
		for _, condition := range p.Status.Conditions {
			if condition.Type == v1.PodScheduled && condition.Status == v1.ConditionFalse {
				return condition.Message, nil
			}
		}
		return "", nil
	}).WithTimeout(schedulingTimeout).Should(gomega.ContainSubstring("cannot allocate all claims"), "pod should be rejected by the DynamicResources plugin")

	node.Status.DeclaredFeatures = append(node.Status.DeclaredFeatures, draoptionalnodeoperations.DRAOptionalNodeOperationsFeatureGate)
	_, err := tCtx.Client().CoreV1().Nodes().UpdateStatus(tCtx, node, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "declare DRAOptionalNodeOperations on node %s", node.Name)

	tCtx.Eventually(func(tCtx ktesting.TContext) (string, error) {
		p, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
		if err != nil {
			return "", err
		}
		return p.Spec.NodeName, nil
	}).WithTimeout(schedulingTimeout).Should(gomega.Equal(node.Name), "pod should get scheduled once node %s declares DRAOptionalNodeOperations", node.Name)
}

// createDedicatedNode creates a ready node with the given taint and removes it when the test is done.
func createDedicatedNode(tCtx ktesting.TContext, name string, taint v1.Taint) *v1.Node {
	tCtx.Helper()
	node, err := tCtx.Client().CoreV1().Nodes().Create(tCtx, &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"kubernetes.io/hostname": name},
		},
	}, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create node %s", name)
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(tCtx.Client().CoreV1().Nodes().Delete(tCtx, name, metav1.DeleteOptions{}), "delete node %s", name)
	})

	node.Status = v1.NodeStatus{
		Capacity: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse(nodeCPUCapacity),
			v1.ResourceMemory: resource.MustParse(nodeMemoryCapacity),
			v1.ResourcePods:   *resource.NewScaledQuantity(maxPodsPerNode, 0),
		},
		Phase:      v1.NodeRunning,
		Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
	}
	node, err = tCtx.Client().CoreV1().Nodes().UpdateStatus(tCtx, node, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "set status of node %s", name)

	// Replaces the not-ready taint added by the TaintNodesByCondition admission check.
	node.Spec.Taints = []v1.Taint{taint}
	node, err = tCtx.Client().CoreV1().Nodes().Update(tCtx, node, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "taint node %s", name)
	return node
}
