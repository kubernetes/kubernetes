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
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func testPartitionableDevices(tCtx ktesting.TContext, enabled bool) {
	if enabled {
		tCtx.Run("PerDeviceNodeSelection", testPerDeviceNodeSelection)
		tCtx.Run("MultiHostDevice", testPartitionableDevicesWithMultiHostDevice)
	} else {
		testDisabled(tCtx)
	}
}

// testDisabled verifies that creating ResourceSlices with node selection
// perDeviceNodeSelection fails when the Partitionable Devices feature is
// disabled.
func testDisabled(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	_, driverName := createTestClass(tCtx, namespace)

	slice := st.MakeResourceSliceWithPerDeviceNodeSelection("slice", driverName)
	_, err := tCtx.Client().ResourceV1().ResourceSlices().Create(tCtx, slice.Obj(), metav1.CreateOptions{})
	require.Error(tCtx, err, "slice should have become invalid after dropping PartitionableDevices")
}

// testPerDeviceNodeSelection verifies that pods are scheduled
// on the correct nodes when they are allocated devices that
// speficy node selection using the perDeviceNodeSelection field
// that was introduced as part of the Partitionable Devices
// feature.
func testPerDeviceNodeSelection(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)

	nodes, err := tCtx.Client().CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list nodes")

	slice := st.MakeResourceSliceWithPerDeviceNodeSelection("slice", driverName)
	for _, node := range nodes.Items {
		slice.Device(fmt.Sprintf("device-for-%s", node.Name), st.NodeName(node.Name))
	}
	createSlice(tCtx, slice.Obj())

	startScheduler(tCtx)

	for i := range nodes.Items {
		claim := st.MakeResourceClaim().
			Name(fmt.Sprintf("claim-%d", i)).
			Namespace(namespace).
			Request(class.Name).
			Obj()
		createdClaim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, fmt.Sprintf("claim name %q", createdClaim.Name))

		pod := st.MakePod().Name(podName).Namespace(namespace).
			Container("my-container").
			Obj()
		createdPod := createPod(tCtx, namespace, fmt.Sprintf("-%d", i), pod, claim)

		scheduledPod := waitForPodScheduled(tCtx, namespace, createdPod.Name)

		allocatedClaim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, createdClaim.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err, fmt.Sprintf("get claim %q", createdClaim.Name))
		tCtx.Expect(allocatedClaim).To(gomega.HaveField("Status.Allocation", gomega.Not(gomega.BeNil())), "Claim should have been allocated.")

		nodeName := scheduledPod.Spec.NodeName
		expectedAllocatedDevice := fmt.Sprintf("device-for-%s", nodeName)
		tCtx.Expect(allocatedClaim.Status.Allocation.Devices.Results[0].Device).To(gomega.Equal(expectedAllocatedDevice))
	}
}

// testPartitionableDevicesWithMultiHostDevice verifies that multiple pods sharing
// a ResourceClaim that is assigned a multi-host devices gets scheduled correctly
// on the nodes selected by the node selector on the device.
func testPartitionableDevicesWithMultiHostDevice(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)

	nodes, err := tCtx.Client().CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list nodes")

	minNodeCount := 4
	if nodeCount := len(nodes.Items); nodeCount < minNodeCount {
		tCtx.Errorf("found only %d nodes, need at least %d", nodeCount, minNodeCount)
	}

	deviceNodes := []string{
		nodes.Items[0].Name,
		nodes.Items[1].Name,
		nodes.Items[2].Name,
		nodes.Items[3].Name,
	}

	slice := st.MakeResourceSliceWithPerDeviceNodeSelection("slice", driverName)
	slice.Device("multi-host-device", &v1.NodeSelector{
		NodeSelectorTerms: []v1.NodeSelectorTerm{{
			MatchExpressions: []v1.NodeSelectorRequirement{{
				Key:      "kubernetes.io/hostname",
				Operator: v1.NodeSelectorOpIn,
				Values:   deviceNodes,
			}},
		}},
	})
	createSlice(tCtx, slice.Obj())

	startScheduler(tCtx)

	claim := st.MakeResourceClaim().
		Name("multi-host-claim").
		Namespace(namespace).
		Request(class.Name).
		Obj()
	createdClaim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, fmt.Sprintf("claim name %q", createdClaim.Name))

	labelKey := "app"
	labelValue := "multiHost"
	labelSelector := &metav1.LabelSelector{
		MatchExpressions: []metav1.LabelSelectorRequirement{
			{
				Key:      labelKey,
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{labelValue},
			},
		},
	}
	var podNames []string
	for i := range 4 {
		pod := st.MakePod().
			Name(podName).
			Namespace(namespace).
			Labels(map[string]string{labelKey: labelValue}).
			PodAntiAffinity("kubernetes.io/hostname", labelSelector, st.PodAntiAffinityWithRequiredReq).
			Container("my-container").
			Obj()
		createdPod := createPod(tCtx, namespace, fmt.Sprintf("-%d", i), pod, claim)
		podNames = append(podNames, createdPod.Name)
	}

	var scheduledOnNodes []string
	for _, podName := range podNames {
		scheduledPod := waitForPodScheduled(tCtx, namespace, podName)
		scheduledOnNodes = append(scheduledOnNodes, scheduledPod.Spec.NodeName)
	}

	tCtx.Expect(scheduledOnNodes).To(gomega.ConsistOf(deviceNodes))
}
