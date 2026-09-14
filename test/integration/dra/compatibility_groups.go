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
	"github.com/onsi/gomega"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

// compatDevice describes a device to place on a shared counter set, together
// with the compatibility groups it declares there.
type compatDevice struct {
	name   string
	groups []string
}

// compatGroupsSlices builds the two node-local ResourceSlices that make up one
// pool: a counter set slice and a device slice (a single ResourceSlice may set
// either sharedCounters or devices, not both). All devices draw from the same
// counter set and declare the given compatibility groups. The counter set has
// room for two devices (each consumes half of it), so two devices can be
// co-allocated only when their compatibility groups intersect.
func compatGroupsSlices(nodeName, driverName, counterSet string, devices ...compatDevice) (counters, deviceSlice *resourceapi.ResourceSlice) {
	c := st.MakeResourceSlice(nodeName, driverName)
	c.Name = nodeName + "-" + driverName + "-counters"
	c.Spec.Pool.ResourceSliceCount = 2
	c.Spec.SharedCounters = []resourceapi.CounterSet{{
		Name:     counterSet,
		Counters: map[string]resourceapi.Counter{"mem": {Value: resource.MustParse("8")}},
	}}

	d := st.MakeResourceSlice(nodeName, driverName)
	d.Name = nodeName + "-" + driverName + "-devices"
	d.Spec.Pool.ResourceSliceCount = 2
	for _, dev := range devices {
		d.Spec.Devices = append(d.Spec.Devices, resourceapi.Device{
			Name: dev.name,
			ConsumesCounters: []resourceapi.DeviceCounterConsumption{{
				CounterSet:          counterSet,
				Counters:            map[string]resourceapi.Counter{"mem": {Value: resource.MustParse("4")}},
				CompatibilityGroups: dev.groups,
			}},
		})
	}
	return c.Obj(), d.Obj()
}

// createCompatGroupsPool creates both slices of a compatibility-groups pool and
// returns the created device slice (the one carrying compatibilityGroups).
func createCompatGroupsPool(tCtx ktesting.TContext, nodeName, driverName, counterSet string, devices ...compatDevice) *resourceapi.ResourceSlice {
	counters, deviceSlice := compatGroupsSlices(nodeName, driverName, counterSet, devices...)
	createSlice(tCtx, counters)
	return createSlice(tCtx, deviceSlice)
}

// testCompatibilityGroups exercises the DRADeviceCompatibilityGroups feature end
// to end through a real apiserver + scheduler. With the feature enabled the
// scheduler must enforce group intersection when co-allocating devices on a
// counter set; with it disabled the apiserver must drop the field on write.
func testCompatibilityGroups(tCtx ktesting.TContext, enabled bool) {
	if !enabled {
		tCtx.Run("FieldDroppedWhenDisabled", testCompatibilityGroupsFieldDropped)
		return
	}
	tCtx.Run("FieldPersistedWhenEnabled", testCompatibilityGroupsFieldPersisted)
	tCtx.Run("CompatibleCoAllocation", testCompatibleCoAllocation)
	tCtx.Run("IncompatibleDevicesUnschedulable", testIncompatibleUnschedulable)
	tCtx.Run("IncompatibleRejectedCompatibleAdmitted", testIncompatibleAcrossNodes)
}

// testCompatibleCoAllocation verifies that two devices declaring the same group
// on a counter set can be co-allocated to one claim.
func testCompatibleCoAllocation(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)
	node := firstNodeName(tCtx)

	createCompatGroupsPool(tCtx, node, driverName, "gpu-0",
		compatDevice{"dev-0", []string{"mig"}},
		compatDevice{"dev-1", []string{"mig"}},
	)
	startScheduler(tCtx)

	claim := st.MakeResourceClaim().Name("claim").Namespace(namespace).
		RequestWithNameCount("req", class.Name, 2).Obj()
	createdClaim := createClaim(tCtx, namespace, "-compat", class, claim)

	pod := st.MakePod().Name(podName).Namespace(namespace).Container("c").Obj()
	createdPod := createPod(tCtx, namespace, "-compat", pod, createdClaim)

	scheduled := waitForPodScheduled(tCtx, namespace, createdPod.Name)
	tCtx.Expect(scheduled.Spec.NodeName).To(gomega.Equal(node))

	allocated, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, createdClaim.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get allocated claim")
	tCtx.Expect(allocated.Status.Allocation).ToNot(gomega.BeNil(), "claim should be allocated")
	tCtx.Expect(allocated.Status.Allocation.Devices.Results).To(gomega.HaveLen(2))
}

// testIncompatibleUnschedulable verifies that a pod is left unschedulable when
// the only devices its claim can draw from declare disjoint compatibility groups
// on a shared counter set. The claim asks for two devices, the node's only two
// devices are "mig" and "vgpu" (no intersection) and both draw from the same
// counter set, so they cannot be co-allocated and no node can satisfy the claim.
func testIncompatibleUnschedulable(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)
	node := firstNodeName(tCtx)

	createCompatGroupsPool(tCtx, node, driverName, "gpu-0",
		compatDevice{"dev-0", []string{"mig"}},
		compatDevice{"dev-1", []string{"vgpu"}},
	)
	startScheduler(tCtx)

	claim := st.MakeResourceClaim().Name("claim").Namespace(namespace).
		RequestWithNameCount("req", class.Name, 2).Obj()
	createdClaim := createClaim(tCtx, namespace, "-incompat", class, claim)

	pod := st.MakePod().Name(podName).Namespace(namespace).Container("c").Obj()
	createdPod := createPod(tCtx, namespace, "-incompat", pod, createdClaim)

	expectPodUnschedulable(tCtx, createdPod, "cannot allocate all claims")

	unallocated, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, createdClaim.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get claim")
	tCtx.Expect(unallocated.Status.Allocation).To(gomega.BeNil(), "incompatible claim must not be allocated")
}

// testIncompatibleAcrossNodes verifies that the scheduler refuses to co-allocate
// an incompatible pair of devices (the only candidates on one node) and instead
// places the pod on a node whose devices share a compatibility group. A second
// pod, created after the compatible node is exhausted, is then left
// unschedulable: this confirms the first pod chose the compatible node because
// the incompatible node genuinely cannot satisfy the claim, not by chance.
func testIncompatibleAcrossNodes(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)
	nodeA, nodeB := twoNodeNames(tCtx)

	// nodeA offers only an incompatible pair; nodeB offers a compatible pair.
	createCompatGroupsPool(tCtx, nodeA, driverName, "gpu-0",
		compatDevice{"dev-0", []string{"mig"}},
		compatDevice{"dev-1", []string{"vgpu"}},
	)
	createCompatGroupsPool(tCtx, nodeB, driverName, "gpu-0",
		compatDevice{"dev-0", []string{"mig"}},
		compatDevice{"dev-1", []string{"mig"}},
	)
	startScheduler(tCtx)

	claim := st.MakeResourceClaim().Name("claim").Namespace(namespace).
		RequestWithNameCount("req", class.Name, 2).Obj()
	createdClaim := createClaim(tCtx, namespace, "-cross", class, claim)

	pod := st.MakePod().Name(podName).Namespace(namespace).Container("c").Obj()
	createdPod := createPod(tCtx, namespace, "-cross", pod, createdClaim)

	scheduled := waitForPodScheduled(tCtx, namespace, createdPod.Name)
	tCtx.Expect(scheduled.Spec.NodeName).To(gomega.Equal(nodeB),
		"pod must land on the node with a compatible device pair, not the incompatible one")

	// The first pod consumed nodeB's whole counter set (both "mig" devices). A
	// second identical pod now has nowhere to go: nodeB is exhausted and nodeA's
	// only pair is incompatible, so it must stay unschedulable.
	claim2 := st.MakeResourceClaim().Name("claim").Namespace(namespace).
		RequestWithNameCount("req", class.Name, 2).Obj()
	createdClaim2 := createClaim(tCtx, namespace, "-cross-2", class, claim2)

	pod2 := st.MakePod().Name(podName).Namespace(namespace).Container("c").Obj()
	createdPod2 := createPod(tCtx, namespace, "-cross-2", pod2, createdClaim2)

	expectPodUnschedulable(tCtx, createdPod2, "cannot allocate all claims")
}

// testCompatibilityGroupsFieldPersisted verifies the enabled half of the
// feature-gate round-trip: with the feature on, compatibilityGroups written to a
// ResourceSlice survives create and is returned on read-back. (The disabled half
// is testCompatibilityGroupsFieldDropped; the ratcheting case - preserved on
// update when the old object already used it - is a unit strategy test, since it
// needs a gate toggle the fixed-per-config integration harness cannot express.)
func testCompatibilityGroupsFieldPersisted(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	_, driverName := createTestClass(tCtx, namespace)
	node := firstNodeName(tCtx)

	created := createCompatGroupsPool(tCtx, node, driverName, "gpu-0",
		compatDevice{"dev-0", []string{"mig"}},
	)
	got, err := tCtx.Client().ResourceV1().ResourceSlices().Get(tCtx, created.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "read the device slice back")
	tCtx.Expect(got.Spec.Devices).To(gomega.HaveLen(1))
	tCtx.Expect(got.Spec.Devices[0].ConsumesCounters[0].CompatibilityGroups).To(
		gomega.ConsistOf("mig"), "compatibilityGroups must be persisted when the feature is enabled")
}

// testCompatibilityGroupsFieldDropped verifies that, with the feature gate
// disabled, the apiserver drops compatibilityGroups on write.
func testCompatibilityGroupsFieldDropped(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	_, driverName := createTestClass(tCtx, namespace)
	node := firstNodeName(tCtx)

	created := createCompatGroupsPool(tCtx, node, driverName, "gpu-0",
		compatDevice{"dev-0", []string{"mig"}},
	)
	tCtx.Expect(created.Spec.Devices).To(gomega.HaveLen(1))
	tCtx.Expect(created.Spec.Devices[0].ConsumesCounters[0].CompatibilityGroups).To(
		gomega.BeEmpty(), "compatibilityGroups must be dropped when the feature is disabled")
}

func firstNodeName(tCtx ktesting.TContext) string {
	tCtx.Helper()
	nodes, err := tCtx.Client().CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list nodes")
	tCtx.Expect(nodes.Items).ToNot(gomega.BeEmpty(), "need at least one node")
	return nodes.Items[0].Name
}

func twoNodeNames(tCtx ktesting.TContext) (string, string) {
	tCtx.Helper()
	nodes, err := tCtx.Client().CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list nodes")
	if len(nodes.Items) < 2 {
		tCtx.Fatalf("need at least two nodes, have %d", len(nodes.Items))
	}
	return nodes.Items[0].Name, nodes.Items[1].Name
}
