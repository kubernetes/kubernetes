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
	"time"

	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
	"k8s.io/utils/ptr"
)

// testPodGroupDRAAllocationConflict reproduces https://github.com/kubernetes/kubernetes/issues/138765.
//
// A gang PodGroup of two pods each requests one device via a prioritized list
// (firstAvailable) that prefers a whole device but falls back to half of it.
// The single underlying device is modeled as a partitionable device backed by a
// shared counter: it can be consumed either as one "whole" device or as two
// "half" devices, but not both at once.
//
// The gang is satisfiable: each pod should get one half. But the scheduler
// allocates DRA devices one pod at a time and never reconsiders an earlier
// pod's choice. The first pod greedily takes the whole device (first in the
// firstAvailable list), exhausting the shared counter, so the second pod cannot
// be allocated and the whole gang is stuck.
//
// This test documents that current (buggy) behavior: the gang never schedules,
// one pod's claim is left holding the whole device while the other reports the
// DRA "cannot allocate all claims" failure, and the gang stays below minCount.
// When #138765 is fixed, the assertions at the end of this function should be
// flipped to expect both pods scheduled, with one pod allocated "half-1" and the
// other "half-2".
func testPodGroupDRAAllocationConflict(tCtx ktesting.TContext) {
	tCtx.Parallel()

	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)
	startScheduler(tCtx)

	// Model one "whole-or-two-halves" device on a node. The shared counter and
	// the devices that consume it must live in the SAME pool: the allocator
	// resolves a device's ConsumesCounters reference only against SharedCounters
	// in slices belonging to the same pool (slices are grouped by {driver, pool
	// name}). A ResourceSlice may set either Devices or SharedCounters but not
	// both, so each pool spans two slices and declares ResourceSliceCount: 2 —
	// the allocator treats a pool as usable only once it has observed all slices
	// in it.
	//
	// "whole" consumes the entire shared counter (2 widgets); "half-1" and
	// "half-2" each consume 1 widget, so the two halves together also exhaust the
	// counter. Allocating "whole" therefore precludes either half, and vice
	// versa. Each device carries a "size" attribute ("whole"/"half") so claims
	// can select by size via CEL (CEL selectors operate on device attributes, not
	// device names).
	const attrDomain = "example.com"
	const counterSetName = "widget-counter"
	const counterName = "widgets"
	sizeAttr := func(size string) map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
		return map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			resourceapi.QualifiedName(attrDomain + "/size"): {StringValue: ptr.To(size)},
		}
	}
	consumes := func(value string) []resourceapi.DeviceCounterConsumption {
		return []resourceapi.DeviceCounterConsumption{{
			CounterSet: counterSetName,
			Counters: map[string]resourceapi.Counter{
				counterName: {Value: resource.MustParse(value)},
			},
		}}
	}
	// createWholeOrHalvesPool publishes the counter+device slices for one such
	// device on the given node/pool.
	createWholeOrHalvesPool := func(node, poolName string) {
		pool := resourceapi.ResourcePool{Name: poolName, ResourceSliceCount: 2}
		createSlice(tCtx, &resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{GenerateName: "counters-"},
			Spec: resourceapi.ResourceSliceSpec{
				NodeName: ptr.To(node),
				Driver:   driverName,
				Pool:     pool,
				SharedCounters: []resourceapi.CounterSet{{
					Name:     counterSetName,
					Counters: map[string]resourceapi.Counter{counterName: {Value: resource.MustParse("2")}},
				}},
			},
		})
		createSlice(tCtx, &resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{GenerateName: "devices-"},
			Spec: resourceapi.ResourceSliceSpec{
				NodeName: ptr.To(node),
				Driver:   driverName,
				Pool:     pool,
				Devices: []resourceapi.Device{
					{Name: "whole", Attributes: sizeAttr("whole"), ConsumesCounters: consumes("2")},
					{Name: "half-1", Attributes: sizeAttr("half"), ConsumesCounters: consumes("1")},
					{Name: "half-2", Attributes: sizeAttr("half"), ConsumesCounters: consumes("1")},
				},
			},
		})
	}
	// The gang and the positive control get independent devices on different
	// nodes/pools so they never compete; this lets the control's allocation
	// linger (integration tests have no real driver to deallocate it) without
	// affecting the gang.
	createWholeOrHalvesPool("worker-0", "gang-pool")
	createWholeOrHalvesPool("worker-1", "control-pool")

	// A prioritized-list request preferring the whole device, falling back to a
	// half. createClaim rewrites every subrequest's DeviceClassName to the test
	// class, so the whole-vs-half distinction is expressed via CEL selectors on
	// the device's "size" attribute rather than via separate device classes.
	wholeSelector := fmt.Sprintf(`device.attributes[%q].size == "whole"`, attrDomain)
	halfSelector := fmt.Sprintf(`device.attributes[%q].size == "half"`, attrDomain)
	makeClaim := func() *resourceapi.ResourceClaim {
		return st.MakeResourceClaim().
			Name("conflict-claim").
			Namespace(namespace).
			RequestWithPrioritizedList(
				st.SubRequestWithSelector("whole", class.Name, wholeSelector),
				st.SubRequestWithSelector("half", class.Name, halfSelector),
			).
			Obj()
	}

	// nodeSelectorPod builds a standalone pod pinned to a node, so it can only be
	// satisfied by that node's device pool.
	nodeSelectorPod := func(node string) *v1.Pod {
		return st.MakePod().Name(podName).Namespace(namespace).Container("my-container").
			NodeSelector(map[string]string{"kubernetes.io/hostname": node}).Obj()
	}

	// Positive control: before testing the gang, prove the setup is actually
	// schedulable. A single standalone pod (no PodGroup), pinned to the control
	// node, must schedule and be allocated the preferred "whole" device. This
	// guards against the test passing for the wrong reason: if the
	// ResourceSlices, counters, CEL selectors, or feature gates were
	// misconfigured, this pod would fail too, so a green "gang does not schedule"
	// assertion could not be mistaken for a setup/validation failure. The control
	// uses its own node/pool, so its (never-deallocated, since integration tests
	// have no real driver) allocation cannot affect the gang.
	controlClaim := createClaim(tCtx, namespace, "-control", class, makeClaim())
	controlPod := createPod(tCtx, namespace, "-control", nodeSelectorPod("worker-1"), controlClaim)
	waitForPodScheduled(tCtx, namespace, controlPod.Name)
	gotControl := waitForClaimAllocatedToDevice(tCtx, namespace, controlClaim.Name, schedulingTimeout)
	tCtx.Expect(gotControl.Status.Allocation.Devices.Results).To(gomega.HaveLen(1), "control claim should allocate exactly one device")
	tCtx.Expect(gotControl.Status.Allocation.Devices.Results[0].Device).To(
		gomega.Equal("whole"),
		"control pod should get the preferred whole device, confirming the prioritized list and shared counter work",
	)

	// One ResourceClaim per pod: the bug is about independent per-pod claims
	// competing for the same shared counter, so the pods must not share a claim.
	claim1 := createClaim(tCtx, namespace, "-1", class, makeClaim())
	claim2 := createClaim(tCtx, namespace, "-2", class, makeClaim())

	podGroup, err := tCtx.Client().SchedulingV1alpha3().PodGroups(namespace).Create(tCtx, &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "podgroup-dra-conflict"},
		Spec: schedulingapi.PodGroupSpec{
			SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
				Gang: &schedulingapi.GangSchedulingPolicy{MinCount: 2},
			},
		},
	}, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create PodGroup")
	schedGroup := &v1.PodSchedulingGroup{PodGroupName: &podGroup.Name}

	newPod := func() *v1.Pod {
		// Pin the gang to the gang node so it competes only for that node's
		// single whole-or-halves device.
		pod := nodeSelectorPod("worker-0")
		pod.Spec.SchedulingGroup = schedGroup
		return pod
	}
	pod1 := createPod(tCtx, namespace, "-a", newPod(), claim1)
	pod2 := createPod(tCtx, namespace, "-b", newPod(), claim2)

	// Known-buggy behavior (#138765): the gang is satisfiable (one half per pod),
	// but it never schedules because the scheduler picks DRA devices one pod at a
	// time and does not backtrack. One pod greedily takes the "whole" device; the
	// other then cannot allocate its claim, so the gang stays below minCount.
	//
	// Pin down that failure mode rather than just "did not schedule": both pods
	// must report PodScheduled=False/Unschedulable, and — crucially — the failure
	// must be the DRA allocation conflict, identified by the DRA-specific
	// "cannot allocate all claims" reason appearing on at least one gang pod,
	// alongside the gang's "minCount (2) cannot be satisfied" message. Asserting
	// the DRA reason is what distinguishes this from an unrelated PodGroup
	// regression where the gang fails to schedule for some non-DRA reason.
	const draConflictMsg = "cannot allocate all claims"
	const gangUnsatisfiedMsg = "minCount (2) cannot be satisfied"

	podScheduledCondition := func(podName string) func(tCtx ktesting.TContext) (*v1.PodCondition, error) {
		return func(tCtx ktesting.TContext) (*v1.PodCondition, error) {
			p, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, podName, metav1.GetOptions{})
			if err != nil {
				return nil, err
			}
			for i := range p.Status.Conditions {
				if p.Status.Conditions[i].Type == v1.PodScheduled {
					return &p.Status.Conditions[i], nil
				}
			}
			return nil, nil
		}
	}

	// Each gang pod must reach PodScheduled=False with Reason=Unschedulable.
	for _, podName := range []string{pod1.Name, pod2.Name} {
		tCtx.Eventually(podScheduledCondition(podName)).WithTimeout(schedulingTimeout).WithPolling(time.Second).Should(
			gomega.And(
				gomega.HaveField("Status", gomega.Equal(v1.ConditionFalse)),
				gomega.HaveField("Reason", gomega.Equal("Unschedulable")),
			),
			"gang pod %s should be reported Unschedulable", podName,
		)
	}

	// The combined failure messages across the two gang pods must show both the
	// DRA allocation conflict and the unsatisfied gang minCount. (The greedy pod
	// reports the gang message; the starved pod reports the DRA message.)
	combinedMessages := func(tCtx ktesting.TContext) (string, error) {
		var combined string
		for _, podName := range []string{pod1.Name, pod2.Name} {
			c, err := podScheduledCondition(podName)(tCtx)
			if err != nil {
				return "", err
			}
			if c != nil {
				combined += c.Message + "\n"
			}
		}
		return combined, nil
	}
	tCtx.Eventually(combinedMessages).WithTimeout(schedulingTimeout).WithPolling(time.Second).Should(
		gomega.And(
			gomega.ContainSubstring(draConflictMsg),
			gomega.ContainSubstring(gangUnsatisfiedMsg),
		),
		"gang failure should be the DRA allocation conflict (%q) under an unsatisfied gang (%q), reproducing #138765",
		draConflictMsg, gangUnsatisfiedMsg,
	)

	// And the failure must persist: the gang must not eventually schedule itself
	// out of the conflict.
	for _, podName := range []string{pod1.Name, pod2.Name} {
		tCtx.Consistently(podScheduledCondition(podName)).WithTimeout(15*time.Second).WithPolling(time.Second).ShouldNot(
			gomega.HaveField("Status", gomega.Equal(v1.ConditionTrue)),
			"gang pod %s should stay unscheduled while #138765 is unfixed", podName,
		)
	}

	// Confirm the on-the-wire allocation state matches the conflict: at most one
	// of the two claims may be allocated, and any allocation must be the greedily
	// chosen "whole" device. The satisfiable outcome (each claim on a different
	// half) must never occur today, so this fails loudly if the bug is fixed.
	allocatedDevices := map[string]string{} // claim name -> device
	for _, claimName := range []string{claim1.Name, claim2.Name} {
		c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claimName, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get claim "+claimName)
		if c.Status.Allocation != nil {
			tCtx.Expect(c.Status.Allocation.Devices.Results).To(gomega.HaveLen(1), "claim %s allocation should have one device", claimName)
			allocatedDevices[claimName] = c.Status.Allocation.Devices.Results[0].Device
		}
	}
	tCtx.Expect(allocatedDevices).NotTo(
		gomega.HaveLen(2),
		"both claims allocated (%v) means the gang scheduled — #138765 appears fixed; update this test to assert the success path (see TODO below)",
		allocatedDevices,
	)
	for claimName, device := range allocatedDevices {
		tCtx.Expect(device).To(
			gomega.Equal("whole"),
			"the partially-allocated claim %s should hold the greedily-chosen whole device, reproducing #138765", claimName,
		)
	}

	// TODO(#138765): once gang scheduling reconsiders DRA allocations, replace
	// the assertions above with:
	//   waitForPodScheduled(tCtx, namespace, pod1.Name)
	//   waitForPodScheduled(tCtx, namespace, pod2.Name)
	// and assert claim1/claim2 are allocated to "half-1" and "half-2" in either
	// order (and never both to "whole").
}
