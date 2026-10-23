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
	"strings"
	"time"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/ktesting"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// preemptionQuietPeriod is the observation window for Consistently checks.
const preemptionQuietPeriod = 10 * time.Second

const (
	reasonCannotAllocate = "cannot allocate all claims"
	reasonNotEligible    = "not eligible while ResourceClaims of preempted pods on the nominated node are being deallocated"
)

// testPreemption exercises DRA preemption scenarios (KEP-5690) without a
// kubelet or ResourceClaim controller, so each subtest controls pod termination
// and claim deallocation explicitly. Subtests use dedicated nodes so settling
// victim claims left behind by one subtest never affect another.
func testPreemption(tCtx ktesting.TContext) {
	tCtx.Run("Basic", testPreemptionBasic)
	tCtx.Run("NoSecondPreemption", testPreemptionNoSecondPreemption)
	tCtx.Run("HoldForPreemptor", testPreemptionHoldForPreemptor)
	tCtx.Run("NominationCleared", testPreemptionNominationCleared)
	tCtx.Run("ScheduledElsewhere", testPreemptionScheduledElsewhere)
	tCtx.Run("HigherPriorityTakesHeldDevice", testPreemptionHigherPriorityTakesHeldDevice)
	tCtx.Run("VictimInPreBind", testPreemptionVictimInPreBind)
	tCtx.Run("ConsumableCapacity", testPreemptionConsumableCapacity)
	tCtx.Run("PassOneReuse", testPreemptionPassOneReuse)
}

type preemptionTest struct {
	namespace  string
	class      *resourceapi.DeviceClass
	driverName string
	priorities map[string]string
}

func newPreemptionTest(tCtx ktesting.TContext) *preemptionTest {
	tCtx.Helper()
	tCtx.Parallel()
	return newSerialPreemptionTest(tCtx)
}

// newSerialPreemptionTest omits tCtx.Parallel() for subtests that assert a
// specific wakeup event (such as EventResourceClaimHoldReleased), since
// concurrent ResourceClaim updates from sibling subtests would also wake
// waiting pods.
func newSerialPreemptionTest(tCtx ktesting.TContext) *preemptionTest {
	tCtx.Helper()
	pt := &preemptionTest{
		namespace:  createTestNamespace(tCtx, nil),
		priorities: make(map[string]string),
	}
	pt.class, pt.driverName = createTestClass(tCtx, pt.namespace)
	for name, value := range map[string]int32{"low": 10, "medium": 50, "high": 100, "highest": 200} {
		priorityClass := &schedulingv1.PriorityClass{
			ObjectMeta: metav1.ObjectMeta{Name: pt.namespace + "-" + name},
			Value:      value,
		}
		_, err := tCtx.Client().SchedulingV1().PriorityClasses().Create(tCtx, priorityClass, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create priority class")
		tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
			tCtx.ExpectNoError(tCtx.Client().SchedulingV1().PriorityClasses().Delete(tCtx, priorityClass.Name, metav1.DeleteOptions{}))
		})
		pt.priorities[name] = priorityClass.Name
	}
	startScheduler(tCtx)
	return pt
}

func (pt *preemptionTest) devices(tCtx ktesting.TContext, nodeName string, devices ...string) {
	tCtx.Helper()
	createSlice(tCtx, st.MakeResourceSlice(nodeName, pt.driverName).Devices(devices...).Obj())
}

// node creates a dedicated test node (with optional taints) deleted on cleanup,
// since the shared worker-* nodes are already claimed by other subtests.
func (pt *preemptionTest) node(tCtx ktesting.TContext, suffix string, taints ...v1.Taint) string {
	tCtx.Helper()
	node := createNode(tCtx, pt.namespace+suffix)
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(tCtx.Client().CoreV1().Nodes().Delete(tCtx, node.Name, metav1.DeleteOptions{}), "delete node")
	})
	if len(taints) > 0 {
		setTaints(tCtx, node.Name, taints...)
	}
	return node.Name
}

func setTaints(tCtx ktesting.TContext, nodeName string, taints ...v1.Taint) {
	tCtx.Helper()
	node, err := tCtx.Client().CoreV1().Nodes().Get(tCtx, nodeName, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get node")
	node.Spec.Taints = taints
	_, err = tCtx.Client().CoreV1().Nodes().Update(tCtx, node, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "set taints of node")
}

func (pt *preemptionTest) pod(tCtx ktesting.TContext, name, priority string, numDevices int) (*v1.Pod, *resourceapi.ResourceClaim) {
	tCtx.Helper()
	claim := st.MakeResourceClaim().Name(name).Namespace(pt.namespace)
	for range numDevices {
		claim = claim.Request(pt.class.Name)
	}
	return pt.podWithClaim(tCtx, name, priority, claim.Obj())
}

func (pt *preemptionTest) podWithMemory(tCtx ktesting.TContext, name, priority, memory string) (*v1.Pod, *resourceapi.ResourceClaim) {
	tCtx.Helper()
	claim := st.MakeResourceClaim().Name(name).Namespace(pt.namespace).Request(pt.class.Name).Obj()
	claim.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
		Requests: map[resourceapi.QualifiedName]resource.Quantity{"memory": resource.MustParse(memory)},
	}
	return pt.podWithClaim(tCtx, name, priority, claim)
}

func (pt *preemptionTest) podWithClaim(tCtx ktesting.TContext, name, priority string, claim *resourceapi.ResourceClaim) (*v1.Pod, *resourceapi.ResourceClaim) {
	tCtx.Helper()
	createdClaim := createClaim(tCtx, pt.namespace, "", pt.class, claim)
	pod := st.MakePod().Name(name).Namespace(pt.namespace).Container("c").Obj()
	pod.Spec.PriorityClassName = pt.priorities[priority]
	return createPod(tCtx, pt.namespace, "", pod, createdClaim), createdClaim
}

func getPod(tCtx ktesting.TContext, pod *v1.Pod) (*v1.Pod, error) {
	return tCtx.Client().CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
}

func waitForPreemption(tCtx ktesting.TContext, victim *v1.Pod) {
	tCtx.Helper()
	tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) { return getPod(tCtx, victim) }).
		WithTimeout(schedulingTimeout).
		Should(gomega.HaveField("DeletionTimestamp", gomega.Not(gomega.BeNil())), "Pod %s should have been preempted.", victim.Name)
}

func waitForNomination(tCtx ktesting.TContext, pod *v1.Pod, nodeName string) {
	tCtx.Helper()
	tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) { return getPod(tCtx, pod) }).
		WithTimeout(schedulingTimeout).
		Should(gomega.HaveField("Status.NominatedNodeName", gomega.Equal(nodeName)), "Pod %s should have been nominated.", pod.Name)
}

// waitForUnschedulable waits for at least one failed scheduling attempt
// matching reason before running a Consistently negative check, avoiding false
// passes before the scheduler has inspected pod.
func waitForUnschedulable(tCtx ktesting.TContext, pod *v1.Pod, reason string) {
	tCtx.Helper()
	tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) { return getPod(tCtx, pod) }).
		WithTimeout(schedulingTimeout).
		Should(gomega.HaveField("Status.Conditions", gomega.ContainElement(gomega.And(
			gomega.HaveField("Type", gomega.Equal(v1.PodScheduled)),
			gomega.HaveField("Status", gomega.Equal(v1.ConditionFalse)),
			gomega.HaveField("Reason", gomega.Equal(v1.PodReasonUnschedulable)),
			gomega.HaveField("Message", gomega.ContainSubstring(reason)),
		))), "Scheduling of pod %s should have failed with %q.", pod.Name, reason)
}

func expectNotPreempted(tCtx ktesting.TContext, preemptor *v1.Pod, reason string, pods ...*v1.Pod) {
	tCtx.Helper()
	waitForUnschedulable(tCtx, preemptor, reason)
	tCtx.Consistently(func(tCtx ktesting.TContext) ([]*v1.Pod, error) {
		var current []*v1.Pod
		for _, pod := range pods {
			pod, err := getPod(tCtx, pod)
			if err != nil {
				return nil, err
			}
			current = append(current, pod)
		}
		return current, nil
	}).WithTimeout(preemptionQuietPeriod).
		Should(gomega.HaveEach(gomega.HaveField("DeletionTimestamp", gomega.BeNil())), "Pods should not get preempted.")
}

func expectNotScheduled(tCtx ktesting.TContext, pod *v1.Pod, reason string) {
	tCtx.Helper()
	waitForUnschedulable(tCtx, pod, reason)
	tCtx.Consistently(func(tCtx ktesting.TContext) (*v1.Pod, error) { return getPod(tCtx, pod) }).
		WithTimeout(preemptionQuietPeriod).
		Should(gomega.HaveField("Spec.NodeName", gomega.BeEmpty()), "Pod %s should not get scheduled.", pod.Name)
}

// removePod force-deletes pod to simulate kubelet pod termination.
func removePod(tCtx ktesting.TContext, pod *v1.Pod) {
	tCtx.Helper()
	err := tCtx.Client().CoreV1().Pods(pod.Namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: new(int64(0))})
	if !apierrors.IsNotFound(err) {
		tCtx.ExpectNoError(err, "force-delete pod")
	}
	waitForNotFound(tCtx, tCtx.Client().CoreV1().Pods(pod.Namespace).Get, pod.Name)
}

// deallocate clears claim's allocation status to simulate the ResourceClaim controller.
func deallocate(tCtx ktesting.TContext, claim *resourceapi.ResourceClaim) {
	tCtx.Helper()
	claim, err := tCtx.Client().ResourceV1().ResourceClaims(claim.Namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get claim")
	claim.Status.ReservedFor = nil
	claim.Status.Allocation = nil
	claim.Status.Devices = nil
	_, err = tCtx.Client().ResourceV1().ResourceClaims(claim.Namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "deallocate claim")
}

func allocatedDevices(tCtx ktesting.TContext, claim *resourceapi.ResourceClaim) []string {
	tCtx.Helper()
	claim, err := tCtx.Client().ResourceV1().ResourceClaims(claim.Namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get claim")
	var devices []string
	if claim.Status.Allocation != nil {
		for _, result := range claim.Status.Allocation.Devices.Results {
			devices = append(devices, result.Device)
		}
	}
	return devices
}

// testPreemptionBasic verifies end-to-end preemption: a high-priority pod
// preempts a low-priority pod holding the sole device, waits while the victim
// claim is still allocated, and binds once the claim is deallocated.
func testPreemptionBasic(tCtx ktesting.TContext) {
	pt := newPreemptionTest(tCtx)
	node := "worker-1"
	pt.devices(tCtx, node, "device-1")
	victim, victimClaim := pt.pod(tCtx, "victim", "low", 1)
	waitForPodScheduled(tCtx, pt.namespace, victim.Name)

	preemptor, _ := pt.pod(tCtx, "preemptor", "high", 1)
	waitForPreemption(tCtx, victim)
	waitForNomination(tCtx, preemptor, node)

	removePod(tCtx, victim)
	expectNotScheduled(tCtx, preemptor, reasonNotEligible)
	deallocate(tCtx, victimClaim)
	scheduled := waitForPodScheduled(tCtx, pt.namespace, preemptor.Name)
	tCtx.Expect(scheduled.Spec.NodeName).To(gomega.Equal(node))
}

// testPreemptionNoSecondPreemption verifies that a nominated preemptor does not
// preempt a pod on another node while its first victim's claim is still settling.
func testPreemptionNoSecondPreemption(tCtx ktesting.TContext) {
	pt := newPreemptionTest(tCtx)
	pt.devices(tCtx, "worker-2", "device-1")
	pt.devices(tCtx, "worker-3", "device-2")
	victim1, victimClaim1 := pt.pod(tCtx, "victim-1", "low", 1)
	victim2, victimClaim2 := pt.pod(tCtx, "victim-2", "low", 1)
	waitForPodScheduled(tCtx, pt.namespace, victim1.Name)
	waitForPodScheduled(tCtx, pt.namespace, victim2.Name)

	preemptor, _ := pt.pod(tCtx, "preemptor", "high", 1)
	var victim, other *v1.Pod
	var victimClaim *resourceapi.ResourceClaim
	tCtx.Eventually(func(tCtx ktesting.TContext) (string, error) {
		for _, pod := range []*v1.Pod{victim1, victim2} {
			current, err := getPod(tCtx, pod)
			if err != nil {
				return "", err
			}
			if current.DeletionTimestamp != nil {
				return current.Name, nil
			}
		}
		return "", nil
	}).WithTimeout(schedulingTimeout).Should(gomega.Not(gomega.BeEmpty()), "One pod should have been preempted.")
	if current, _ := getPod(tCtx, victim1); current.DeletionTimestamp != nil {
		victim, victimClaim, other = victim1, victimClaim1, victim2
	} else {
		victim, victimClaim, other = victim2, victimClaim2, victim1
	}
	nominatedNode, err := getPod(tCtx, victim)
	tCtx.ExpectNoError(err)
	waitForNomination(tCtx, preemptor, nominatedNode.Spec.NodeName)

	// Deleting victim wakes preemptor via AssignedPod/Delete; PodEligibleToPreempt
	// must keep it from preempting other while victimClaim is still allocated.
	removePod(tCtx, victim)
	expectNotPreempted(tCtx, preemptor, reasonNotEligible, other)

	deallocate(tCtx, victimClaim)
	scheduled := waitForPodScheduled(tCtx, pt.namespace, preemptor.Name)
	tCtx.Expect(scheduled.Spec.NodeName).To(gomega.Equal(nominatedNode.Spec.NodeName))
}

// setupHold has a two-device preemptor preempt two single-device victims on
// node and deallocates only the first victim's claim, leaving device-1 free in
// the cluster but held in memory for preemptor while victimClaim2 stays allocated.
func setupHold(tCtx ktesting.TContext, pt *preemptionTest, node string) (preemptor *v1.Pod, victimClaim2 *resourceapi.ResourceClaim) {
	tCtx.Helper()
	pt.devices(tCtx, node, "device-1", "device-2")
	// Schedule sequentially so victim-1 deterministically gets device-1.
	victim1, victimClaim1 := pt.pod(tCtx, "victim-1", "low", 1)
	waitForPodScheduled(tCtx, pt.namespace, victim1.Name)
	tCtx.Expect(allocatedDevices(tCtx, victimClaim1)).To(gomega.Equal([]string{"device-1"}))
	victim2, victimClaim2 := pt.pod(tCtx, "victim-2", "low", 1)
	waitForPodScheduled(tCtx, pt.namespace, victim2.Name)

	preemptor, _ = pt.pod(tCtx, "preemptor", "high", 2)
	waitForPreemption(tCtx, victim1)
	waitForPreemption(tCtx, victim2)
	waitForNomination(tCtx, preemptor, node)
	removePod(tCtx, victim1)
	removePod(tCtx, victim2)
	deallocate(tCtx, victimClaim1)
	return preemptor, victimClaim2
}

// testPreemptionHoldForPreemptor verifies that pods with priority <= preemptor
// cannot take a held device until preemptor is deleted, and that
// EventResourceClaimHoldReleased wakes them once the hold drops.
func testPreemptionHoldForPreemptor(tCtx ktesting.TContext) {
	pt := newSerialPreemptionTest(tCtx)
	preemptor, _ := setupHold(tCtx, pt, "worker-4")

	equal, equalClaim := pt.pod(tCtx, "equal", "high", 1)
	lower, _ := pt.pod(tCtx, "lower", "medium", 1)
	expectNotScheduled(tCtx, equal, reasonCannotAllocate)
	expectNotScheduled(tCtx, lower, reasonCannotAllocate)
	// Remove lower before releasing the hold; otherwise lower might bind
	// device-1 first and get preempted by equal, stalling without a kubelet.
	removePod(tCtx, lower)

	tCtx.ExpectNoError(tCtx.Client().CoreV1().Pods(pt.namespace).Delete(tCtx, preemptor.Name, metav1.DeleteOptions{}))
	waitForPodScheduled(tCtx, pt.namespace, equal.Name)
	tCtx.Expect(allocatedDevices(tCtx, equalClaim)).To(gomega.Equal([]string{"device-1"}))
}

// testPreemptionNominationCleared verifies that clearing preemptor's
// NominatedNodeName releases its device hold and wakes waiting pods.
func testPreemptionNominationCleared(tCtx ktesting.TContext) {
	pt := newSerialPreemptionTest(tCtx)
	preemptor, _ := setupHold(tCtx, pt, "worker-7")

	equal, equalClaim := pt.pod(tCtx, "equal", "high", 1)
	expectNotScheduled(tCtx, equal, reasonCannotAllocate)

	// Patch status rather than UpdateStatus to avoid conflicting with concurrent scheduler status writes.
	_, err := tCtx.Client().CoreV1().Pods(pt.namespace).Patch(tCtx, preemptor.Name, types.MergePatchType, []byte(`{"status":{"nominatedNodeName":null}}`), metav1.PatchOptions{}, "status")
	tCtx.ExpectNoError(err, "clear nominated node name")
	waitForPodScheduled(tCtx, pt.namespace, equal.Name)
	tCtx.Expect(allocatedDevices(tCtx, equalClaim)).To(gomega.Equal([]string{"device-1"}))
}

// testPreemptionScheduledElsewhere verifies that when a nominated preemptor
// binds on another node, releasing its hold wakes a pod waiting on the original
// node. Giving lower a toleration for otherNode's taint ensures removing that
// taint wakes only preemptor, isolating EventResourceClaimHoldReleased as
// lower's sole wakeup trigger.
func testPreemptionScheduledElsewhere(tCtx ktesting.TContext) {
	pt := newSerialPreemptionTest(tCtx)
	holdNode := pt.node(tCtx, "-hold")
	preemptor, _ := setupHold(tCtx, pt, holdNode)

	taint := v1.Taint{Key: "example.com/" + pt.namespace, Effect: v1.TaintEffectNoSchedule}
	otherNode := pt.node(tCtx, "-other", taint)
	pt.devices(tCtx, otherNode, "device-3", "device-4")

	lowerClaim := createClaim(tCtx, pt.namespace, "", pt.class, st.MakeResourceClaim().Name("lower").Namespace(pt.namespace).Request(pt.class.Name).Obj())
	lower := st.MakePod().Name("lower").Namespace(pt.namespace).Container("c").
		NodeSelector(map[string]string{"kubernetes.io/hostname": holdNode}).Toleration(taint.Key).Obj()
	lower.Spec.PriorityClassName = pt.priorities["medium"]
	lower = createPod(tCtx, pt.namespace, "", lower, lowerClaim)
	expectNotScheduled(tCtx, lower, reasonCannotAllocate)

	setTaints(tCtx, otherNode)
	scheduled := waitForPodScheduled(tCtx, pt.namespace, preemptor.Name)
	tCtx.Expect(scheduled.Spec.NodeName).To(gomega.Equal(otherNode))

	scheduled = waitForPodScheduled(tCtx, pt.namespace, lower.Name)
	tCtx.Expect(scheduled.Spec.NodeName).To(gomega.Equal(holdNode))
	tCtx.Expect(allocatedDevices(tCtx, lowerClaim)).To(gomega.Equal([]string{"device-1"}))
}

// testPreemptionHigherPriorityTakesHeldDevice verifies that a pod with higher
// priority than the nominated preemptor can take a held device, forcing the
// preemptor to preempt on another node once its remaining victim claim settles.
func testPreemptionHigherPriorityTakesHeldDevice(tCtx ktesting.TContext) {
	pt := newPreemptionTest(tCtx)
	// Medium-priority victim on worker-6 is initially less attractive to
	// preempt than the low-priority victims in setupHold on worker-5.
	pt.devices(tCtx, "worker-6", "device-3", "device-4")
	another, _ := pt.pod(tCtx, "another", "medium", 2)
	waitForPodScheduled(tCtx, pt.namespace, another.Name)
	preemptor, victimClaim2 := setupHold(tCtx, pt, "worker-5")

	higher, higherClaim := pt.pod(tCtx, "higher", "highest", 1)
	waitForPodScheduled(tCtx, pt.namespace, higher.Name)
	tCtx.Expect(allocatedDevices(tCtx, higherClaim)).To(gomega.Equal([]string{"device-1"}))

	// Deallocating victimClaim2 unblocks PodEligibleToPreempt on worker-5; with
	// device-1 gone, preemptor no longer fits there and preempts on worker-6.
	deallocate(tCtx, victimClaim2)
	waitForPreemption(tCtx, another)
	waitForNomination(tCtx, preemptor, "worker-6")
}

// testPreemptionVictimInPreBind verifies preemption of a pod blocked on device
// binding conditions in PreBind: Unreserve leaves its claim allocated with an
// empty ReservedFor, and the re-queued victim's own PostFilter deallocates it.
func testPreemptionVictimInPreBind(tCtx ktesting.TContext) {
	pt := newPreemptionTest(tCtx)
	createSlice(tCtx, &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("%s-%s", nodeName, pt.driverName)},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: new(nodeName),
			Pool:     resourceapi.ResourcePool{Name: nodeName, ResourceSliceCount: 1},
			Driver:   pt.driverName,
			Devices: []resourceapi.Device{{
				Name:                     "device-1",
				BindingConditions:        []string{bindingCondition},
				BindingFailureConditions: []string{failureCondition},
			}},
		},
	})
	victim, victimClaim := pt.pod(tCtx, "victim", "low", 1)
	tCtx.Eventually(func(tCtx ktesting.TContext) ([]string, error) {
		claim, err := tCtx.Client().ResourceV1().ResourceClaims(victimClaim.Namespace).Get(tCtx, victimClaim.Name, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		var consumers []string
		for _, consumer := range claim.Status.ReservedFor {
			consumers = append(consumers, consumer.Name)
		}
		return consumers, nil
	}).WithTimeout(schedulingTimeout).Should(gomega.Equal([]string{victim.Name}), "Claim of pod in PreBind should be reserved.")

	_, preemptorClaim := pt.pod(tCtx, "preemptor", "high", 1)
	tCtx.Eventually(func(tCtx ktesting.TContext) ([]string, error) {
		return allocatedDevices(tCtx, preemptorClaim), nil
	}).WithTimeout(schedulingTimeout).Should(gomega.Equal([]string{"device-1"}), "Claim of the preemptor should get the device.")
	current, err := getPod(tCtx, victim)
	tCtx.ExpectNoError(err)
	tCtx.Expect(current.Spec.NodeName).To(gomega.BeEmpty(), "preempted pod should not be bound")
	tCtx.Expect(strings.Join(allocatedDevices(tCtx, victimClaim), ",")).To(gomega.BeEmpty(), "claim of the preempted pod should be deallocated")
}

// testPreemptionConsumableCapacity exercises the 80Gi shared-device example
// from KEP-5690: preemptor needs 70Gi and preempts two 30Gi victims; after one
// victim claim is deallocated, 40Gi of the 50Gi free capacity is held on top of
// the 30Gi still allocated to victim-2, leaving 10Gi available for small pods.
func testPreemptionConsumableCapacity(tCtx ktesting.TContext) {
	pt := newPreemptionTest(tCtx)
	node := pt.node(tCtx, "")
	slice := st.MakeResourceSlice(node, pt.driverName).
		Device("device-1", map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{"memory": {Value: resource.MustParse("80Gi")}}).
		Obj()
	slice.Spec.Devices[0].AllowMultipleAllocations = new(true)
	createSlice(tCtx, slice)
	victim1, victimClaim1 := pt.podWithMemory(tCtx, "victim-1", "low", "30Gi")
	victim2, victimClaim2 := pt.podWithMemory(tCtx, "victim-2", "low", "30Gi")
	waitForPodScheduled(tCtx, pt.namespace, victim1.Name)
	waitForPodScheduled(tCtx, pt.namespace, victim2.Name)

	preemptor, _ := pt.podWithMemory(tCtx, "preemptor", "high", "70Gi")
	waitForPreemption(tCtx, victim1)
	waitForPreemption(tCtx, victim2)
	waitForNomination(tCtx, preemptor, node)
	removePod(tCtx, victim1)
	removePod(tCtx, victim2)
	deallocate(tCtx, victimClaim1)

	// 70Gi held (offset by 30Gi still allocated to victim-2) leaves 10Gi free,
	// so a 40Gi pod is rejected while a 10Gi pod still fits.
	lower, _ := pt.podWithMemory(tCtx, "lower", "medium", "40Gi")
	expectNotScheduled(tCtx, lower, reasonCannotAllocate)
	small, _ := pt.podWithMemory(tCtx, "small", "medium", "10Gi")
	scheduled := waitForPodScheduled(tCtx, pt.namespace, small.Name)
	tCtx.Expect(scheduled.Spec.NodeName).To(gomega.Equal(node))

	deallocate(tCtx, victimClaim2)
	scheduled = waitForPodScheduled(tCtx, pt.namespace, preemptor.Name)
	tCtx.Expect(scheduled.Spec.NodeName).To(gomega.Equal(node))
}

// testPreemptionPassOneReuse verifies that RunFilterPluginsWithNominatedPods
// reuses pass 1's device choice in pass 2 rather than re-running the allocator
// without holds (which would pick free, held device-1 ahead of device-4).
func testPreemptionPassOneReuse(tCtx ktesting.TContext) {
	pt := newPreemptionTest(tCtx)
	node := pt.node(tCtx, "")
	pt.devices(tCtx, node, "device-1", "device-2", "device-3", "device-4")
	var victims []*v1.Pod
	var victimClaims []*resourceapi.ResourceClaim
	// Schedule sequentially so victim-<n> deterministically gets device-<n>.
	for i := 1; i <= 3; i++ {
		victim, victimClaim := pt.pod(tCtx, fmt.Sprintf("victim-%d", i), "low", 1)
		waitForPodScheduled(tCtx, pt.namespace, victim.Name)
		tCtx.Expect(allocatedDevices(tCtx, victimClaim)).To(gomega.Equal([]string{fmt.Sprintf("device-%d", i)}))
		victims = append(victims, victim)
		victimClaims = append(victimClaims, victimClaim)
	}
	// Higher-priority blocker occupies device-4 so preemptor holds device-1..3.
	blocker, blockerClaim := pt.pod(tCtx, "blocker", "highest", 1)
	waitForPodScheduled(tCtx, pt.namespace, blocker.Name)
	tCtx.Expect(allocatedDevices(tCtx, blockerClaim)).To(gomega.Equal([]string{"device-4"}))

	preemptor, _ := pt.pod(tCtx, "preemptor", "high", 3)
	for _, victim := range victims {
		waitForPreemption(tCtx, victim)
	}
	waitForNomination(tCtx, preemptor, node)
	for _, victim := range victims {
		removePod(tCtx, victim)
	}
	deallocate(tCtx, victimClaims[0])

	// Freeing device-4 leaves two free devices (held device-1 and unheld
	// device-4), still short of the three preemptor needs while victim-2 and
	// victim-3 are settling.
	removePod(tCtx, blocker)
	deallocate(tCtx, blockerClaim)

	lower, lowerClaim := pt.pod(tCtx, "lower", "medium", 1)
	waitForPodScheduled(tCtx, pt.namespace, lower.Name)
	tCtx.Expect(allocatedDevices(tCtx, lowerClaim)).To(gomega.Equal([]string{"device-4"}))
}
