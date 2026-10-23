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

// preemptionQuietPeriod is how long the tests check that something does not
// happen.
const preemptionQuietPeriod = 10 * time.Second

// testPreemption covers preemption of pods with ResourceClaims (KEP-5690).
//
// The ResourceClaim controller does not run, so the tests decide when
// preempted pods are gone (force deletion, there is no kubelet) and when their
// claims get deallocated.
//
// Each test uses its own nodes because claims of preempted pods defer
// preemption on their node, and claims which a test does not deallocate stay
// allocated.
func testPreemption(tCtx ktesting.TContext) {
	tCtx.Run("Basic", testPreemptionBasic)
	tCtx.Run("NoSecondPreemption", testPreemptionNoSecondPreemption)
	tCtx.Run("HoldForPreemptor", testPreemptionHoldForPreemptor)
	tCtx.Run("NominationCleared", testPreemptionNominationCleared)
	tCtx.Run("HigherPriorityTakesHeldDevice", testPreemptionHigherPriorityTakesHeldDevice)
	tCtx.Run("VictimInPreBind", testPreemptionVictimInPreBind)
	tCtx.Run("ConsumableCapacity", testPreemptionConsumableCapacity)
}

type preemptionTest struct {
	namespace  string
	class      *resourceapi.DeviceClass
	driverName string
	priorities map[string]string
}

// newPreemptionTest creates a namespace, a device class and priority classes
// "low" < "medium" < "high" < "highest" for the test.
func newPreemptionTest(tCtx ktesting.TContext) *preemptionTest {
	tCtx.Helper()
	tCtx.Parallel()
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

// devices publishes devices on the node.
func (pt *preemptionTest) devices(tCtx ktesting.TContext, nodeName string, devices ...string) {
	tCtx.Helper()
	createSlice(tCtx, st.MakeResourceSlice(nodeName, pt.driverName).Devices(devices...).Obj())
}

// pod creates a pod with the priority and a claim which requests numDevices devices.
func (pt *preemptionTest) pod(tCtx ktesting.TContext, name, priority string, numDevices int) (*v1.Pod, *resourceapi.ResourceClaim) {
	tCtx.Helper()
	claim := st.MakeResourceClaim().Name(name).Namespace(pt.namespace)
	for range numDevices {
		claim = claim.Request(pt.class.Name)
	}
	return pt.podWithClaim(tCtx, name, priority, claim.Obj())
}

// podWithMemory creates a pod with the priority and a claim which requests
// the amount of memory from one device.
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

func expectNotPreempted(tCtx ktesting.TContext, pods ...*v1.Pod) {
	tCtx.Helper()
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

func expectNotScheduled(tCtx ktesting.TContext, pod *v1.Pod) {
	tCtx.Helper()
	tCtx.Consistently(func(tCtx ktesting.TContext) (*v1.Pod, error) { return getPod(tCtx, pod) }).
		WithTimeout(preemptionQuietPeriod).
		Should(gomega.HaveField("Spec.NodeName", gomega.BeEmpty()), "Pod %s should not get scheduled.", pod.Name)
}

// removePod simulates that the kubelet has stopped a preempted pod.
func removePod(tCtx ktesting.TContext, pod *v1.Pod) {
	tCtx.Helper()
	err := tCtx.Client().CoreV1().Pods(pod.Namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: new(int64(0))})
	if !apierrors.IsNotFound(err) {
		tCtx.ExpectNoError(err, "force-delete pod")
	}
	waitForNotFound(tCtx, tCtx.Client().CoreV1().Pods(pod.Namespace).Get, pod.Name)
}

// deallocate simulates the ResourceClaim controller after the consumers of
// the claim are gone.
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

// testPreemptionBasic checks the whole flow: a pod preempts another pod which
// uses the only device, and gets scheduled once the claim of the preempted
// pod is deallocated.
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
	expectNotScheduled(tCtx, preemptor)
	deallocate(tCtx, victimClaim)
	scheduled := waitForPodScheduled(tCtx, pt.namespace, preemptor.Name)
	tCtx.Expect(scheduled.Spec.NodeName).To(gomega.Equal(node))
}

// testPreemptionNoSecondPreemption checks that the preemptor does not preempt
// more pods while the claims of its victims are not deallocated yet.
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

	// The deletion of the victim triggers another scheduling attempt,
	// which must not preempt the other pod.
	removePod(tCtx, victim)
	expectNotPreempted(tCtx, other)
	tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) { return getPod(tCtx, preemptor) }).
		WithTimeout(schedulingTimeout).
		Should(gomega.HaveField("Status.Conditions", gomega.ContainElement(gomega.HaveField("Message",
			gomega.ContainSubstring("not eligible while ResourceClaims of preempted pods on the nominated node are being deallocated")))))

	deallocate(tCtx, victimClaim)
	scheduled := waitForPodScheduled(tCtx, pt.namespace, preemptor.Name)
	tCtx.Expect(scheduled.Spec.NodeName).To(gomega.Equal(nominatedNode.Spec.NodeName))
}

// setupHold lets a preemptor which needs two devices preempt two pods and
// waits until one of the two victim claims is deallocated, so that one of
// the devices is free and held for the preemptor.
func setupHold(tCtx ktesting.TContext, pt *preemptionTest, node string) (preemptor *v1.Pod, victimClaim2 *resourceapi.ResourceClaim) {
	tCtx.Helper()
	pt.devices(tCtx, node, "device-1", "device-2")
	// One after the other, so that the device which gets freed is device-1.
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

// testPreemptionHoldForPreemptor checks that pods with lower or equal priority
// don't get the device which preemption freed before the preemptor gets
// scheduled, and that they get it when the preemptor goes away.
func testPreemptionHoldForPreemptor(tCtx ktesting.TContext) {
	pt := newPreemptionTest(tCtx)
	preemptor, _ := setupHold(tCtx, pt, "worker-4")

	equal, equalClaim := pt.pod(tCtx, "equal", "high", 1)
	lower, _ := pt.pod(tCtx, "lower", "medium", 1)
	expectNotScheduled(tCtx, equal)
	expectNotScheduled(tCtx, lower)
	// Otherwise it might get the device first once it is not held anymore
	// and then get preempted by equal, which never completes without a
	// kubelet.
	removePod(tCtx, lower)

	// Releasing the hold makes the device available.
	tCtx.ExpectNoError(tCtx.Client().CoreV1().Pods(pt.namespace).Delete(tCtx, preemptor.Name, metav1.DeleteOptions{}))
	waitForPodScheduled(tCtx, pt.namespace, equal.Name)
	tCtx.Expect(allocatedDevices(tCtx, equalClaim)).To(gomega.Equal([]string{"device-1"}))
}

// testPreemptionNominationCleared checks that clearing the nominated node
// name of the preemptor releases the device held for it. The preemptor
// itself cannot use the device alone and cannot preempt again because no
// pod is left on the node.
func testPreemptionNominationCleared(tCtx ktesting.TContext) {
	pt := newPreemptionTest(tCtx)
	preemptor, _ := setupHold(tCtx, pt, "worker-7")

	equal, equalClaim := pt.pod(tCtx, "equal", "high", 1)
	expectNotScheduled(tCtx, equal)

	// A patch because the scheduler updates the status, too.
	_, err := tCtx.Client().CoreV1().Pods(pt.namespace).Patch(tCtx, preemptor.Name, types.MergePatchType, []byte(`{"status":{"nominatedNodeName":null}}`), metav1.PatchOptions{}, "status")
	tCtx.ExpectNoError(err, "clear nominated node name")
	waitForPodScheduled(tCtx, pt.namespace, equal.Name)
	tCtx.Expect(allocatedDevices(tCtx, equalClaim)).To(gomega.Equal([]string{"device-1"}))
}

// testPreemptionHigherPriorityTakesHeldDevice checks that a pod with higher
// priority than the preemptor may take a held device, and that the preemptor
// then preempts again.
func testPreemptionHigherPriorityTakesHeldDevice(tCtx ktesting.TContext) {
	pt := newPreemptionTest(tCtx)
	// Preempting this pod is less attractive than preempting the pods with
	// lower priority in setupHold.
	pt.devices(tCtx, "worker-6", "device-3", "device-4")
	another, _ := pt.pod(tCtx, "another", "medium", 2)
	waitForPodScheduled(tCtx, pt.namespace, another.Name)
	preemptor, victimClaim2 := setupHold(tCtx, pt, "worker-5")

	higher, higherClaim := pt.pod(tCtx, "higher", "highest", 1)
	waitForPodScheduled(tCtx, pt.namespace, higher.Name)
	tCtx.Expect(allocatedDevices(tCtx, higherClaim)).To(gomega.Equal([]string{"device-1"}))

	// Once the claim of the other victim is deallocated, the preemptor
	// does not fit anymore and has to look elsewhere.
	deallocate(tCtx, victimClaim2)
	waitForPreemption(tCtx, another)
	waitForNomination(tCtx, preemptor, "worker-6")
}

// testPreemptionVictimInPreBind checks preemption of a pod which waits for
// binding conditions in PreBind. Its claim is allocated already and stays
// allocated without consumer after the preemption, which the ResourceClaim
// controller does not handle. The PostFilter of the preempted pod has to
// deallocate it.
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

// testPreemptionConsumableCapacity covers a device with consumable capacity
// (the 80Gi example in the KEP): the preemptor needs capacity from two
// pods. While the claim of one of them is still allocated, the capacity of
// the other one is free but held for the preemptor.
func testPreemptionConsumableCapacity(tCtx ktesting.TContext) {
	pt := newPreemptionTest(tCtx)
	// All other nodes are used by other tests.
	node := createNode(tCtx, pt.namespace)
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(tCtx.Client().CoreV1().Nodes().Delete(tCtx, node.Name, metav1.DeleteOptions{}), "delete node")
	})
	slice := st.MakeResourceSlice(node.Name, pt.driverName).
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
	waitForNomination(tCtx, preemptor, node.Name)
	removePod(tCtx, victim1)
	removePod(tCtx, victim2)
	deallocate(tCtx, victimClaim1)

	// 50Gi are free, the preemptor needs 40Gi of them on top of the 30Gi
	// which are still allocated for victim-2.
	lower, _ := pt.podWithMemory(tCtx, "lower", "medium", "40Gi")
	expectNotScheduled(tCtx, lower)

	deallocate(tCtx, victimClaim2)
	scheduled := waitForPodScheduled(tCtx, pt.namespace, preemptor.Name)
	tCtx.Expect(scheduled.Spec.NodeName).To(gomega.Equal(node.Name))
}
