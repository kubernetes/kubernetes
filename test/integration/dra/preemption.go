/*
Copyright 2024 The Kubernetes Authors.

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

	"k8s.io/utils/ptr"

	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

func testPreemption(tCtx ktesting.TContext, enabled bool) {
	tCtx.Parallel()

	startScheduler(tCtx)
	startClaimController(tCtx)

	if !enabled {
		runSubTest(tCtx, "preemption-disabled", testPreemptionDisabled)
	} else {
		runSubTest(tCtx, "direct-claim-preemption", testDirectClaimPreemption)
		runSubTest(tCtx, "template-preemption", testTemplatePreemption)
		runSubTest(tCtx, "priority-aware-preemption", testPriorityAwarePreemption)
		runSubTest(tCtx, "podgroup-preemption", testPodGroupPreemption)
		runSubTest(tCtx, "consumable-capacity-preemption", testConsumableCapacityPreemption)
		runSubTest(tCtx, "partitionable-devices-preemption", testPartitionableDevicesPreemption)
		runSubTest(tCtx, "binding-conditions-preemption", testBindingConditionsPreemption)
	}
}

// testDirectClaimPreemption verifies that a high-priority pod can preempt a low-priority pod
// when both reference a ResourceClaim directly. It ensures that the victim pod is marked
// for deletion and receives the DisruptionTarget condition.
func testDirectClaimPreemption(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)

	class, driverName := createTestClass(tCtx, namespace)

	// Create a single device slice on worker-0
	slice := st.MakeResourceSlice("worker-0", driverName).Devices("device-1")
	createSlice(tCtx, slice.Obj())

	lowPriority := createPriorityClass(tCtx, namespace, "low-priority", 100)
	highPriority := createPriorityClass(tCtx, namespace, "high-priority", 200)

	// A low priority pod with a single claim takes the only device.
	victimClaim := createClaim(tCtx, namespace, "-victim", class, claim)
	victimPod := podWithClaimName.DeepCopy()
	victimPod.Spec.PriorityClassName = lowPriority
	victim := createPod(tCtx, namespace, "-victim", victimPod, victimClaim)

	// Wait for victim to be scheduled
	victim = waitForPodScheduled(tCtx, namespace, victim.Name)

	// Make sure the reservedFor field has been updated by the scheduler. We check this sepearately
	// from the PodScheduled check since this is a check on the claim rather than the pod.
	tCtx.Eventually(func(tCtx ktesting.TContext) int {
		c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, victimClaim.Name, metav1.GetOptions{})
		if err != nil {
			return 0
		}
		return len(c.Status.ReservedFor)
	}).WithTimeout(30*time.Second).Should(gomega.Equal(1), "victim claim should be reserved for the pod")

	// A high priority pod references a claim requesting a device from the same class.
	preemptorClaim := createClaim(tCtx, namespace, "-preemptor", class, claim)
	preemptorPod := podWithClaimName.DeepCopy()
	preemptorPod.Spec.PriorityClassName = highPriority
	_ = createPod(tCtx, namespace, "-preemptor", preemptorPod, preemptorClaim)

	// wait for victim to be marked for eviction (deletion timestamp and disruption target)
	waitForPodPreempted(tCtx, namespace, victim.Name)

	waitForClaimReleased(tCtx, namespace, victimClaim.Name)
}

// testPreemptionDisabled verifies that preemption fails for direct claims
// when the DRAPreemption feature gate is disabled.
func testPreemptionDisabled(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)

	class, driverName := createTestClass(tCtx, namespace)

	// Create a single device slice on worker-0
	slice := st.MakeResourceSlice("worker-0", driverName).Devices("device-1")
	createSlice(tCtx, slice.Obj())

	lowPriority := createPriorityClass(tCtx, namespace, "low-priority", 100)
	highPriority := createPriorityClass(tCtx, namespace, "high-priority", 200)

	// A low priority pod with a single claim takes the only device.
	victimClaim := createClaim(tCtx, namespace, "-victim", class, claim)
	victimPod := podWithClaimName.DeepCopy()
	victimPod.Spec.PriorityClassName = lowPriority
	victim := createPod(tCtx, namespace, "-victim", victimPod, victimClaim)

	// Wait for victim to be scheduled
	victim = waitForPodScheduled(tCtx, namespace, victim.Name)

	// A high priority pod references a claim requesting a device from the same class.
	preemptorClaim := createClaim(tCtx, namespace, "-preemptor", class, claim)
	preemptorPod := podWithClaimName.DeepCopy()
	preemptorPod.Spec.PriorityClassName = highPriority
	preemptor := createPod(tCtx, namespace, "-preemptor", preemptorPod, preemptorClaim)

	// Preemptor should FAIL to be scheduled because without the fix,
	// it cannot find victims holding direct claims!
	waitForPodUnschedulable(tCtx, namespace, preemptor.Name)
}

func waitForPodUnschedulable(tCtx ktesting.TContext, namespace, podName string) {
	tCtx.Helper()
	tCtx.Eventually(func(tCtx ktesting.TContext) bool {
		p, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, podName, metav1.GetOptions{})
		if err != nil {
			return false
		}
		_, cond := podutil.GetPodCondition(&p.Status, v1.PodScheduled)
		return cond != nil && cond.Status == v1.ConditionFalse && cond.Reason == v1.PodReasonUnschedulable
	}).WithTimeout(30*time.Second).Should(gomega.BeTrue(), fmt.Sprintf("Pod %s should be unschedulable", podName))
}

// testTemplatePreemption verifies that preemption works when pods use ResourceClaimTemplates.
// It ensures that the generated claims are correctly handled during preemption dry-runs
// and that the victim pod is preempted when a higher priority pod needs the resource.
func testTemplatePreemption(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)

	class, driverName := createTestClass(tCtx, namespace)

	slice := st.MakeResourceSlice("worker-0", driverName).Devices("device-1")
	createSlice(tCtx, slice.Obj())

	lowPriority := createPriorityClass(tCtx, namespace, "low-priority", 100)
	highPriority := createPriorityClass(tCtx, namespace, "high-priority", 200)

	claim := st.MakeResourceClaim().Name("my-claim").Namespace(namespace).Request(class.Name).Obj()
	template := &resourceapi.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "my-template", Namespace: namespace},
		Spec: resourceapi.ResourceClaimTemplateSpec{
			Spec: claim.Spec,
		},
	}
	_, err := tCtx.Client().ResourceV1().ResourceClaimTemplates(namespace).Create(tCtx, template, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create template")

	victimPod := podWithClaimName.DeepCopy()
	victimPod.Spec.PriorityClassName = lowPriority
	victim := createPod(tCtx, namespace, "-victim-0", victimPod, template)
	victim = waitForPodScheduled(tCtx, namespace, victim.Name)

	// Retrieve the generated claim
	claims, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list generated claims")
	if len(claims.Items) != 1 {
		tCtx.Fatalf("Expected exactly 1 generated claim, got %d", len(claims.Items))
	}
	generatedClaimName := claims.Items[0].Name

	// Make sure the reservedFor field has been updated by the scheduler.
	tCtx.Eventually(func(tCtx ktesting.TContext) int {
		c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, generatedClaimName, metav1.GetOptions{})
		if err != nil {
			return 0
		}
		return len(c.Status.ReservedFor)
	}).WithTimeout(30*time.Second).Should(gomega.Equal(1), "generated claim should be reserved for the pod")

	preemptorClaim := createClaim(tCtx, namespace, "-preemptor", class, claim)
	preemptorPod := podWithClaimName.DeepCopy()
	preemptorPod.Spec.PriorityClassName = highPriority
	_ = createPod(tCtx, namespace, "-preemptor", preemptorPod, preemptorClaim)

	// wait for victim to be marked for eviction (deletion timestamp and disruption target)
	waitForPodPreempted(tCtx, namespace, victim.Name)

	waitForClaimReleased(tCtx, namespace, generatedClaimName)
}

// testPodGroupPreemption verifies that the scheduler does NOT preempt pods that belong to a PodGroup
// to avoid breaking gang scheduling guarantees.
// It ensures the preemptor fails to find victims and the victim remains running.
func testPodGroupPreemption(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)

	class, driverName := createTestClass(tCtx, namespace)

	slice := st.MakeResourceSlice("worker-0", driverName).Devices("device-1")
	createSlice(tCtx, slice.Obj())

	lowPriority := createPriorityClass(tCtx, namespace, "low-priority", 100)
	highPriority := createPriorityClass(tCtx, namespace, "high-priority", 200)

	victimClaim := createClaim(tCtx, namespace, "-victim", class, claim)

	podGroup := &schedulingv1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-pod-group",
			Namespace: namespace,
		},
		Spec: schedulingv1alpha3.PodGroupSpec{
			SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
				Basic: &schedulingv1alpha3.BasicSchedulingPolicy{},
			},
			ResourceClaims: []schedulingv1alpha3.PodGroupResourceClaim{
				{
					Name:              victimClaim.Name,
					ResourceClaimName: &victimClaim.Name,
				},
			},
		},
	}
	var err error
	podGroup, err = tCtx.Client().SchedulingV1alpha3().PodGroups(namespace).Create(tCtx, podGroup, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create PodGroup")

	victimPod := podWithClaimName.DeepCopy()
	victimPod.Spec.PriorityClassName = lowPriority
	victimPod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
		PodGroupName: &podGroup.Name,
	}
	victim := createPod(tCtx, namespace, "-victim", victimPod, victimClaim)

	victim = waitForPodScheduled(tCtx, namespace, victim.Name)

	// Make sure the reservedFor field has been updated by the system to point to the PodGroup.
	tCtx.Eventually(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaim, error) {
		return tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, victimClaim.Name, metav1.GetOptions{})
	}).WithTimeout(30*time.Second).Should(gomega.HaveField("Status.ReservedFor", gomega.ConsistOf(gomega.HaveField("UID", gomega.Equal(podGroup.UID)))), "victim claim should be reserved strictly for the podgroup")

	// A high priority pod wants same class
	preemptorClaim := createClaim(tCtx, namespace, "-preemptor", class, claim)
	preemptorPod := podWithClaimName.DeepCopy()
	preemptorPod.Spec.PriorityClassName = highPriority
	preemptor := createPod(tCtx, namespace, "-preemptor", preemptorPod, preemptorClaim)

	// wait for preemptor to officially fail to find preemption victims on any node
	tCtx.Eventually(func(tCtx ktesting.TContext) bool {
		p, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, preemptor.Name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		for _, cond := range p.Status.Conditions {
			if cond.Type == v1.PodScheduled && cond.Status == v1.ConditionFalse && cond.Reason == "Unschedulable" && strings.Contains(cond.Message, "No preemption victims found") {
				return true
			}
		}
		return false
	}).WithTimeout(10*time.Second).Should(gomega.BeTrue(), "Preemptor should be unschedulable")

	// Ensure the victim pod was NOT marked for eviction
	pod, _ := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, victim.Name, metav1.GetOptions{})
	if pod.DeletionTimestamp != nil {
		tCtx.Fatalf("Victim pod was incorrectly marked for disruption during PodGroup preemption check (DeletionTimestamp set)")
	}
	_, cond := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget)
	if cond != nil {
		tCtx.Fatalf("Victim pod was incorrectly marked for disruption during PodGroup preemption check (DisruptionTarget condition set)")
	}
}

// testConsumableCapacityPreemption verifies that the scheduler can perform minimal preemption
// when using consumable capacity. It schedules multiple low-priority pods sharing a device,
// and ensures that a high-priority pod only preempts ENOUGH victims to satisfy its request,
// leaving others running.
func testConsumableCapacityPreemption(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)

	class, driverName := createTestClass(tCtx, namespace)

	// Create a ResourceSlice with a single device with a capacity of 10 units.
	capacityName := resourceapi.QualifiedName("example.com/cpus")
	slice := st.MakeResourceSlice("worker-0", driverName).Device("device-1",
		map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			capacityName: {Value: resource.MustParse("10")},
		},
	).Obj()
	trueVal := true
	slice.Spec.Devices[0].AllowMultipleAllocations = &trueVal
	createSlice(tCtx, slice)

	lowPriority := createPriorityClass(tCtx, namespace, "low-priority", 100)
	highPriority := createPriorityClass(tCtx, namespace, "high-priority", 200)

	// Claim for victims requesting 4 units
	victimClaim := st.MakeResourceClaim().Name("claim").Request(class.Name).Obj()
	victimClaim.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
		Requests: map[resourceapi.QualifiedName]resource.Quantity{
			capacityName: resource.MustParse("4"),
		},
	}

	// Create Victim A
	victimClaimA := createClaim(tCtx, namespace, "-victim-a", class, victimClaim)
	victimPodA := podWithClaimName.DeepCopy()
	victimPodA.Spec.PriorityClassName = lowPriority
	victimA := createPod(tCtx, namespace, "-victim-a", victimPodA, victimClaimA)
	waitForPodScheduled(tCtx, namespace, victimA.Name)

	// Create Victim B
	victimClaimB := createClaim(tCtx, namespace, "-victim-b", class, victimClaim)
	victimPodB := podWithClaimName.DeepCopy()
	victimPodB.Spec.PriorityClassName = lowPriority
	victimB := createPod(tCtx, namespace, "-victim-b", victimPodB, victimClaimB)
	waitForPodScheduled(tCtx, namespace, victimB.Name)

	// Prototype for preemptor claim requesting 6 units
	preemptorClaim := st.MakeResourceClaim().Name("claim").Request(class.Name).Obj()
	preemptorClaim.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
		Requests: map[resourceapi.QualifiedName]resource.Quantity{
			capacityName: resource.MustParse("6"),
		},
	}

	// Create preemptor claim
	preemptorClaim = createClaim(tCtx, namespace, "-preemptor", class, preemptorClaim)
	preemptorPod := podWithClaimName.DeepCopy()
	preemptorPod.Spec.PriorityClassName = highPriority
	preemptor := createPod(tCtx, namespace, "-preemptor", preemptorPod, preemptorClaim)

	// Preemptor should trigger preemption and get scheduled
	waitForPodScheduled(tCtx, namespace, preemptor.Name)

	// Verify that exactly one of the victims is preempted
	tCtx.Eventually(func(tCtx ktesting.TContext) int {
		podA, errA := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, victimA.Name, metav1.GetOptions{})
		podB, errB := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, victimB.Name, metav1.GetOptions{})
		if errA != nil || errB != nil {
			return 0
		}

		preemptedCount := 0
		var preemptedPod *v1.Pod
		if podA.DeletionTimestamp != nil {
			preemptedCount++
			preemptedPod = podA
		}
		if podB.DeletionTimestamp != nil {
			preemptedCount++
			preemptedPod = podB
		}

		if preemptedCount != 1 {
			return 0
		}

		_, cond := podutil.GetPodCondition(&preemptedPod.Status, v1.DisruptionTarget)
		if cond != nil && cond.Status == v1.ConditionTrue {
			return 1
		}
		return 0
	}).WithTimeout(30*time.Second).Should(gomega.Equal(1), "Exactly one victim pod should be preempted with DisruptionTarget condition")

	// Figure out which victim was preempted and verify its claim is released
	podA, _ := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, victimA.Name, metav1.GetOptions{})
	var preemptedClaimName string
	if podA.DeletionTimestamp != nil {
		preemptedClaimName = victimClaimA.Name
	} else {
		preemptedClaimName = victimClaimB.Name
	}

	waitForClaimReleased(tCtx, namespace, preemptedClaimName)
}

// testPartitionableDevicesPreemption verifies that the scheduler can aggregate capacity
// freed by preempting multiple pods that use different devices sharing the same counter pool.
func testPartitionableDevicesPreemption(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)

	class, driverName := createTestClass(tCtx, namespace)

	// Create a ResourceSlice with SharedCounters (10 units)
	sliceCounters := st.MakeResourceSlice("worker-0", driverName).Obj()
	sliceCounters.Name += "-counters"
	sliceCounters.Spec.Pool.ResourceSliceCount = 2
	sliceCounters.Spec.SharedCounters = []resourceapi.CounterSet{
		{
			Name: "gpu-pool",
			Counters: map[string]resourceapi.Counter{
				"gpu": {Value: resource.MustParse("10")},
			},
		},
	}
	createSlice(tCtx, sliceCounters)

	// Create a ResourceSlice with Devices consuming from the pool
	sliceDevices := st.MakeResourceSlice("worker-0", driverName).Obj()
	sliceDevices.Name += "-devices"
	sliceDevices.Spec.Pool.ResourceSliceCount = 2
	sliceDevices.Spec.Devices = []resourceapi.Device{
		{
			Name: "device-a",
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				"name": {StringValue: ptr.To("device-a")},
			},
			ConsumesCounters: []resourceapi.DeviceCounterConsumption{
				{
					CounterSet: "gpu-pool",
					Counters: map[string]resourceapi.Counter{
						"gpu": {Value: resource.MustParse("4")},
					},
				},
			},
		},
		{
			Name: "device-b",
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				"name": {StringValue: ptr.To("device-b")},
			},
			ConsumesCounters: []resourceapi.DeviceCounterConsumption{
				{
					CounterSet: "gpu-pool",
					Counters: map[string]resourceapi.Counter{
						"gpu": {Value: resource.MustParse("4")},
					},
				},
			},
		},
		{
			Name: "device-c",
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				"name": {StringValue: ptr.To("device-c")},
			},
			ConsumesCounters: []resourceapi.DeviceCounterConsumption{
				{
					CounterSet: "gpu-pool",
					Counters: map[string]resourceapi.Counter{
						"gpu": {Value: resource.MustParse("8")},
					},
				},
			},
		},
	}
	createSlice(tCtx, sliceDevices)

	lowPriority := createPriorityClass(tCtx, namespace, "low-priority", 10)
	highPriority := createPriorityClass(tCtx, namespace, "high-priority", 100)

	// Create Victim A (claims device-a, 4 units)
	victimClaimA := createClaimWithSelector(tCtx, namespace, "claim-a", class, driverName, "device-a")
	victimPodA := podWithClaimName.DeepCopy()
	victimPodA.Spec.PriorityClassName = lowPriority
	victimA := createPod(tCtx, namespace, "-victim-a", victimPodA, victimClaimA)
	waitForPodScheduled(tCtx, namespace, victimA.Name)

	// Create Victim B (claims device-b, 4 units)
	victimClaimB := createClaimWithSelector(tCtx, namespace, "claim-b", class, driverName, "device-b")
	victimPodB := podWithClaimName.DeepCopy()
	victimPodB.Spec.PriorityClassName = lowPriority
	victimB := createPod(tCtx, namespace, "-victim-b", victimPodB, victimClaimB)
	waitForPodScheduled(tCtx, namespace, victimB.Name)

	// Create Preemptor (claims device-c, 8 units)
	preemptorClaim := createClaimWithSelector(tCtx, namespace, "claim-c", class, driverName, "device-c")
	preemptorPod := podWithClaimName.DeepCopy()
	preemptorPod.Spec.PriorityClassName = highPriority
	preemptor := createPod(tCtx, namespace, "-preemptor", preemptorPod, preemptorClaim)

	// Preemptor should trigger preemption and get scheduled
	waitForPodScheduled(tCtx, namespace, preemptor.Name)

	// Verify that BOTH victims are preempted!
	tCtx.Eventually(func(tCtx ktesting.TContext) int {
		podA, errA := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, victimA.Name, metav1.GetOptions{})
		podB, errB := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, victimB.Name, metav1.GetOptions{})
		if errA != nil || errB != nil {
			return 0
		}

		preemptedCount := 0
		if podA.DeletionTimestamp != nil {
			_, cond := podutil.GetPodCondition(&podA.Status, v1.DisruptionTarget)
			if cond != nil && cond.Status == v1.ConditionTrue {
				preemptedCount++
			}
		}
		if podB.DeletionTimestamp != nil {
			_, cond := podutil.GetPodCondition(&podB.Status, v1.DisruptionTarget)
			if cond != nil && cond.Status == v1.ConditionTrue {
				preemptedCount++
			}
		}
		return preemptedCount
	}).WithTimeout(30*time.Second).Should(gomega.Equal(2), "Both victim pods should be preempted with DisruptionTarget condition")

	waitForClaimReleased(tCtx, namespace, victimClaimA.Name)
	waitForClaimReleased(tCtx, namespace, victimClaimB.Name)
}

// testBindingConditionsPreemption verifies that preemption correctly handles claims
// with binding conditions, ensuring state is cleared and assumed pods can be preempted.
func testBindingConditionsPreemption(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)

	class, driverName := createTestClass(tCtx, namespace)

	// Create a ResourceSlice with a device that requires a binding condition.
	slice := st.MakeResourceSlice("worker-0", driverName).Obj()
	slice.Spec.Devices = []resourceapi.Device{
		{
			Name:              "with-binding",
			BindingConditions: []string{"attached"},
		},
	}
	createSlice(tCtx, slice)

	lowPriority := createPriorityClass(tCtx, namespace, "low-priority", 10)
	highPriority := createPriorityClass(tCtx, namespace, "high-priority", 100)

	// Create Victim Pod claiming the device
	claim := st.MakeResourceClaim().Name("claim").Request(class.Name).Obj()
	victimClaim := createClaim(tCtx, namespace, "-victim", class, claim)
	victimPod := podWithClaimName.DeepCopy()
	victimPod.Spec.PriorityClassName = lowPriority
	victim := createPod(tCtx, namespace, "-victim", victimPod, victimClaim)

	// Wait for claim to be allocated. The pod will be stuck in PreBind waiting for condition.
	waitForClaimAllocatedToDevice(tCtx, namespace, victimClaim.Name, schedulingTimeout)

	// Create Preemptor Pod claiming the same device class
	preemptorClaim := createClaim(tCtx, namespace, "-preemptor", class, claim)
	preemptorPod := podWithClaimName.DeepCopy()
	preemptorPod.Spec.PriorityClassName = highPriority
	preemptor := createPod(tCtx, namespace, "-preemptor", preemptorPod, preemptorClaim)

	// Preemptor should trigger preemption and get scheduled
	waitForPodScheduled(tCtx, namespace, preemptor.Name)

	// Verify that the Victim pod is preempted (deleted)
	waitForPodPreempted(tCtx, namespace, victim.Name)
}

// testPriorityAwarePreemption verifies that the scheduler chooses the lowest priority victim
// when multiple options fit, using only GA features (whole devices).
func testPriorityAwarePreemption(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)

	class, driverName := createTestClass(tCtx, namespace)

	// Create a ResourceSlice with two devices on worker-0
	slice := st.MakeResourceSlice("worker-0", driverName).Obj()
	slice.Spec.Devices = []resourceapi.Device{
		{
			Name: "device-1",
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				"name": {StringValue: ptr.To("device-1")},
			},
		},
		{
			Name: "device-2",
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				"name": {StringValue: ptr.To("device-2")},
			},
		},
	}
	createSlice(tCtx, slice)

	// Create PriorityClasses with different values
	lowPriority := createPriorityClass(tCtx, namespace, "low-priority", 10)
	midPriority := createPriorityClass(tCtx, namespace, "mid-priority", 20)
	highPriority := createPriorityClass(tCtx, namespace, "high-priority", 100)

	// Create Victim A (Priority 10, claims device-1)
	victimClaimA := createClaimWithSelector(tCtx, namespace, "claim-a", class, driverName, "device-1")
	victimPodA := podWithClaimName.DeepCopy()
	victimPodA.Spec.PriorityClassName = lowPriority
	victimA := createPod(tCtx, namespace, "-victim-a", victimPodA, victimClaimA)
	waitForPodScheduled(tCtx, namespace, victimA.Name)

	// Create Victim B (Priority 20, claims device-2)
	victimClaimB := createClaimWithSelector(tCtx, namespace, "claim-b", class, driverName, "device-2")
	victimPodB := podWithClaimName.DeepCopy()
	victimPodB.Spec.PriorityClassName = midPriority
	victimB := createPod(tCtx, namespace, "-victim-b", victimPodB, victimClaimB)
	waitForPodScheduled(tCtx, namespace, victimB.Name)

	// Create Preemptor (Priority 100, requests ANY device of the class)
	claim := st.MakeResourceClaim().Name("claim-c").Request(class.Name).Obj()
	preemptorClaim := createClaim(tCtx, namespace, "-preemptor", class, claim)
	preemptorPod := podWithClaimName.DeepCopy()
	preemptorPod.Spec.PriorityClassName = highPriority
	preemptor := createPod(tCtx, namespace, "-preemptor", preemptorPod, preemptorClaim)

	// Preemptor should trigger preemption and get scheduled
	waitForPodScheduled(tCtx, namespace, preemptor.Name)

	// Verify that Victim A (Priority 10) is preempted
	waitForPodPreempted(tCtx, namespace, victimA.Name)

	// Verify that Victim B (Priority 20) remains running
	podB, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, victimB.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get victim B")
	tCtx.Expect(podB.DeletionTimestamp).To(gomega.BeNil(), "Victim B should not be preempted")
}

func createPriorityClass(tCtx ktesting.TContext, namespace string, name string, value int32) string {
	tCtx.Helper()
	fullName := name + "-" + namespace
	pc := &schedulingv1.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{Name: fullName},
		Value:      value,
	}
	_, err := tCtx.Client().SchedulingV1().PriorityClasses().Create(tCtx, pc, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create priority class")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Log("Cleaning up PriorityClass...")
		deleteAndWait(tCtx, tCtx.Client().SchedulingV1().PriorityClasses().Delete, tCtx.Client().SchedulingV1().PriorityClasses().Get, pc.Name)
	})
	return fullName
}

func waitForPodPreempted(tCtx ktesting.TContext, namespace, podName string) {
	tCtx.Helper()
	tCtx.Eventually(func(tCtx ktesting.TContext) bool {
		p, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, podName, metav1.GetOptions{})
		if err != nil {
			return false
		}
		if p.DeletionTimestamp == nil {
			return false
		}
		_, cond := podutil.GetPodCondition(&p.Status, v1.DisruptionTarget)
		return cond != nil && cond.Status == v1.ConditionTrue
	}).WithTimeout(30*time.Second).Should(gomega.BeTrue(), fmt.Sprintf("Pod %s should be preempted (deletion timestamp set and DisruptionTarget condition true)", podName))
}

func waitForClaimReleased(tCtx ktesting.TContext, namespace, claimName string) {
	tCtx.Helper()
	tCtx.Eventually(func(tCtx ktesting.TContext) int {
		c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claimName, metav1.GetOptions{})
		if err != nil {
			return -1
		}
		return len(c.Status.ReservedFor)
	}).WithTimeout(30*time.Second).Should(gomega.Equal(0), fmt.Sprintf("Claim %s should be released", claimName))
}

func createClaimWithSelector(tCtx ktesting.TContext, namespace string, name string, class *resourceapi.DeviceClass, driverName string, deviceName string) *resourceapi.ResourceClaim {
	tCtx.Helper()
	claim := st.MakeResourceClaim().Name(name).Request(class.Name).Obj()
	claim.Spec.Devices.Requests[0].Exactly.Selectors = []resourceapi.DeviceSelector{
		{
			CEL: &resourceapi.CELDeviceSelector{
				Expression: fmt.Sprintf(`device.attributes["%s"].name == "%s"`, driverName, deviceName),
			},
		},
	}
	return createClaim(tCtx, namespace, "", class, claim)
}
