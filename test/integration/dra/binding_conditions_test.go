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
	"github.com/onsi/gomega/gstruct"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/utils/format"
	"k8s.io/kubernetes/test/utils/ktesting"
)

var (
	nodeName           = "worker-0"
	poolWithBinding    = nodeName + "-with-binding"
	poolWithoutBinding = nodeName + "-without-binding"
	bindingCondition   = "attached"
	failureCondition   = "failed"
)

// testDeviceBindingConditions is the entry point for running each integration test that verifies DeviceBindingConditions.
// Some of these tests use device taints, and they assume that DRADeviceTaints is enabled.
//
// In addition, some tests use custom scheduler configuration and therefore
// can't run in parallel. The ones that use the default scheduler configuration
// will run in parallel with each other, but not in parallel with
// non-DeviceBindingConditions tests.
func testDeviceBindingConditions(tCtx ktesting.TContext, enabled bool) {
	tCtx.Run("BasicFlow", func(tCtx ktesting.TContext) { testDeviceBindingConditionsBasicFlow(tCtx, enabled) })
	if enabled {
		tCtx.Run("FailureTaints", func(tCtx ktesting.TContext) { testDeviceBindingFailureConditionsReschedule(tCtx, true) })
		tCtx.Run("FailureRemove", func(tCtx ktesting.TContext) { testDeviceBindingFailureConditionsReschedule(tCtx, false) })
		tCtx.Run("TimeoutReached", func(tCtx ktesting.TContext) { testDeviceBindingConditionsTimeoutReached(tCtx) })
		tCtx.Run("TimeoutRecover", func(tCtx ktesting.TContext) { testDeviceBindingConditionsTimeoutRecovery(tCtx) })
	}
}

// testBindingConditionsBasicFlow tests scheduling with mixed devices: one with BindingConditions, one without.
// It verifies that the scheduler prioritizes the device without BindingConditions for the first pod.
// The second pod then uses the device with BindingConditions. The test checks that the scheduler retries
// after an initial binding failure of the second pod, ensuring successful scheduling after rescheduling.
func testDeviceBindingConditionsBasicFlow(tCtx ktesting.TContext, enabled bool) {
	tCtx.Parallel()

	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)

	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: namespace + "-",
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &nodeName,
			Pool: resourceapi.ResourcePool{
				Name:               poolWithBinding,
				ResourceSliceCount: 1,
			},
			Driver: driverName,
			Devices: []resourceapi.Device{
				{
					Name:                     "with-binding",
					BindingConditions:        []string{bindingCondition},
					BindingFailureConditions: []string{failureCondition},
				},
			},
		},
	}
	slice, err := tCtx.Client().ResourceV1().ResourceSlices().Create(tCtx, slice, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create slice")

	haveBindingConditionFields := len(slice.Spec.Devices[0].BindingConditions) > 0 || len(slice.Spec.Devices[0].BindingFailureConditions) > 0
	if !enabled {
		if haveBindingConditionFields {
			tCtx.Fatalf("Expected device binding condition fields to get dropped, got instead:\n%s", format.Object(slice, 1))
		}
		return
	}
	if !haveBindingConditionFields {
		tCtx.Fatalf("Expected device binding condition fields to be stored, got instead:\n%s", format.Object(slice, 1))
	}

	sliceWithoutBinding := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: namespace + "-without-binding-",
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &nodeName,
			Pool: resourceapi.ResourcePool{
				Name:               poolWithoutBinding,
				ResourceSliceCount: 1,
			},
			Driver: driverName,
			Devices: []resourceapi.Device{
				{
					Name: "without-binding",
				},
			},
		},
	}
	_, err = tCtx.Client().ResourceV1().ResourceSlices().Create(tCtx, sliceWithoutBinding, metav1.CreateOptions{FieldValidation: "Strict"})
	tCtx.ExpectNoError(err, "create slice without binding conditions")

	startScheduler(tCtx)
	// Schedule first pod and wait for the scheduler to reach the binding phase, which marks the claim as allocated.
	start := time.Now()
	claim1 := createClaim(tCtx, namespace, "-a", class, claim)
	pod := createPod(tCtx, namespace, "-a", podWithClaimName, claim1)
	claim1 = waitForClaimAllocatedToDevice(tCtx, namespace, claim1.Name, schedulingTimeout)
	end := time.Now()
	gomega.NewWithT(tCtx).Expect(claim1).To(gomega.HaveField("Status.Allocation", gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Devices": gomega.Equal(resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Request: claim1.Spec.Devices.Requests[0].Name,
				Driver:  driverName,
				Pool:    poolWithoutBinding,
				Device:  "without-binding",
			}}}),
		// NodeSelector intentionally not checked - that's covered elsewhere.
		"AllocationTimestamp": gomega.HaveField("Time", gomega.And(
			gomega.BeTemporally(">=", start.Truncate(time.Second) /* may get rounded down during round-tripping */),
			gomega.BeTemporally("<=", end),
		)),
	}))), "first allocated claim")

	waitForPodScheduled(tCtx, namespace, pod.Name)

	// Second pod should get the device with binding conditions.
	claim2 := createClaim(tCtx, namespace, "-b", class, claim)
	pod = createPod(tCtx, namespace, "-b", podWithClaimName, claim2)
	claim2 = waitForClaimAllocatedToDevice(tCtx, namespace, claim2.Name, schedulingTimeout)
	end = time.Now()
	gomega.NewWithT(tCtx).Expect(claim2).To(gomega.HaveField("Status.Allocation", gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Devices": gomega.Equal(resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Request:                  claim2.Spec.Devices.Requests[0].Name,
				Driver:                   driverName,
				Pool:                     poolWithBinding,
				Device:                   "with-binding",
				BindingConditions:        []string{bindingCondition},
				BindingFailureConditions: []string{failureCondition},
			}}}),
		// NodeSelector intentionally not checked - that's covered elsewhere.
		"AllocationTimestamp": gomega.HaveField("Time", gomega.And(
			gomega.BeTemporally(">=", start.Truncate(time.Second) /* may get rounded down during round-tripping */),
			gomega.BeTemporally("<=", end),
		)),
	}))), "second allocated claim")

	// fail the binding condition for the second claim, so that it gets scheduled later.
	err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
		latest, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim2.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		latest.Status.Devices = []resourceapi.AllocatedDeviceStatus{{
			Driver: driverName,
			Pool:   poolWithBinding,
			Device: "with-binding",
			Conditions: []metav1.Condition{{
				Type:               failureCondition,
				Status:             metav1.ConditionTrue,
				ObservedGeneration: latest.Generation,
				LastTransitionTime: metav1.Now(),
				Reason:             "Testing",
				Message:            "The test has seen the allocation and is failing the binding.",
			}},
		}}
		_, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).UpdateStatus(tCtx, latest, metav1.UpdateOptions{})
		return err
	})
	tCtx.ExpectNoError(err, "add binding failure condition to second claim")

	// Then wait until the scheduler has cleared the device statuses again.
	waitForClaim(tCtx, namespace, claim2.Name, schedulingTimeout,
		gomega.HaveField("Status.Devices", gomega.HaveLen(0)),
		"claim should have cleared device conditions after rescheduling",
	)

	// allocation restored?
	claim2 = waitForClaimAllocatedToDevice(tCtx, namespace, claim2.Name, schedulingTimeout)

	// Now it's safe to set the final binding condition.
	// Allow the scheduler to proceed.
	err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
		latest, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim2.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		// Write status.devices for the CURRENT allocation device.
		latest.Status.Devices = []resourceapi.AllocatedDeviceStatus{{
			Driver: driverName,
			Pool:   poolWithBinding,
			Device: "with-binding",
			Conditions: []metav1.Condition{{
				Type:               bindingCondition,
				Status:             metav1.ConditionTrue,
				ObservedGeneration: latest.Generation,
				LastTransitionTime: metav1.Now(),
				Reason:             "Testing",
				Message:            "The test has seen the allocation.",
			}},
		}}
		_, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).UpdateStatus(tCtx, latest, metav1.UpdateOptions{})
		return err
	})
	tCtx.ExpectNoError(err, "add binding condition to second claim")
	waitForPodScheduled(tCtx, namespace, pod.Name)
}

// testBindingFailureReschedule verifies scheduling behavior when device preparation fails on a node.
// It tests that a BindingFailure is written, and the scheduler successfully reschedules the pod
// to a different node where binding succeeds. This ensures that failure recovery via rescheduling works as expected.
// Device preparation failure is simulated in two ways: by applying DeviceTaints or by removing the device from ResourceSlice.
// The simulation method is controlled via the `useTaints` argument: when true, DeviceTaints are used; when false, the device is removed from ResourceSlice.
func testDeviceBindingFailureConditionsReschedule(tCtx ktesting.TContext, useTaints bool) {
	tCtx.Parallel()

	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)
	startScheduler(tCtx)

	anotherNodeName := "worker-1"
	anotherPoolWithoutBinding := anotherNodeName + "-without-binding"

	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: namespace + "-",
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &nodeName,
			Pool: resourceapi.ResourcePool{
				Name:               poolWithBinding,
				ResourceSliceCount: 1,
			},
			Driver: driverName,
			Devices: []resourceapi.Device{
				{
					Name:                     "with-binding",
					BindingConditions:        []string{bindingCondition},
					BindingFailureConditions: []string{failureCondition},
				},
			},
		},
	}
	slice = createSlice(tCtx, slice)

	// Schedule the first pod to a device that has binding conditions set,
	// ensuring the initial allocation occurs on the intended node.
	claim1 := createClaim(tCtx, namespace, "-a", class, claim)
	pod := createPod(tCtx, namespace, "-a", podWithClaimName, claim1)
	claim1 = waitForClaimAllocatedToDevice(tCtx, namespace, claim1.Name, schedulingTimeout)
	gomega.NewWithT(tCtx).Expect(claim1).To(gomega.HaveField("Status.Allocation", gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Devices": gomega.Equal(resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Request:                  claim1.Spec.Devices.Requests[0].Name,
				Driver:                   driverName,
				Pool:                     poolWithBinding,
				Device:                   "with-binding",
				BindingConditions:        []string{bindingCondition},
				BindingFailureConditions: []string{failureCondition},
			}}}),
	}))), "third allocated claim to the device with binding conditions")

	if useTaints {
		// Add taint to the device with binding conditions,
		// preventing further scheduling to this device.
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			latest, err := tCtx.Client().ResourceV1().ResourceSlices().Get(tCtx, slice.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			slice = latest.DeepCopy()
			slice.Spec.Devices[0].Taints = []resourceapi.DeviceTaint{
				{
					Key:    "dra-test.k8s.io/preparation-failed",
					Value:  "true",
					Effect: resourceapi.DeviceTaintEffectNoSchedule,
				},
			}
			_, err = tCtx.Client().ResourceV1().ResourceSlices().Update(tCtx, slice, metav1.UpdateOptions{})
			return err
		})
		tCtx.ExpectNoError(err, "add taint to second slice")
	} else {
		// Remove the device from the slice to simulate its unavailability due to preparation failure.
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			latest, err := tCtx.Client().ResourceV1().ResourceSlices().Get(tCtx, slice.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			slice = latest.DeepCopy()
			slice.Spec.Devices = nil
			_, err = tCtx.Client().ResourceV1().ResourceSlices().Update(tCtx, slice, metav1.UpdateOptions{})
			return err
		})
		tCtx.ExpectNoError(err, "remove devices in slice")
	}

	// Create a new slice on a different node with a device that has no binding conditions,
	// allowing the scheduler to retry and allocate the claim successfully.
	sliceWithoutBinding := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: namespace + "-without-binding-",
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &anotherNodeName,
			Pool: resourceapi.ResourcePool{
				Name:               anotherPoolWithoutBinding,
				ResourceSliceCount: 1,
			},
			Driver: driverName,
			Devices: []resourceapi.Device{
				{
					Name: "without-binding",
				},
			},
		},
	}
	createSlice(tCtx, sliceWithoutBinding)

	// Explicitly fail the binding condition for the third claim to trigger rescheduling logic.
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		latest, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim1.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		claim1 = latest.DeepCopy()
		claim1.Status.Devices = []resourceapi.AllocatedDeviceStatus{{
			Driver: driverName,
			Pool:   poolWithBinding,
			Device: "with-binding",
			Conditions: []metav1.Condition{{
				Type:               failureCondition,
				Status:             metav1.ConditionTrue,
				ObservedGeneration: claim1.Generation,
				LastTransitionTime: metav1.Now(),
				Reason:             "Testing",
				Message:            "The test has seen the allocation and is failing the binding.",
			}},
		}}
		_, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).UpdateStatus(tCtx, claim1, metav1.UpdateOptions{})
		return err
	})
	tCtx.ExpectNoError(err, "add binding failure condition to claim")

	// Then wait until the scheduler has cleared the device statuses again.
	waitForClaim(tCtx, namespace, claim1.Name, schedulingTimeout,
		gomega.HaveField("Status.Devices", gomega.HaveLen(0)),
		"claim should have cleared device conditions after rescheduling",
	)

	// allocation restored?
	claim1 = waitForClaimAllocatedToDevice(tCtx, namespace, claim1.Name, schedulingTimeout)

	gomega.NewWithT(tCtx).Expect(claim1).To(gomega.HaveField("Status.Allocation", gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Devices": gomega.Equal(resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Request: claim1.Spec.Devices.Requests[0].Name,
				Driver:  driverName,
				Pool:    anotherPoolWithoutBinding,
				Device:  "without-binding",
			}}}),
	}))), "third allocated claim to the device without binding conditions")

	waitForPodScheduled(tCtx, namespace, pod.Name)
}

// testBindingConditionsTimeoutReachedd verifies that a short bindingTimeout triggers
// a PreBind timeout when the required BindingConditions never become true.
//
// It runs the scheduler with non-standard settings and thus cannot run in parallel.
func testDeviceBindingConditionsTimeoutReached(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	class, driver := createTestClass(tCtx, namespace)

	// One device that REQUIRES a binding condition.
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{GenerateName: namespace + "-timeout-"},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &nodeName,
			Pool: resourceapi.ResourcePool{
				Name:               poolWithBinding,
				ResourceSliceCount: 1,
			},
			Driver: driver,
			Devices: []resourceapi.Device{{
				Name:                     "with-binding",
				BindingConditions:        []string{bindingCondition},
				BindingFailureConditions: []string{failureCondition},
			}},
		},
	}
	createSlice(tCtx, slice)

	wantTO := 6 * time.Second // bindingTimeout
	maxTO := wantTO * 19 / 10 // 1.9 * wantTO

	// Start the scheduler with a short binding timeout.
	cfg := fmt.Sprintf(`
profiles:
- schedulerName: default-scheduler
  pluginConfig:
  - name: DynamicResources
    args:
      bindingTimeout: %s
`, wantTO)

	startSchedulerWithConfig(tCtx, cfg)

	// Create claim+pod: allocation happens, then scheduler waits in PreBind.
	claim1 := createClaim(tCtx, namespace, "-timeout-enforced", class, claim)
	pod := createPod(tCtx, namespace, "-timeout-enforced", podWithClaimName, claim1)

	// Wait until the claim is allocated.
	allocatedClaim := waitForClaimAllocatedToDevice(tCtx, namespace, claim1.Name, schedulingTimeout)

	gomega.NewWithT(tCtx).Expect(allocatedClaim).To(gomega.HaveField(
		"Status.Allocation",
		gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Devices": gomega.Equal(resourceapi.DeviceAllocationResult{
				Results: []resourceapi.DeviceRequestAllocationResult{{
					Request:                  allocatedClaim.Spec.Devices.Requests[0].Name,
					Driver:                   driver,
					Pool:                     poolWithBinding,
					Device:                   "with-binding",
					BindingConditions:        []string{bindingCondition},
					BindingFailureConditions: []string{failureCondition},
				}},
			}),
		}),
		)), "Claim must be allocated to the condition-gated device")

	tStart := time.Now()

	// The scheduler should hit the binding timeout and surface that on the pod.
	// We poll the pod's conditions until we see a message containing "binding timeout".
	tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) {
		return tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	}).WithTimeout(maxTO).WithPolling(300*time.Millisecond).Should(
		gomega.HaveField("Status.Conditions",
			gomega.ContainElement(
				gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Message": gomega.ContainSubstring("binding timeout"),
				}),
			),
		),
		"pod should report binding timeout in a condition message",
	)

	elapsed := time.Since(tStart)
	gomega.NewWithT(tCtx).Expect(elapsed).To(
		gomega.BeNumerically("<=", maxTO),
		"bindingTimeout should trigger roughly near %s (observed %v)", wantTO, elapsed,
	)
	// Verify that the pod remains unscheduled after the binding timeout.
	tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) {
		return tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	}).WithTimeout(wantTO).WithPolling(200 * time.Millisecond).Should(gomega.SatisfyAll(
		gomega.HaveField("Spec.NodeName", gomega.BeEmpty()),

		gomega.HaveField("Status.Conditions",
			gomega.Not(gomega.ContainElement(v1.PodCondition{
				Type:   v1.PodScheduled,
				Status: v1.ConditionTrue,
			}))),
	))
}

// testDeviceBindingConditionsTimeoutRecovery verifies that when a device with BindingConditions
// fails to become ready within the timeout (BindingTimeout enforced), and a new device without
// binding conditions is added, the scheduler reschedules the claim to the new available device.
//
// It runs the scheduler with non-standard settings and thus cannot run in parallel.
func testDeviceBindingConditionsTimeoutRecovery(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)

	// Start the scheduler with a short binding timeout.
	const cfg = `
profiles:
- schedulerName: default-scheduler
  pluginConfig:
  - name: DynamicResources
    args:
      bindingTimeout: 5s
`
	startSchedulerWithConfig(tCtx, cfg)

	// Initial slice: one device that *requires* a binding condition that never becomes true.
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: namespace + "-",
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &nodeName,
			Pool: resourceapi.ResourcePool{
				Name:               poolWithBinding,
				ResourceSliceCount: 1,
			},
			Driver: driverName,
			Devices: []resourceapi.Device{{
				Name:                     "with-binding",
				BindingConditions:        []string{bindingCondition},
				BindingFailureConditions: []string{failureCondition},
			}},
		},
	}
	createSlice(tCtx, slice)
	claim1 := createClaim(tCtx, namespace, "-timeout", class, claim)
	pod := createPod(tCtx, namespace, "-timeout", podWithClaimName, claim1)

	claim1 = waitForClaimAllocatedToDevice(tCtx, namespace, claim1.Name, schedulingTimeout)
	gomega.NewWithT(tCtx).Expect(claim1).To(gomega.HaveField(
		"Status.Allocation",
		gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Devices": gomega.Equal(resourceapi.DeviceAllocationResult{
				Results: []resourceapi.DeviceRequestAllocationResult{{
					Request:                  claim.Spec.Devices.Requests[0].Name,
					Driver:                   driverName,
					Pool:                     poolWithBinding,
					Device:                   "with-binding",
					BindingConditions:        []string{bindingCondition},
					BindingFailureConditions: []string{failureCondition},
				}},
			}),
		}),
		)), "Expected allocation to the condition-gated device")

	sliceWithoutBinding := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{GenerateName: namespace + "-recovery-"},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &nodeName,
			Pool: resourceapi.ResourcePool{
				Name:               poolWithoutBinding,
				ResourceSliceCount: 1,
			},
			Driver: driverName,
			Devices: []resourceapi.Device{{
				Name: "without-binding",
			}},
		},
	}
	sliceWithoutBinding = createSlice(tCtx, sliceWithoutBinding)

	// Ensure the ResourceSlice has been created before the binding timeout occurs.
	tCtx.Eventually(func(tCtx ktesting.TContext) error {
		p, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
		if err == nil {
			for _, c := range p.Status.Conditions {
				if strings.Contains(strings.ToLower(c.Message), "binding timeout") {
					return gomega.StopTrying("binding timeout occurred before slice is created")
				}
			}
		}
		_, err = tCtx.Client().ResourceV1().ResourceSlices().Get(tCtx, sliceWithoutBinding.Name, metav1.GetOptions{})
		return err
	}).WithTimeout(schedulingTimeout).WithPolling(300*time.Millisecond).Should(
		gomega.Succeed(), "slice must be created before binding timeout")

	// Wait until the binding timeout occurs.
	tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) {
		return tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	}).WithTimeout(20*time.Second).WithPolling(300*time.Millisecond).Should(
		gomega.HaveField("Status.Conditions",
			gomega.ContainElement(
				gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Message": gomega.ContainSubstring("binding timeout"),
				}),
			),
		),
		"pod should report binding timeout before reallocation",
	)

	// Verify recovery to the newly added device without BindingConditions through rescheduling triggered by binding timeout.
	tCtx.Eventually(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaim, error) {
		return tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim1.Name, metav1.GetOptions{})
	}).WithTimeout(schedulingTimeout).WithPolling(1*time.Second).Should(gomega.HaveField(
		"Status.Allocation",
		gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Devices": gomega.Equal(resourceapi.DeviceAllocationResult{
				Results: []resourceapi.DeviceRequestAllocationResult{{
					Request: claim.Spec.Devices.Requests[0].Name,
					Driver:  driverName,
					Pool:    poolWithoutBinding,
					Device:  "without-binding",
				}},
			}),
		}),
		)), "Expected allocation to the device without binding conditions")

	waitForPodScheduled(tCtx, namespace, pod.Name)
}
