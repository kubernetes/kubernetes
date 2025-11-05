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
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/utils/format"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// testDeviceBindingConditions tests scheduling with mixed devices: one with BindingConditions, one without.
// It verifies that the scheduler prioritizes the device without BindingConditions for the first pod.
// The second pod then uses the device with BindingConditions. The test checks that the scheduler retries
// after an initial binding failure of the second pod, ensuring successful scheduling after rescheduling.
func testDeviceBindingConditions(tCtx ktesting.TContext, enabled bool) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)

	nodeName := "worker-0"
	poolWithBinding := nodeName + "-with-binding"
	poolWithoutBinding := nodeName + "-without-binding"
	bindingCondition := "attached"
	failureCondition := "failed"
	startScheduler(tCtx)

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

	// Schedule first pod and wait for the scheduler to reach the binding phase, which marks the claim as allocated.
	start := time.Now()
	claim1 := createClaim(tCtx, namespace, "-a", class, claim)
	pod := createPod(tCtx, namespace, "-a", claim1, podWithClaimName)
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *resourceapi.ResourceClaim {
		c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim1.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err)
		claim1 = c
		return claim1
	}).WithTimeout(10*time.Second).WithPolling(time.Second).Should(gomega.HaveField("Status.Allocation", gomega.Not(gomega.BeNil())), "Claim should have been allocated.")
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
	pod = createPod(tCtx, namespace, "-b", claim2, podWithClaimName)
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *resourceapi.ResourceClaim {
		c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim2.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err)
		claim2 = c
		return claim2
	}).WithTimeout(10*time.Second).WithPolling(time.Second).Should(gomega.HaveField("Status.Allocation", gomega.Not(gomega.BeNil())), "Claim should have been allocated.")
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

	// Wait until the claim.status.Devices[0].Conditions become nil again after rescheduling.
	setConditionsFlag := false
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *resourceapi.ResourceClaim {
		c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim2.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get claim")
		claim2 = c
		// Phase 1: saw conditions once
		if claim2.Status.Devices != nil && len(claim2.Status.Devices[0].Conditions) != 0 {
			setConditionsFlag = true
		}
		// Phase 2: after seeing conditions, wait until they are cleared AND allocation is present for the same device.
		if setConditionsFlag {
			// conditions cleared?
			cleared := len(claim2.Status.Devices) == 0
			// allocation restored and matches the intended device?
			allocated := claim2.Status.Allocation != nil &&
				len(claim2.Status.Allocation.Devices.Results) == 1 &&
				claim2.Status.Allocation.Devices.Results[0].Driver == driverName &&
				claim2.Status.Allocation.Devices.Results[0].Pool == poolWithBinding &&
				claim2.Status.Allocation.Devices.Results[0].Device == "with-binding"
			if cleared && allocated {
				return nil // done waiting
			}
		}
		return claim2
	}).WithTimeout(30*time.Second).WithPolling(time.Second).Should(gomega.BeNil(), "claim should be re-allocated to with-binding before proceeding")
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
