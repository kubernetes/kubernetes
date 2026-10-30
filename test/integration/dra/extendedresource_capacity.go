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
	"github.com/onsi/gomega/gstruct"
	gtypes "github.com/onsi/gomega/types"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/ktesting"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

const sharesCapacity resourceapi.QualifiedName = "shares"

func testExtendedResourceCapacity(tCtx ktesting.TContext) {
	tCtx.Run("OneShare", func(tCtx ktesting.TContext) {
		testExtendedResourceCapacityAllocated(tCtx, "1", "2", 1)
	})
	tCtx.Run("TwoPodsShareOneDevice", func(tCtx ktesting.TContext) {
		testExtendedResourceCapacityTwoPods(tCtx)
	})
	tCtx.Run("CountThree", func(tCtx ktesting.TContext) {
		// One ExactDeviceRequest does not consume the same device more than once,
		// so count 3 needs three devices that can each provide one share.
		testExtendedResourceCapacityAllocated(tCtx, "3", "1", 3)
	})
	tCtx.Run("UnknownCapacity", func(tCtx ktesting.TContext) {
		testExtendedResourceCapacityUnknown(tCtx)
	})
}

func testExtendedResourceCapacityAllocated(tCtx ktesting.TContext, quantity, deviceShares string, allocations int) {
	tCtx.Parallel()

	namespace := createTestNamespace(tCtx, nil)
	resourceName := "example.com/" + namespace
	class, driverName := createTestClassWithSpec(tCtx, namespace, capacityClassSpec(resourceName, "1"))
	gomega.NewWithT(tCtx).Expect(class.Spec.ExtendedResourceCapacityRequests).ToNot(gomega.BeNil())

	slice := shareableSliceWithDevices("worker-0", driverName, deviceShares, allocations)
	createSlice(tCtx, slice)
	startScheduler(tCtx)

	pod := st.MakePod().Name(podName).Namespace(namespace).Container("test-container").Obj()
	pod = createPodWithExtendedResource(tCtx, namespace, resourceName, quantity, pod)
	waitForPodScheduled(tCtx, namespace, pod.Name)

	claim := waitForExtendedResourceClaim(tCtx, namespace, pod.Name)
	devices := make([]string, allocations)
	for i := range allocations {
		devices[i] = fmt.Sprintf("device-%d", i+1)
	}
	expectShareAllocations(tCtx, claim, devices, resource.MustParse("1"))
}

func testExtendedResourceCapacityTwoPods(tCtx ktesting.TContext) {
	tCtx.Parallel()

	namespace := createTestNamespace(tCtx, nil)
	resourceName := "example.com/" + namespace
	_, driverName := createTestClassWithSpec(tCtx, namespace, capacityClassSpec(resourceName, "1"))
	createSlice(tCtx, shareableSlice("worker-0", driverName, "2"))
	startScheduler(tCtx)

	var devices []string
	for i, name := range []string{podName, podName + "-2"} {
		pod := st.MakePod().Name(name).Namespace(namespace).Container("test-container").Obj()
		pod = createPodWithExtendedResource(tCtx, namespace, resourceName, "1", pod)
		waitForPodScheduled(tCtx, namespace, pod.Name)
		claim := waitForExtendedResourceClaim(tCtx, namespace, pod.Name)
		expectShareAllocations(tCtx, claim, []string{device1}, resource.MustParse("1"))
		devices = append(devices, claim.Status.Allocation.Devices.Results[0].Device)
		if i == 1 {
			gomega.NewWithT(tCtx).Expect(devices[0]).To(gomega.Equal(devices[1]), "both pods should consume the same device")
		}
	}
}

func testExtendedResourceCapacityUnknown(tCtx ktesting.TContext) {
	tCtx.Parallel()

	namespace := createTestNamespace(tCtx, nil)
	resourceName := "example.com/" + namespace
	spec := capacityClassSpec(resourceName, "1")
	spec.ExtendedResourceCapacityRequests.Requests = map[resourceapi.QualifiedName]resource.Quantity{
		"missing": resource.MustParse("1"),
	}
	_, driverName := createTestClassWithSpec(tCtx, namespace, spec)
	createSlice(tCtx, shareableSlice("worker-0", driverName, "4"))
	startScheduler(tCtx)

	pod := st.MakePod().Name(podName).Namespace(namespace).Container("test-container").Obj()
	pod = createPodWithExtendedResource(tCtx, namespace, resourceName, "1", pod)
	expectPodUnschedulable(tCtx, pod, "cannot allocate all claims")
}

func capacityClassSpec(resourceName, shares string) *resourceapi.DeviceClassSpec {
	return &resourceapi.DeviceClassSpec{
		ExtendedResourceName: &resourceName,
		ExtendedResourceCapacityRequests: &resourceapi.CapacityRequirements{
			Requests: map[resourceapi.QualifiedName]resource.Quantity{
				sharesCapacity: resource.MustParse(shares),
			},
		},
	}
}

func shareableSlice(nodeName, driverName, shares string) *resourceapi.ResourceSlice {
	return shareableSliceWithDevices(nodeName, driverName, shares, 1)
}

func shareableSliceWithDevices(nodeName, driverName, shares string, devices int) *resourceapi.ResourceSlice {
	slice := st.MakeResourceSlice(nodeName, driverName).Obj()
	slice.Spec.Devices = make([]resourceapi.Device, devices)
	for i := range devices {
		slice.Spec.Devices[i] = resourceapi.Device{
			Name:                     fmt.Sprintf("device-%d", i+1),
			AllowMultipleAllocations: ptr.To(true),
			Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
				sharesCapacity: {Value: resource.MustParse(shares)},
			},
		}
	}
	return slice
}

func waitForExtendedResourceClaim(tCtx ktesting.TContext, namespace, podName string) *resourceapi.ResourceClaim {
	tCtx.Helper()
	var claim *resourceapi.ResourceClaim
	tCtx.Eventually(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaim, error) {
		list, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).List(tCtx, metav1.ListOptions{})
		if err != nil {
			return nil, err
		}
		for i := range list.Items {
			item := &list.Items[i]
			if item.Annotations[resourceapi.ExtendedResourceClaimAnnotation] != "true" || item.Status.Allocation == nil {
				continue
			}
			for _, owner := range item.OwnerReferences {
				if owner.Name == podName {
					claim = item
					return item, nil
				}
			}
		}
		return nil, fmt.Errorf("allocated extended resource claim for %s not found", podName)
	}).WithTimeout(time.Minute).WithPolling(time.Second).Should(gomega.Not(gomega.BeNil()))
	return claim
}

func expectShareAllocations(tCtx ktesting.TContext, claim *resourceapi.ResourceClaim, devices []string, shares resource.Quantity) {
	tCtx.Helper()
	gomega.NewWithT(tCtx).Expect(claim.Spec.Devices.Requests).To(gomega.HaveLen(1))
	request := claim.Spec.Devices.Requests[0].Exactly
	gomega.NewWithT(tCtx).Expect(request).ToNot(gomega.BeNil())
	gomega.NewWithT(tCtx).Expect(request.Count).To(gomega.Equal(int64(len(devices))))
	gomega.NewWithT(tCtx).Expect(request.Capacity).To(gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Requests": gomega.Equal(map[resourceapi.QualifiedName]resource.Quantity{
			sharesCapacity: resource.MustParse("1"),
		}),
	})))

	var matchers []gtypes.GomegaMatcher
	for _, name := range devices {
		matchers = append(matchers, gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Device": gomega.Equal(name),
			"ConsumedCapacity": gomega.Equal(map[resourceapi.QualifiedName]resource.Quantity{
				sharesCapacity: shares,
			}),
			"ShareID": gomega.Not(gomega.BeNil()),
		}))
	}
	gomega.NewWithT(tCtx).Expect(claim.Status.Allocation.Devices.Results).To(gomega.ConsistOf(matchers))
}
