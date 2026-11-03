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
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/ktesting"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func testExtendedResource(tCtx ktesting.TContext, enabled, explicit bool) {
	tCtx.Parallel()

	namespace := createTestNamespace(tCtx, nil)

	// Create a DeviceClass with or without ExtendedResourceName based on whether we're testing explicit or implicit extended resources.
	var resourceName string
	var spec *resourceapi.DeviceClassSpec
	if explicit {
		resourceName = "example.com/" + namespace
		// Set extended resource name in the DeviceClass spec only for explicit resources.
		// It's not required for implicit extended resources.
		spec = &resourceapi.DeviceClassSpec{
			ExtendedResourceName: &resourceName,
		}
	}
	class, driverName := createTestClassWithSpec(tCtx, namespace, spec)
	if explicit {
		if enabled {
			require.NotEmpty(tCtx, class.Spec.ExtendedResourceName, "should store ExtendedResourceName")
		} else {
			require.Empty(tCtx, class.Spec.ExtendedResourceName, "should strip ExtendedResourceName")
		}
	} else {
		// For implicit extended resources, derive the resource name from the class.
		resourceName = resourceapi.ResourceDeviceClassPrefix + class.Name
	}

	slice := st.MakeResourceSlice("worker-0", driverName).Devices(device1)
	createSlice(tCtx, slice.Obj())

	startScheduler(tCtx)

	podWithOneContainer := st.MakePod().Name(podName).Namespace(namespace).Container("test-container").Obj()
	pod := createPodWithExtendedResource(tCtx, namespace, resourceName, "1", podWithOneContainer)

	var schedulingAttempted gtypes.GomegaMatcher
	if enabled {
		// Scheduled using device1 in the slice above.
		schedulingAttempted = gomega.HaveField("Status.Conditions", gomega.ContainElement(
			gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Type":   gomega.Equal(v1.PodScheduled),
				"Status": gomega.Equal(v1.ConditionTrue),
			}),
		))
	} else {
		schedulingAttempted = gomega.HaveField("Status.Conditions", gomega.ContainElement(
			gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Type":    gomega.Equal(v1.PodScheduled),
				"Status":  gomega.Equal(v1.ConditionFalse),
				"Reason":  gomega.Equal("Unschedulable"),
				"Message": gomega.Equal(fmt.Sprintf("0/8 nodes are available: 8 Insufficient %s. preemption: 0/8 nodes are available: 8 Preemption is not helpful for scheduling.", resourceName)),
			}),
		))
	}
	tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) {
		return tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	}).WithTimeout(time.Minute).WithPolling(time.Second).Should(schedulingAttempted)
}

func testExtendedResourceCapacity(tCtx ktesting.TContext, enabled bool) {
	if !enabled {
		tCtx.Run("FieldDropped", func(tCtx ktesting.TContext) {
			tCtx.Parallel()
			namespace := createTestNamespace(tCtx, nil)
			class, _ := createTestClassWithSpec(tCtx, namespace, sharesClassSpec("example.com/"+namespace, "shares"))
			require.Nil(tCtx, class.Spec.ExtendedResourceCapacityRequests, "should strip ExtendedResourceCapacityRequests")
		})
		return
	}

	tCtx.Run("SharedDevice", func(tCtx ktesting.TContext) {
		tCtx.Parallel()
		namespace := createTestNamespace(tCtx, nil)
		resourceName := "example.com/" + namespace
		_, driverName := createTestClassWithSpec(tCtx, namespace, sharesClassSpec(resourceName, "shares"))
		createSlice(tCtx, sharedDevicesSlice(driverName, "2", device1))
		startScheduler(tCtx)

		for _, suffix := range []string{"-a", "-b"} {
			pod := st.MakePod().Name(podName + suffix).Namespace(namespace).Container("test-container").Obj()
			pod = createPodWithExtendedResource(tCtx, namespace, resourceName, "1", pod)
			waitForPodScheduled(tCtx, namespace, pod.Name)
			claim := getExtendedResourceClaim(tCtx, namespace, pod.Name)
			expectOneShareEach(tCtx, claim, device1)
		}

		pod := st.MakePod().Name(podName + "-c").Namespace(namespace).Container("test-container").Obj()
		pod = createPodWithExtendedResource(tCtx, namespace, resourceName, "1", pod)
		expectPodUnschedulable(tCtx, pod, "cannot allocate all claims")
	})

	// One request does not use the same device twice.
	tCtx.Run("QuantityIsAllocationCount", func(tCtx ktesting.TContext) {
		tCtx.Parallel()
		namespace := createTestNamespace(tCtx, nil)
		resourceName := "example.com/" + namespace
		_, driverName := createTestClassWithSpec(tCtx, namespace, sharesClassSpec(resourceName, "shares"))
		createSlice(tCtx, sharedDevicesSlice(driverName, "2", device1, device2))
		startScheduler(tCtx)

		pod := st.MakePod().Name(podName).Namespace(namespace).Container("test-container").Obj()
		pod = createPodWithExtendedResource(tCtx, namespace, resourceName, "2", pod)
		waitForPodScheduled(tCtx, namespace, pod.Name)
		claim := getExtendedResourceClaim(tCtx, namespace, pod.Name)
		expectOneShareEach(tCtx, claim, device1, device2)
	})

	tCtx.Run("UnknownCapacity", func(tCtx ktesting.TContext) {
		tCtx.Parallel()
		namespace := createTestNamespace(tCtx, nil)
		resourceName := "example.com/" + namespace
		_, driverName := createTestClassWithSpec(tCtx, namespace, sharesClassSpec(resourceName, "memory"))
		createSlice(tCtx, sharedDevicesSlice(driverName, "2", device1))
		startScheduler(tCtx)

		pod := st.MakePod().Name(podName).Namespace(namespace).Container("test-container").Obj()
		pod = createPodWithExtendedResource(tCtx, namespace, resourceName, "1", pod)
		expectPodUnschedulable(tCtx, pod, "cannot allocate all claims")
	})
}

func sharesClassSpec(resourceName string, capacityName resourceapi.QualifiedName) *resourceapi.DeviceClassSpec {
	return &resourceapi.DeviceClassSpec{
		ExtendedResourceName: &resourceName,
		ExtendedResourceCapacityRequests: &resourceapi.CapacityRequirements{
			Requests: map[resourceapi.QualifiedName]resource.Quantity{
				capacityName: resource.MustParse("1"),
			},
		},
	}
}

func sharedDevicesSlice(driverName, shares string, deviceNames ...string) *resourceapi.ResourceSlice {
	slice := st.MakeResourceSlice("worker-0", driverName).Obj()
	for _, name := range deviceNames {
		slice.Spec.Devices = append(slice.Spec.Devices, resourceapi.Device{
			Name:                     name,
			AllowMultipleAllocations: new(true),
			Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
				"shares": {Value: resource.MustParse(shares)},
			},
		})
	}
	return slice
}

func getExtendedResourceClaim(tCtx ktesting.TContext, namespace, podName string) *resourceapi.ResourceClaim {
	tCtx.Helper()
	var claim *resourceapi.ResourceClaim
	tCtx.Eventually(func(tCtx ktesting.TContext) error {
		pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, podName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if pod.Status.ExtendedResourceClaimStatus == nil {
			return fmt.Errorf("pod %s has no extended resource claim status yet", podName)
		}
		claim, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, pod.Status.ExtendedResourceClaimStatus.ResourceClaimName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if claim.Status.Allocation == nil {
			return fmt.Errorf("claim %s is not allocated yet", claim.Name)
		}
		return nil
	}).WithTimeout(time.Minute).WithPolling(time.Second).Should(gomega.Succeed())
	return claim
}

func expectOneShareEach(tCtx ktesting.TContext, claim *resourceapi.ResourceClaim, deviceNames ...string) {
	tCtx.Helper()
	oneShare := map[resourceapi.QualifiedName]resource.Quantity{"shares": resource.MustParse("1")}
	gomega.NewWithT(tCtx).Expect(claim.Spec.Devices.Requests).To(gomega.ConsistOf(
		gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Exactly": gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Count":    gomega.Equal(int64(len(deviceNames))),
				"Capacity": gomega.Equal(&resourceapi.CapacityRequirements{Requests: oneShare}),
			})),
		}),
	))
	var results []gtypes.GomegaMatcher
	for _, name := range deviceNames {
		results = append(results, gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Device":           gomega.Equal(name),
			"ShareID":          gomega.Not(gomega.BeNil()),
			"ConsumedCapacity": gomega.Equal(oneShare),
		}))
	}
	gomega.NewWithT(tCtx).Expect(claim.Status.Allocation.Devices.Results).To(gomega.ConsistOf(results))
}
