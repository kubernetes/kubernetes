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

package e2edra

import (
	"fmt"
	"strings"

	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func builderForPartitionableDevices(tCtx ktesting.TContext, nodes *drautils.Nodes) *drautils.Builder {
	// setup driver
	driver := drautils.NewDriverInstance(tCtx)
	driver.SetNameSuffix(tCtx, "partitionable")
	driver.IsLocal = true
	driver.Run(tCtx, "/var/lib/kubelet", nodes, nil)

	b := drautils.NewBuilderNow(tCtx, driver)
	b.SkipCleanup = true
	return b
}

var devices = []resourceapi.Device{
	{
		Name: "memory0-2g",
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"memory": {Value: resource.MustParse("2Gi")},
		},
		ConsumesCounters: []resourceapi.DeviceCounterConsumption{
			{
				CounterSet: "memory-pool",
				Counters: map[string]resourceapi.Counter{
					"memory": {
						Value: resource.MustParse("2Gi"),
					},
				},
			},
		},
	},
	{
		Name: "memory1-2g",
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"memory": {Value: resource.MustParse("2Gi")},
		},
		ConsumesCounters: []resourceapi.DeviceCounterConsumption{
			{
				CounterSet: "memory-pool",
				Counters: map[string]resourceapi.Counter{
					"memory": {
						Value: resource.MustParse("2Gi"),
					},
				},
			},
		},
	},
	{
		Name: "memory2-2g",
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"memory": {Value: resource.MustParse("2Gi")},
		},
		ConsumesCounters: []resourceapi.DeviceCounterConsumption{
			{
				CounterSet: "memory-pool",
				Counters: map[string]resourceapi.Counter{
					"memory": {
						Value: resource.MustParse("2Gi"),
					},
				},
			},
		},
	},
	{
		Name: "memory3-2g",
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"memory": {Value: resource.MustParse("2Gi")},
		},
		ConsumesCounters: []resourceapi.DeviceCounterConsumption{
			{
				CounterSet: "memory-pool",
				Counters: map[string]resourceapi.Counter{
					"memory": {
						Value: resource.MustParse("2Gi"),
					},
				},
			},
		},
	},
	{
		Name: "memory4-4g",
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"memory": {Value: resource.MustParse("4Gi")},
		},
		ConsumesCounters: []resourceapi.DeviceCounterConsumption{
			{
				CounterSet: "memory-pool",
				Counters: map[string]resourceapi.Counter{
					"memory": {
						Value: resource.MustParse("4Gi"),
					},
				},
			},
		},
	},
	{
		Name: "memory5-4g",
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"memory": {Value: resource.MustParse("4Gi")},
		},
		ConsumesCounters: []resourceapi.DeviceCounterConsumption{
			{
				CounterSet: "memory-pool",
				Counters: map[string]resourceapi.Counter{
					"memory": {
						Value: resource.MustParse("4Gi"),
					},
				},
			},
		},
	},
}

var sharedCounters = []resourceapi.CounterSet{
	{
		Name: "memory-pool",
		Counters: map[string]resourceapi.Counter{
			"memory": {
				Value: resource.MustParse("8Gi"),
			},
		},
	},
}

// partitionableDevices tests the DRAPartitionableDevices feature across upgrade/downgrade.
// It creates ResourceSlices with SharedCounters and devices that consume those counters,
// then verifies that these API fields are preserved across cluster version transitions.
//
// This test focuses on API preservation rather than scheduler behavior, similar to
// the resourceClaimDeviceStatus test.
func partitionableDevices(tCtx ktesting.TContext, b *drautils.Builder) upgradedTestFunc {
	driverName := b.Driver.Name

	// check version to create a single resource slice for 1.34 and separate resource slices for >=1.35.
	// https://github.com/kubernetes/enhancements/pull/5515#issuecomment-3268402336
	serverVersion, err := tCtx.Client().Discovery().ServerVersion()
	tCtx.ExpectNoError(err, "get cluster server version")
	semanticVersion, err := version.ParseGeneric(serverVersion.GitVersion)
	tCtx.ExpectNoError(err, "parse cluster server version %q", serverVersion.GitVersion)
	isLessThan135 := semanticVersion.LessThan(version.MustParseSemantic("v1.35.0"))

	var resourceSlices []*resourceapi.ResourceSlice
	if isLessThan135 {
		resourceSlices = resourceSlicesForLessThan135(driverName)
	} else {
		resourceSlices = resourceSlicesFor135AndNewer(driverName)
	}
	for _, slice := range resourceSlices {
		b.Create(tCtx, slice)
	}

	namespace := tCtx.Namespace()

	// Create pods requesting different memory sizes via resource claims.
	// The pool has 8Gi total: a 2Gi + 4Gi allocation should succeed (6Gi used),
	// but a second 4Gi allocation should fail (would exceed 8Gi capacity).
	pod2g, claim2g := createPodWithClaim(tCtx, b, driverName, "2Gi")
	pod4g, claim4g := createPodWithClaim(tCtx, b, driverName, "4Gi")

	// Wait for pods that fit within counter capacity to be running.
	tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod2g))
	tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod4g))

	// A second 4Gi pod should be unschedulable because the shared counter
	// pool only has 2Gi remaining (8Gi total - 2Gi - 4Gi = 2Gi).
	pod4g2, claim4g2 := createPodWithClaim(tCtx, b, driverName, "4Gi")
	tCtx.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(tCtx, tCtx.Client(), pod4g2.Name, namespace), "Pod with a claim over the counter should be unschedulable")

	// A second 2Gi pod shouold be schedulable.
	pod2g2, claim2g2 := createPodWithClaim(tCtx, b, driverName, "2Gi")
	tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod2g2))
	b.DeletePodAndWaitForNotFound(tCtx, pod2g2)
	tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claim2g2.Name, metav1.DeleteOptions{}))

	return func(tCtx ktesting.TContext) downgradedTestFunc {
		// After upgrade: the scheduled pods are still running, and the unschedulable pod remains unschedulable.

		// Recreate resource slices
		// If the cluster was upgraded from <1.35, it needs to recreate resource slice because it's invalid now
		// because >= 1.35 clusters require two slices for sharedCounters and devices.
		if isLessThan135 {
			for _, slice := range resourceSlices {
				tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceSlices().Delete(tCtx, slice.Name, metav1.DeleteOptions{}))
			}
			resourceSlices = resourceSlicesFor135AndNewer(driverName)
			for _, slice := range resourceSlices {
				b.Create(tCtx, slice)
			}
		}

		verifyPodsState(tCtx, namespace, []*v1.Pod{pod2g, pod4g}, pod4g2)

		// Verify scheduling behavior and shared counters are preserved after upgrade.
		verifyScheduling(tCtx, b, driverName, namespace)

		// Make sure the old resources are deleted before downgrade, otherwise when the cluster is downgraded to <1.35,
		// only the devices slice is evaluated without shared counters, and pod4g2 will be scheduled.
		for _, slice := range resourceSlices {
			tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceSlices().Delete(tCtx, slice.Name, metav1.DeleteOptions{}))
		}

		return func(tCtx ktesting.TContext) {
			// After downgrade: verify that the resource state is preserved.

			// Recreate resource slices
			// If the cluster was upgraded from <1.35, it needs to recreate resource slice because it's invalid now
			// because < 1.35 clusters require single slice for sharedCounters and devices.
			if isLessThan135 {
				resourceSlices = resourceSlicesForLessThan135(driverName)
				for _, slice := range resourceSlices {
					b.Create(tCtx, slice)
				}
			}

			verifyPodsState(tCtx, namespace, []*v1.Pod{pod2g, pod4g}, pod4g2)

			// Verify scheduling behavior and shared counters are preserved after downgrade.
			verifyScheduling(tCtx, b, driverName, namespace)

			// Cleanup: delete pods and claims.
			b.DeletePodAndWaitForNotFound(tCtx, pod2g)
			b.DeletePodAndWaitForNotFound(tCtx, pod4g)
			b.DeletePodAndWaitForNotFound(tCtx, pod4g2)

			tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claim2g.Name, metav1.DeleteOptions{}))
			tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claim4g.Name, metav1.DeleteOptions{}))
			tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claim4g2.Name, metav1.DeleteOptions{}))

			for _, slice := range resourceSlices {
				tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceSlices().Delete(tCtx, slice.Name, metav1.DeleteOptions{}))
			}
		}
	}
}

// verifyScheduling verifies that a 4Gi pod is unschedulable (only 2Gi remains),
// that shared counters report 2Gi available, and that a 2Gi pod is schedulable.
func verifyScheduling(tCtx ktesting.TContext, b *drautils.Builder, driverName, namespace string) {
	pod4g, claim4g := createPodWithClaim(tCtx, b, driverName, "4Gi")
	tCtx.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(tCtx, tCtx.Client(), pod4g.Name, namespace), "Pod with a claim over the counter should be unschedulable")
	b.DeletePodAndWaitForNotFound(tCtx, pod4g)
	tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claim4g.Name, metav1.DeleteOptions{}))

	pod2g, claim2g := createPodWithClaim(tCtx, b, driverName, "2Gi")
	tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod2g))
	b.DeletePodAndWaitForNotFound(tCtx, pod2g)
	tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claim2g.Name, metav1.DeleteOptions{}))
}

// verifyPodsState checks that the given running pods are still running with no
// container restarts, and that the unschedulable pod remains unschedulable.
func verifyPodsState(tCtx ktesting.TContext, namespace string, runningPods []*v1.Pod, unschedulablePod *v1.Pod) {
	for _, pod := range runningPods {
		p, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get Pod %s", pod.Name)
		tCtx.Expect(p.Status.Phase).To(gomega.Equal(v1.PodRunning), "Pod %s should still be running", pod.Name)
		for _, cs := range p.Status.ContainerStatuses {
			tCtx.Expect(cs.RestartCount).To(gomega.Equal(int32(0)), "Container %s in Pod %s should have no restarts", cs.Name, pod.Name)
		}
	}

	tCtx.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(tCtx, tCtx.Client(), unschedulablePod.Name, namespace), "Pod %s should still be unschedulable", unschedulablePod.Name)
}

func createPodWithClaim(tCtx ktesting.TContext, b *drautils.Builder, driverName, value string) (*v1.Pod, *resourceapi.ResourceClaim) {
	claim := b.ExternalClaim()
	claim.Spec.Devices.Requests[0] = resourceapi.DeviceRequest{
		Name: fmt.Sprintf("%s-memory", strings.ToLower(value)),
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: b.ClassName(),
			Selectors: []resourceapi.DeviceSelector{
				{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("%s")) == 0`, driverName, value),
					},
				},
			},
		},
	}

	pod := b.PodExternal()
	podClaimName := fmt.Sprintf("claim-%s", claim.Name)
	pod.Spec.ResourceClaims = []v1.PodResourceClaim{
		{
			Name:              podClaimName,
			ResourceClaimName: &claim.Name,
		},
	}
	pod.Spec.Containers[0].Resources.Claims = []v1.ResourceClaim{{Name: podClaimName}}
	b.Create(tCtx, claim, pod)
	return pod, claim
}

func resourceSlicesFor135AndNewer(driverName string) []*resourceapi.ResourceSlice {
	counterSliceName := "partitionable-counter-slice"
	devicesSliceName := "partitionable-devices-slice"

	counterSlice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: counterSliceName,
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:         driverName,
			AllNodes:       ptr.To(true),
			SharedCounters: sharedCounters,
			Pool: resourceapi.ResourcePool{
				Name:               "partitionable-pool",
				Generation:         1,
				ResourceSliceCount: 2, // One for counters, one for devices
			},
		},
	}

	devicesSlice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: devicesSliceName,
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:   driverName,
			AllNodes: ptr.To(true),
			Devices:  devices,
			Pool: resourceapi.ResourcePool{
				Name:               "partitionable-pool",
				Generation:         1,
				ResourceSliceCount: 2,
			},
		},
	}

	return []*resourceapi.ResourceSlice{counterSlice, devicesSlice}
}

func resourceSlicesForLessThan135(driverName string) []*resourceapi.ResourceSlice {
	partitionableSliceName := "partitionable-slice"

	partitionableSlice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: partitionableSliceName,
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:         driverName,
			AllNodes:       ptr.To(true),
			SharedCounters: sharedCounters,
			Devices:        devices,
			Pool: resourceapi.ResourcePool{
				Name:               "partitionable-pool",
				Generation:         1,
				ResourceSliceCount: 1,
			},
		},
	}

	return []*resourceapi.ResourceSlice{partitionableSlice}
}
