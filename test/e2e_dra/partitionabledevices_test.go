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

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// partitionableDeviceResources tests the DRAPartitionableDevices feature across upgrade/downgrade.
// It creates ResourceSlices with SharedCounters and devices that consume those counters,
// then verifies that these API fields are preserved across cluster version transitions.
//
// This test focuses on API preservation rather than scheduler behavior, similar to
// the resourceClaimDeviceStatus test.
func partitionableDeviceResources(nodes *drautils.Nodes) map[string]resourceslice.DriverResources {
	nodename := nodes.NodeNames[0]
	return map[string]resourceslice.DriverResources{
		nodename: {
			Pools: map[string]resourceslice.Pool{
				"partitionable-pool": {
					Slices: []resourceslice.Slice{
						// Devices and SharedCounters must be exclusive per design.
						{SharedCounters: partitionableSharedCounters},
						{Devices: partitionableDeviceList},
					},
				},
			},
		},
	}
}

var partitionableDeviceList = []resourceapi.Device{
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

var partitionableSharedCounters = []resourceapi.CounterSet{
	{
		Name: "memory-pool",
		Counters: map[string]resourceapi.Counter{
			"memory": {
				Value: resource.MustParse("8Gi"),
			},
		},
	},
}

func partitionableDevices(tCtx ktesting.TContext, b *drautils.Builder) upgradedTestFunc {
	driverName := b.Driver.Name
	namespace := tCtx.Namespace()

	// Create pods requesting different memory sizes via resource claims.
	// The pool has 8Gi total: a 2Gi + 4Gi allocation should succeed (6Gi used),
	// but a second 4Gi allocation should fail (would exceed 8Gi capacity).
	pod2g, claim2g := partitionableCreatePodWithClaim(tCtx, b, driverName, "2Gi")
	pod4g, claim4g := partitionableCreatePodWithClaim(tCtx, b, driverName, "4Gi")

	// Wait for pods to be running so that the third pod won't be scheduled before they are scheduled.
	tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod2g))
	tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod4g))

	// A second 4Gi pod should be unschedulable because the shared counter
	// pool only has 2Gi remaining (8Gi total - 2Gi - 4Gi = 2Gi).
	pod4g2, claim4g2 := partitionableCreatePodWithClaim(tCtx, b, driverName, "4Gi")
	tCtx.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(tCtx, tCtx.Client(), pod4g2.Name, namespace), "Pod with a claim over the counter should be unschedulable")

	partitionableDoTest(tCtx, b, driverName, []*v1.Pod{pod2g, pod4g}, pod4g2)

	return func(tCtx ktesting.TContext) downgradedTestFunc {
		// After upgrade
		partitionableDoTest(tCtx, b, driverName, []*v1.Pod{pod2g, pod4g}, pod4g2)

		return func(tCtx ktesting.TContext) {
			// After downgrade
			partitionableDoTest(tCtx, b, driverName, []*v1.Pod{pod2g, pod4g}, pod4g2)

			// Cleanup: delete pods and claims.
			b.DeletePodAndWaitForNotFound(tCtx, pod2g)
			b.DeletePodAndWaitForNotFound(tCtx, pod4g)
			b.DeletePodAndWaitForNotFound(tCtx, pod4g2)

			tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claim2g.Name, metav1.DeleteOptions{}))
			tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claim4g.Name, metav1.DeleteOptions{}))
			tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claim4g2.Name, metav1.DeleteOptions{}))
		}
	}
}

// partitionableDoTest verifies that the given running pods are still running, that the unschedulable pod
// remains unschedulable, that a new 4Gi pod is unschedulable (only 2Gi remains), and that
// a new 2Gi pod is schedulable.
func partitionableDoTest(tCtx ktesting.TContext, b *drautils.Builder, driverName string, runningPods []*v1.Pod, unschedulablePod *v1.Pod) {
	namespace := tCtx.Namespace()

	// Verify that previously running pods are still running.
	for _, pod := range runningPods {
		b.TestPod(tCtx, pod)
	}

	// Verify that the unschedulable pod is still unschedulable.
	tCtx.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(tCtx, tCtx.Client(), unschedulablePod.Name, namespace), "Pod %s should still be unschedulable", unschedulablePod.Name)

	// A new 4Gi pod should be unschedulable because only 2Gi remains.
	pod4g, claim4g := partitionableCreatePodWithClaim(tCtx, b, driverName, "4Gi")
	tCtx.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(tCtx, tCtx.Client(), pod4g.Name, namespace), "Pod with a claim over the counter should be unschedulable")
	b.DeletePodAndWaitForNotFound(tCtx, pod4g)
	tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claim4g.Name, metav1.DeleteOptions{}))

	// A new 2Gi pod should be schedulable because 2Gi is still available.
	pod2g, claim2g := partitionableCreatePodWithClaim(tCtx, b, driverName, "2Gi")
	b.TestPod(tCtx, pod2g)
	b.DeletePodAndWaitForNotFound(tCtx, pod2g)
	tCtx.ExpectNoError(tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claim2g.Name, metav1.DeleteOptions{}))
}

func partitionableCreatePodWithClaim(tCtx ktesting.TContext, b *drautils.Builder, driverName, value string) (*v1.Pod, *resourceapi.ResourceClaim) {
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

	pod := b.Pod()
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
