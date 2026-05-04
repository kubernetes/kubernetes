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
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
	"k8s.io/utils/ptr"
)

type testEnv struct {
	namespace  string
	nodeName   string
	class      *resourceapi.DeviceClass
	driverName string
	poolName   string
}

func testNodeAllocatableResources(tCtx ktesting.TContext, enabled bool) {
	if !enabled {
		return
	}
	tCtx.Run("ConsumablePool", testNodeAllocatableResourcesConsumablePool)
	tCtx.Run("IndividualDevices", testNodeAllocatableResourcesIndividualDevices)
	tCtx.Run("ClaimSharing", testNodeAllocatableResourceClaimSharing)
	tCtx.Run("PodLevelResourceValidation", testPodLevelResourceValidation)
	tCtx.Run("InsufficientNodeResources", testInsufficientNodeResources)
	tCtx.Run("ClaimTemplateBasedAllocation", testNodeAllocatableResourcesWithClaimTemplate)
	tCtx.Run("UnreferencedClaimInPod", testNodeAllocatableUnreferencedClaimInPod)
}

func setupTestEnv(tCtx ktesting.TContext, nodeNum int) *testEnv {

	nodeName := fmt.Sprintf("worker-%d", nodeNum)
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)
	poolName := namespace + "-pool"

	return &testEnv{
		namespace:  namespace,
		nodeName:   nodeName,
		class:      class,
		driverName: driverName,
		poolName:   poolName,
	}
}

func createSliceAndStartScheduler(tCtx ktesting.TContext, slice *resourceapi.ResourceSlice) {
	tCtx.Helper()
	createSlice(tCtx, slice)
	startScheduler(tCtx)
}

func makeSlice(env *testEnv, devices []resourceapi.Device) *resourceapi.ResourceSlice {
	return &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: env.namespace + "-",
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &env.nodeName,
			Pool: resourceapi.ResourcePool{
				Name:               env.poolName,
				ResourceSliceCount: 1,
			},
			Driver:  env.driverName,
			Devices: devices,
		},
	}
}

func expectPodUnschedulable(tCtx ktesting.TContext, pod *v1.Pod, reason string) {
	tCtx.Helper()
	tCtx.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(tCtx, tCtx.Client(), pod.Name, pod.Namespace), fmt.Sprintf("expected pod to be unschedulable because %q", reason))
	pod, err := tCtx.Client().CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err)
	gomega.NewWithT(tCtx).Expect(pod).To(gomega.HaveField("Status.Conditions", gomega.ContainElement(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Type":    gomega.Equal(v1.PodScheduled),
		"Status":  gomega.Equal(v1.ConditionFalse),
		"Reason":  gomega.Equal(v1.PodReasonUnschedulable),
		"Message": gomega.ContainSubstring(reason),
	}))))
}

func verifyPodNodeAllocatableStatus(tCtx ktesting.TContext, namespace, podName string, expectedStatus []v1.NodeAllocatableResourceClaimStatus) {
	tCtx.Helper()
	var statusMatchers []gtypes.GomegaMatcher
	for _, expected := range expectedStatus {
		statusMatchers = append(statusMatchers, gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"ResourceClaimName": gomega.ContainSubstring(expected.ResourceClaimName),
			"Containers":        gomega.Equal(expected.Containers),
			"Resources":         gomega.Equal(expected.Resources),
		}))
	}
	tCtx.Eventually(func(tCtx ktesting.TContext) []v1.NodeAllocatableResourceClaimStatus {
		pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, podName, metav1.GetOptions{})
		if err != nil {
			tCtx.Logf("Error getting pod: %v", err)
			return nil
		}
		return pod.Status.NodeAllocatableResourceClaimStatuses
	}).WithTimeout(30*time.Second).WithPolling(200*time.Millisecond).Should(gomega.ConsistOf(statusMatchers), "pod node allocatable resource claim status")
}

func createPodWithNodeAllocatableClaim(tCtx ktesting.TContext, env *testEnv, claimName, podName string, numContainers int) (*v1.Pod, *resourceapi.ResourceClaim) {
	tCtx.Helper()
	// Create a ResourceClaim for the node allocatable resource class
	claim := st.MakeResourceClaim().
		Name(claimName).
		Namespace(env.namespace).
		Request(env.class.Name).
		Obj()
	claim = createClaim(tCtx, env.namespace, "", env.class, claim)

	// Create a Pod that uses the node allocatable claim
	pod := st.MakePod().Name(podName).Namespace(env.namespace).Obj()

	containers := make([]v1.Container, 0, numContainers)
	for i := range numContainers {
		containers = append(containers, v1.Container{
			Name:  fmt.Sprintf("my-container-%d", i+1),
			Image: "test-image",
			Resources: v1.ResourceRequirements{
				Claims: []v1.ResourceClaim{
					{Name: claimName},
				},
			},
		})
	}
	pod.Spec.Containers = containers
	pod.Spec.NodeSelector = map[string]string{"kubernetes.io/hostname": env.nodeName}

	// createPod will generate the pod.Spec.ResourceClaims entry
	pod = createPod(tCtx, env.namespace, "", pod, claim)
	return pod, claim
}

func testNodeAllocatableResourcesConsumablePool(tCtx ktesting.TContext) {
	tCtx.Parallel()
	env := setupTestEnv(tCtx, 0)

	cpuCapacityKey := resourceapi.QualifiedName("dra.example.com/cpu")
	memCapacityKey := resourceapi.QualifiedName("dra.example.com/memory")

	devices := []resourceapi.Device{
		{
			Name:                     "node-allocatable-device-0",
			AllowMultipleAllocations: ptr.To(true),
			Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
				cpuCapacityKey: {Value: resource.MustParse(nodeCPUCapacity)},
				memCapacityKey: {Value: resource.MustParse(nodeMemoryCapacity)},
			},
			NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
				v1.ResourceCPU:    {CapacityKey: &cpuCapacityKey},
				v1.ResourceMemory: {CapacityKey: &memCapacityKey},
			},
		},
	}
	slice := makeSlice(env, devices)
	env.poolName = env.namespace + "-node-allocatable-pool" // Override poolName for this test
	slice.Spec.Pool.Name = env.poolName
	createSliceAndStartScheduler(tCtx, slice)

	podName := "test-pod-consumable-claim"
	claimName := "node-allocatable-claim-consumable"
	pod, claim := createPodWithNodeAllocatableClaim(tCtx, env, claimName, podName, 2)

	waitForPodScheduled(tCtx, env.namespace, pod.Name)
	allocatedClaim := waitForClaimAllocatedToDevice(tCtx, env.namespace, claim.Name, schedulingTimeout)

	gomega.NewWithT(tCtx).Expect(allocatedClaim).To(
		gomega.HaveField("Status.Allocation", gstruct.PointTo(
			gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Devices": gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Results": gomega.ConsistOf(
						gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
							"Request": gomega.Equal(claim.Spec.Devices.Requests[0].Name),
							"Driver":  gomega.Equal(env.driverName),
							"Pool":    gomega.Equal(env.poolName),
							"Device":  gomega.Equal("node-allocatable-device-0"),
							"ConsumedCapacity": gomega.Equal(map[resourceapi.QualifiedName]resource.Quantity{
								cpuCapacityKey: resource.MustParse(nodeCPUCapacity),
								memCapacityKey: resource.MustParse(nodeMemoryCapacity),
							}),
							"ShareID": gomega.Not(gomega.BeNil()),
						}),
					),
				}),
			}),
		)),
		"node allocatable claim allocation",
	)

	expectedStatus := []v1.NodeAllocatableResourceClaimStatus{{
		ResourceClaimName: claim.Name,
		Containers:        []string{"my-container-1", "my-container-2"},
		Resources: map[v1.ResourceName]resource.Quantity{
			v1.ResourceCPU:    resource.MustParse(nodeCPUCapacity),
			v1.ResourceMemory: resource.MustParse(nodeMemoryCapacity),
		},
	}}
	verifyPodNodeAllocatableStatus(tCtx, env.namespace, pod.Name, expectedStatus)

	anotherPod := st.MakePod().Name("another-pod").Namespace(env.namespace).Obj()
	anotherPod.Spec.NodeSelector = map[string]string{"kubernetes.io/hostname": env.nodeName}
	anotherPod.Spec.Containers = []v1.Container{
		{
			Name:  "test-container",
			Image: "test-image",
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")},
			},
		},
	}
	anotherPod = createPod(tCtx, env.namespace, "", anotherPod)
	expectPodUnschedulable(tCtx, anotherPod, "Insufficient cpu")
}

func testNodeAllocatableResourcesIndividualDevices(tCtx ktesting.TContext) {
	tCtx.Parallel()
	env := setupTestEnv(tCtx, 1)

	numCPUsPerDevice := resource.MustParse("16")
	devices := []resourceapi.Device{
		{
			Name: "numa-0-cpus",
			NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
				v1.ResourceCPU: {AllocationMultiplier: &numCPUsPerDevice},
			},
		},
		{
			Name: "numa-1-cpus",
			NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
				v1.ResourceCPU: {AllocationMultiplier: &numCPUsPerDevice},
			},
		},
	}
	slice := makeSlice(env, devices)
	env.poolName = env.namespace + "-node-allocatable-pool" // Override poolName for this test
	slice.Spec.Pool.Name = env.poolName
	createSliceAndStartScheduler(tCtx, slice)

	podName := "test-pod-individual"
	claimName := "node-allocatable-claim-individual"
	pod, claim := createPodWithNodeAllocatableClaim(tCtx, env, claimName, podName, 2)

	waitForPodScheduled(tCtx, env.namespace, pod.Name)
	allocatedClaim := waitForClaimAllocatedToDevice(tCtx, env.namespace, claim.Name, schedulingTimeout)

	gomega.NewWithT(tCtx).Expect(allocatedClaim).To(
		gomega.HaveField("Status.Allocation", gstruct.PointTo(
			gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Devices": gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Results": gomega.ConsistOf(
						gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
							"Request": gomega.Equal(claim.Spec.Devices.Requests[0].Name),
							"Driver":  gomega.Equal(env.driverName),
							"Pool":    gomega.Equal(env.poolName),
							"Device":  gomega.MatchRegexp("numa-[0-1]+-cpus"),
						}),
					),
				}),
			}),
		)),
		"node allocatable claim allocation",
	)

	expectedStatus := []v1.NodeAllocatableResourceClaimStatus{{
		ResourceClaimName: claim.Name,
		Containers:        []string{"my-container-1", "my-container-2"},
		Resources: map[v1.ResourceName]resource.Quantity{
			v1.ResourceCPU: numCPUsPerDevice,
		},
	}}
	verifyPodNodeAllocatableStatus(tCtx, env.namespace, pod.Name, expectedStatus)
}

func testNodeAllocatableResourceClaimSharing(tCtx ktesting.TContext) {
	tCtx.Parallel()
	env := setupTestEnv(tCtx, 2)

	cpuMultiplier := resource.MustParse("1")
	devices := []resourceapi.Device{
		{
			Name: "dev-sharetest",
			NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
				v1.ResourceCPU: {AllocationMultiplier: &cpuMultiplier},
			},
		},
	}
	createSliceAndStartScheduler(tCtx, makeSlice(env, devices))

	pod1, claim := createPodWithNodeAllocatableClaim(tCtx, env, "node-allocatable-claim", "pod1", 2)
	waitForPodScheduled(tCtx, env.namespace, pod1.Name)
	_ = waitForClaimAllocatedToDevice(tCtx, env.namespace, claim.Name, schedulingTimeout)

	// Pod 2 - Should NOT schedule as the claim is already used on this node
	pod2 := st.MakePod().Name("pod2").Namespace(env.namespace).Obj()
	container1 := v1.Container{
		Name:  "c1",
		Image: "test-image",
		Resources: v1.ResourceRequirements{
			Claims: []v1.ResourceClaim{
				{Name: claim.Name}, // USE EXISTING CLAIM
			},
		},
	}
	pod2.Spec.Containers = []v1.Container{container1}
	pod2.Spec.NodeSelector = map[string]string{"kubernetes.io/hostname": env.nodeName}
	pod2 = createPod(tCtx, env.namespace, "", pod2, claim) // Pass the existing claim
	expectPodUnschedulable(tCtx, pod2, "is already used by another pod")
}

func testPodLevelResourceValidation(tCtx ktesting.TContext) {
	tCtx.Parallel()
	env := setupTestEnv(tCtx, 3)

	cpuMultiplier := resource.MustParse("4")
	devices := []resourceapi.Device{
		{
			Name: "dev0",
			NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
				v1.ResourceCPU: {AllocationMultiplier: &cpuMultiplier},
			},
		},
		{
			Name: "dev1",
			NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
				v1.ResourceCPU: {AllocationMultiplier: &cpuMultiplier},
			},
		},
	}
	createSliceAndStartScheduler(tCtx, makeSlice(env, devices))

	tCtx.Run("SufficientPodLevelRequest", func(tCtx ktesting.TContext) {
		claimName := "claim-podlevel-sufficient"
		podName := "pod-podlevel-sufficient"
		claim := st.MakeResourceClaim().
			Name(claimName).
			Namespace(env.namespace).
			Request(env.class.Name).
			Obj()
		claim = createClaim(tCtx, env.namespace, "", env.class, claim)

		pod := st.MakePod().Name(podName).Namespace(env.namespace).Obj()
		pod.Spec.Resources = &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("10"), // 2 (container) + 4 (DRA) < 10
			},
		}
		container1 := v1.Container{
			Name:  "c1",
			Image: "test-image",
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("2"),
				},
				Claims: []v1.ResourceClaim{
					{Name: claimName},
				},
			},
		}
		pod.Spec.Containers = []v1.Container{container1}
		pod.Spec.NodeSelector = map[string]string{"kubernetes.io/hostname": env.nodeName}
		pod = createPod(tCtx, env.namespace, "", pod, claim)

		waitForPodScheduled(tCtx, env.namespace, pod.Name)
	})

	tCtx.Run("InsufficientPodLevelRequest", func(tCtx ktesting.TContext) {
		claimName := "claim-podlevel-insufficient"
		podName := "pod-podlevel-insufficient"
		claim := st.MakeResourceClaim().
			Name(claimName).
			Namespace(env.namespace).
			Request(env.class.Name).
			Obj()
		claim = createClaim(tCtx, env.namespace, "", env.class, claim)

		pod := st.MakePod().Name(podName).Namespace(env.namespace).Obj()
		pod.Spec.Resources = &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("5"), // 2 (container) + 4 (DRA) > 5
			},
		}
		container1 := v1.Container{
			Name:  "c1",
			Image: "test-image",
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("2"),
				},
				Claims: []v1.ResourceClaim{
					{Name: claimName},
				},
			},
		}
		pod.Spec.Containers = []v1.Container{container1}
		pod.Spec.NodeSelector = map[string]string{"kubernetes.io/hostname": env.nodeName}
		pod = createPod(tCtx, env.namespace, "", pod, claim)

		expectPodUnschedulable(tCtx, pod, "pod level request for cpu is insufficient to cover the aggregated container and node-allocatable DRA requests")
	})
}

func testInsufficientNodeResources(tCtx ktesting.TContext) {
	tCtx.Parallel()
	env := setupTestEnv(tCtx, 4)

	tCtx.Run("ExceedNodeResourceCapacityMultiplier", func(tCtx ktesting.TContext) {
		cpuMultiplier := resource.MustParse(nodeCPUCapacity)
		cpuMultiplier.Add(resource.MustParse("1"))

		devices := []resourceapi.Device{
			{
				Name: "dev-exceed-cpu",
				NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
					v1.ResourceCPU: {AllocationMultiplier: &cpuMultiplier},
				},
			},
		}
		createSliceAndStartScheduler(tCtx, makeSlice(env, devices))

		pod, _ := createPodWithNodeAllocatableClaim(tCtx, env, "claim-exceed-cpu", "pod-exceed-cpu", 1)
		expectPodUnschedulable(tCtx, pod, "Insufficient cpu")
	})

	tCtx.Run("ExceedNodeCapacityWithCapacityKey", func(tCtx ktesting.TContext) {
		cpuCapacityKey := resourceapi.QualifiedName("dra.example.com/cpu")
		exceedCPU := resource.MustParse(nodeCPUCapacity)
		exceedCPU.Add(resource.MustParse("1"))

		devices := []resourceapi.Device{
			{
				Name: "dev-exceed-cpu-capkey",
				Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					cpuCapacityKey: {Value: exceedCPU},
				},
				NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
					v1.ResourceCPU: {CapacityKey: &cpuCapacityKey},
				},
				AllowMultipleAllocations: ptr.To(true),
			},
		}
		createSliceAndStartScheduler(tCtx, makeSlice(env, devices))

		pod, _ := createPodWithNodeAllocatableClaim(tCtx, env, "claim-exceed-cpu-capkey", "pod-exceed-cpu-capkey", 1)
		expectPodUnschedulable(tCtx, pod, "Insufficient cpu")
	})

	tCtx.Run("StandardPodFailsDueToInsufficientResourcesAfterDRAConsumption", func(tCtx ktesting.TContext) {
		cpuMultiplier := resource.MustParse("98") // Consume most of the CPU
		devices := []resourceapi.Device{
			{
				Name: "dev-consume-most",
				NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
					v1.ResourceCPU: {AllocationMultiplier: &cpuMultiplier},
				},
			},
		}
		createSliceAndStartScheduler(tCtx, makeSlice(env, devices))

		pod1, _ := createPodWithNodeAllocatableClaim(tCtx, env, "claim-consume-most", "pod-consume-most", 1)
		waitForPodScheduled(tCtx, env.namespace, pod1.Name)
		waitForClaimAllocatedToDevice(tCtx, env.namespace, "claim-consume-most", schedulingTimeout)

		pod2 := st.MakePod().Name("pod-standard-fails").Namespace(env.namespace).Obj()
		pod2.Spec.Containers = []v1.Container{
			{
				Name:  "c2",
				Image: "test-image",
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("5"),
					},
				},
			},
		}
		pod2.Spec.NodeSelector = map[string]string{"kubernetes.io/hostname": env.nodeName}
		pod2 = createPod(tCtx, env.namespace, "-std-fails", pod2)
		expectPodUnschedulable(tCtx, pod2, "Insufficient cpu")
	})

	tCtx.Run("StandardPodSucceedsAfterDRA", func(tCtx ktesting.TContext) {
		cpuMultiplier := resource.MustParse("50") // Consume half of the CPU
		devices := []resourceapi.Device{
			{
				Name: "dev-consume-half",
				NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
					v1.ResourceCPU: {AllocationMultiplier: &cpuMultiplier},
				},
			},
		}
		createSliceAndStartScheduler(tCtx, makeSlice(env, devices))

		pod1, _ := createPodWithNodeAllocatableClaim(tCtx, env, "claim-consume-half", "pod-consume-half", 1)
		waitForPodScheduled(tCtx, env.namespace, pod1.Name)
		waitForClaimAllocatedToDevice(tCtx, env.namespace, "claim-consume-half", schedulingTimeout)

		pod2 := st.MakePod().Name("pod-standard-succeeds").Namespace(env.namespace).Obj()
		pod2.Spec.Containers = []v1.Container{
			{
				Name:  "c2",
				Image: "test-image",
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("5"),
					},
				},
			},
		}
		pod2.Spec.NodeSelector = map[string]string{"kubernetes.io/hostname": env.nodeName}
		pod2 = createPod(tCtx, env.namespace, "-std-succeeds", pod2)
		waitForPodScheduled(tCtx, env.namespace, pod2.Name)
	})
}

func testNodeAllocatableResourcesWithClaimTemplate(tCtx ktesting.TContext) {
	tCtx.Parallel()
	env := setupTestEnv(tCtx, 6) // Use a different node index

	cpuCapacityKey := resourceapi.QualifiedName("dra.example.com/cpu")
	memCapacityKey := resourceapi.QualifiedName("dra.example.com/memory")

	devices := []resourceapi.Device{
		{
			Name:                     "dev0",
			AllowMultipleAllocations: ptr.To(true),
			Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
				cpuCapacityKey: {Value: resource.MustParse(nodeCPUCapacity)},
				memCapacityKey: {Value: resource.MustParse(nodeMemoryCapacity)},
			},
			NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
				v1.ResourceCPU:    {CapacityKey: &cpuCapacityKey},
				v1.ResourceMemory: {CapacityKey: &memCapacityKey},
			},
		},
	}
	slice := makeSlice(env, devices)
	env.poolName = env.namespace + "-pool-template"
	slice.Spec.Pool.Name = env.poolName
	createSliceAndStartScheduler(tCtx, slice)
	startClaimController(tCtx)

	templateName := "node-allocatable-resource-template"
	podClaimName := "claim1"
	podName := "pod-with-claim-template"

	claimTemplate := &resourceapi.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      templateName,
			Namespace: env.namespace,
		},
		Spec: resourceapi.ResourceClaimTemplateSpec{
			Spec: resourceapi.ResourceClaimSpec{
				Devices: resourceapi.DeviceClaim{
					Requests: []resourceapi.DeviceRequest{
						{
							Name: "req",
							Exactly: &resourceapi.ExactDeviceRequest{
								DeviceClassName: env.class.Name,
								Capacity: &resourceapi.CapacityRequirements{
									Requests: map[resourceapi.QualifiedName]resource.Quantity{
										cpuCapacityKey: resource.MustParse("10"),
										memCapacityKey: resource.MustParse("100"),
									},
								},
							},
						},
					},
				},
			},
		},
	}
	_, err := tCtx.Client().ResourceV1().ResourceClaimTemplates(env.namespace).Create(tCtx, claimTemplate, metav1.CreateOptions{})
	tCtx.ExpectNoError(err)

	claimTemplateRef := v1.PodResourceClaim{
		Name:                      podClaimName,
		ResourceClaimTemplateName: &claimTemplate.Name,
	}

	pod := st.MakePod().Name(podName).Namespace(env.namespace).PodResourceClaims(claimTemplateRef).
		Containers([]v1.Container{{Name: "c1", Image: "test", Resources: v1.ResourceRequirements{Claims: []v1.ResourceClaim{{Name: podClaimName}}}}}).
		NodeSelector(map[string]string{"kubernetes.io/hostname": env.nodeName}).
		Obj()
	pod = createPod(tCtx, env.namespace, "", pod)

	waitForPodScheduled(tCtx, env.namespace, pod.Name)

	expectedStatus := []v1.NodeAllocatableResourceClaimStatus{{
		ResourceClaimName: podName, // The genereate claim based on template contains pod name
		Containers:        []string{"c1"},
		Resources: map[v1.ResourceName]resource.Quantity{
			v1.ResourceCPU:    resource.MustParse("10"),
			v1.ResourceMemory: resource.MustParse("100"),
		},
	}}
	verifyPodNodeAllocatableStatus(tCtx, env.namespace, pod.Name, expectedStatus)
}

func testNodeAllocatableUnreferencedClaimInPod(tCtx ktesting.TContext) {
	tCtx.Parallel()
	env := setupTestEnv(tCtx, 7) // Use a different node index

	cpuCapacityKey := resourceapi.QualifiedName("dra.example.com/cpu")
	devices := []resourceapi.Device{
		{
			Name:                     "dev0",
			AllowMultipleAllocations: ptr.To(true),
			Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
				cpuCapacityKey: {Value: resource.MustParse(nodeCPUCapacity)},
			},
			NodeAllocatableResourceMappings: map[v1.ResourceName]resourceapi.NodeAllocatableResourceMapping{
				v1.ResourceCPU: {CapacityKey: &cpuCapacityKey},
			},
		},
	}
	slice := makeSlice(env, devices)
	env.poolName = env.namespace + "-pool-unused"
	slice.Spec.Pool.Name = env.poolName
	createSliceAndStartScheduler(tCtx, slice)
	startClaimController(tCtx)

	claimName := "unused-claim"
	podName := "pod-with-unused-claim"

	// Create a ResourceClaim for the node allocatable resource class
	claim := st.MakeResourceClaim().
		Name(claimName).
		Namespace(env.namespace).
		Request(env.class.Name).
		Obj()
	claim = createClaim(tCtx, env.namespace, "", env.class, claim)

	// Create a Pod that has the node allocatable claim in spec.ResourceClaims, but no container uses it.
	pod := st.MakePod().Name(podName).Namespace(env.namespace).
		Containers([]v1.Container{{Name: "c1", Image: "test"}}).
		NodeSelector(map[string]string{"kubernetes.io/hostname": env.nodeName}).
		Obj()
	pod = createPod(tCtx, env.namespace, "", pod, claim)

	expectedErrorMsg := "node-allocatable resource claim not referenced by any container within the pod"
	expectPodUnschedulable(tCtx, pod, expectedErrorMsg)
}
