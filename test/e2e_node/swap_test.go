/*
Copyright 2023 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"fmt"
	"path/filepath"
	"strconv"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	testutils "k8s.io/kubernetes/test/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	cgroupBasePath        = "/sys/fs/cgroup/"
	cgroupV1SwapLimitFile = "/memory/memory.memsw.limit_in_bytes"
	cgroupV2SwapLimitFile = "memory.swap.max"
	cgroupV1MemLimitFile  = "/memory/memory.limit_in_bytes"
)

var _ = SIGDescribe("Swap", "[LinuxOnly]", nodefeature.Swap, func() {
	f := framework.NewDefaultFramework("swap-qos")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	ginkgo.Context("", func() {
		// Note that memoryRequestEqualLimit is effective only when qosClass is PodQOSBestEffort.

		ginkgo.DescribeTable("with configuration", func(qosClass v1.PodQOSClass, memoryRequestEqualLimit bool) {
			ginkgo.By(fmt.Sprintf("Creating a pod of QOS class %s. memoryRequestEqualLimit: %t", qosClass, memoryRequestEqualLimit))
			pod := getSwapTestPod(f.Namespace.Name, qosClass, memoryRequestEqualLimit)
			pod = runPodAndWaitUntilScheduled(f, pod)

			isCgroupV2 := isPodCgroupV2(f, pod)
			isLimitedSwap := isLimitedSwap()

			gomega.Expect(isCgroupV2).To(gomega.BeTrueBecause("cgroup v2 is required for swap"))
			gomega.Expect(isSwapFeatureGateEnabled()).To(gomega.BeTrueBecause("NodeSwap feature should be on"))
			gomega.Expect(isLimitedSwap).To(gomega.BeFalseBecause("expecting unlimited swap"))
			ginkgo.By("expecting unlimited swap")
			expectUnlimitedSwap(f, pod, isCgroupV2)
		},
			ginkgo.Entry("QOS Best-effort", v1.PodQOSBestEffort, false),
			ginkgo.Entry("QOS Burstable", v1.PodQOSBurstable, false),
			ginkgo.Entry("QOS Burstable with memory request equals to limit", v1.PodQOSBurstable, true),
			ginkgo.Entry("QOS Guaranteed", v1.PodQOSGuaranteed, false),
		)
	})
})

var _ = SIGDescribe("Swap", "[LinuxOnly]", nodefeature.Swap, framework.WithSerial(), func() {
	// These tests assume the following, and will fail otherwise:
	// - The node is provisioned with swap
	// - The node is configured with cgroup v2
	// - The swap feature gate is enabled
	// - The node has no more than 15GB of memory
	var (
		nodeName        string
		nodeTotalMemory *resource.Quantity
		nodeUsedMemory  *resource.Quantity
		swapCapacity    *resource.Quantity
		podClient       *e2epod.PodClient
	)
	f := framework.NewDefaultFramework("swap-serial-tests")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	ginkgo.BeforeEach(func() {
		gomega.Expect(isSwapFeatureGateEnabled()).To(gomega.BeTrueBecause("swap feature gate is not enabled"))
		podClient = e2epod.NewPodClient(f)

		sleepingPod := getSleepingPod(f.Namespace.Name)
		sleepingPod = runPodAndWaitUntilScheduled(f, sleepingPod)

		gomega.Expect(isPodCgroupV2(f, sleepingPod)).To(gomega.BeTrueBecause("node uses cgroup v1"))

		nodeName = sleepingPod.Spec.NodeName
		gomega.Expect(nodeName).ToNot(gomega.BeEmpty(), "node name is empty")

		swapCapacity = getSwapCapacity(f, sleepingPod)
		gomega.Expect(swapCapacity.IsZero()).To(gomega.BeFalseBecause("node swap capacity is zero"))

		err := podClient.Delete(context.Background(), sleepingPod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		nodeTotalMemory, nodeUsedMemory = getMemoryCapacity(f, nodeName)
		gomega.Expect(nodeTotalMemory.IsZero()).To(gomega.BeFalseBecause("node memory capacity is zero"))
		gomega.Expect(nodeUsedMemory.IsZero()).To(gomega.BeFalseBecause("node used memory is zero"))

		ginkgo.By(fmt.Sprintf("Setting node values. nodeName: %s, nodeTotalMemory: %d, nodeUsedMemory: %d, swapCapacity: %d",
			nodeName, nodeTotalMemory.Value(), nodeUsedMemory.Value(), swapCapacity.Value()))
	})

	ginkgo.Context("LimitedSwap", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.EvictionHard = map[string]string{string(evictionapi.SignalMemoryAvailable): "0%", string(evictionapi.SignalAllocatableMemoryAvailable): "0%"}
			initialConfig.EvictionSoft = map[string]string{string(evictionapi.SignalMemoryAvailable): "0%", string(evictionapi.SignalAllocatableMemoryAvailable): "0%"}
			initialConfig.EvictionMinimumReclaim = map[string]string{}

			initialConfig.MemorySwap.SwapBehavior = types.LimitedSwap
		})

		ginkgo.It("should report memory.swap.max is not set to max with limited swap", func() {
			podClient = e2epod.NewPodClient(f)

			swapPod := getSwapTestPod(f.Namespace.Name, v1.PodQOSBurstable, false)
			swapPod = runPodAndWaitUntilScheduled(f, swapPod)

			expectedSwapLimit := calcSwapForBurstablePod(f, swapPod)
			expectLimitedSwap(f, swapPod, expectedSwapLimit)

		})
	})
})

func getSleepingPod(namespace string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "sleeping-test-pod-swap-" + rand.String(5),
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyAlways,
			Containers: []v1.Container{
				{
					Name:    "busybox-container",
					Image:   busyboxImage,
					Command: []string{"sleep", "600"},
				},
			},
		},
	}
}

func expectPodRunningReady(pod *v1.Pod) {
	const expectOffset = 1

	isReady, err := testutils.PodRunningReady(pod)
	gomega.ExpectWithOffset(expectOffset, err).ToNot(gomega.HaveOccurred())
	gomega.ExpectWithOffset(expectOffset, isReady).To(gomega.BeTrueBecause("pod should be ready"))
}

func runPodAndWaitUntilScheduled(f *framework.Framework, pod *v1.Pod) *v1.Pod {
	ginkgo.By("running swap test pod")
	podClient := e2epod.NewPodClient(f)
	pod = podClient.CreateSync(context.Background(), pod)
	expectPodRunningReady(pod)

	return pod
}

func isSwapFeatureGateEnabled() bool {
	ginkgo.By("figuring if NodeSwap feature gate is turned on")
	return utilfeature.DefaultFeatureGate.Enabled(features.NodeSwap)
}

func readCgroupFile(f *framework.Framework, pod *v1.Pod, filename string) string {
	filePath := filepath.Join(cgroupBasePath, filename)

	ginkgo.By("reading cgroup file " + filePath)
	output := e2epod.ExecCommandInContainer(f, pod.Name, pod.Spec.Containers[0].Name, "/bin/sh", "-ec", "cat "+filePath)

	return output
}

func isPodCgroupV2(f *framework.Framework, pod *v1.Pod) bool {
	ginkgo.By("figuring is test pod runs cgroup v2")
	output := e2epod.ExecCommandInContainer(f, pod.Name, pod.Spec.Containers[0].Name, "/bin/sh", "-ec", `if test -f "/sys/fs/cgroup/cgroup.controllers"; then echo "true"; else echo "false"; fi`)

	return output == "true"
}

func expectUnlimitedSwap(f *framework.Framework, pod *v1.Pod, isCgroupV2 bool) {
	if isCgroupV2 {
		swapLimit := readCgroupFile(f, pod, cgroupV2SwapLimitFile)
		gomega.ExpectWithOffset(1, swapLimit).To(gomega.Equal("max"), "max swap allowed should be \"max\"")
	} else {
		swapPlusMemLimit := readCgroupFile(f, pod, cgroupV1SwapLimitFile)
		gomega.ExpectWithOffset(1, swapPlusMemLimit).To(gomega.Equal("-1"))
	}
}

// supports v2 only as v1 shouldn't support LimitedSwap
func expectLimitedSwap(f *framework.Framework, pod *v1.Pod, expectedSwapLimit int64) {
	swapLimitStr := readCgroupFile(f, pod, cgroupV2SwapLimitFile)

	swapLimit, err := strconv.Atoi(swapLimitStr)
	framework.ExpectNoError(err, "cannot convert swap limit to int")

	// cgroup values are always aligned w.r.t. the page size, which is usually 4Ki
	const cgroupAlignment int64 = 4 * 1024 // 4Ki
	const errMsg = "swap limitation is not as expected"

	gomega.ExpectWithOffset(1, int64(swapLimit)).To(
		gomega.Or(
			gomega.BeNumerically(">=", expectedSwapLimit-cgroupAlignment),
			gomega.BeNumerically("<=", expectedSwapLimit+cgroupAlignment),
		),
		errMsg,
	)
}

func getSwapCapacity(f *framework.Framework, pod *v1.Pod) *resource.Quantity {
	output := e2epod.ExecCommandInContainer(f, pod.Name, pod.Spec.Containers[0].Name, "/bin/sh", "-ec", "free -b | grep Swap | xargs | cut -d\" \" -f2")

	swapCapacityBytes, err := strconv.Atoi(output)
	framework.ExpectNoError(err, "cannot convert swap size to int")

	ginkgo.By(fmt.Sprintf("providing swap capacity: %d", swapCapacityBytes))

	return resource.NewQuantity(int64(swapCapacityBytes), resource.BinarySI)
}

func getMemoryCapacity(f *framework.Framework, nodeName string) (memCapacity, usedMemory *resource.Quantity) {
	node, err := f.ClientSet.CoreV1().Nodes().Get(context.Background(), nodeName, metav1.GetOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("failed getting node %s", nodeName))

	nodeOrigCapacity := node.Status.Capacity[v1.ResourceMemory]
	memCapacity = cloneQuantity(nodeOrigCapacity)
	usedMemory = cloneQuantity(nodeOrigCapacity)

	usedMemory.Sub(node.Status.Allocatable[v1.ResourceMemory])
	return
}

func calcSwapForBurstablePod(f *framework.Framework, pod *v1.Pod) int64 {
	gomega.Expect(pod.Spec.NodeName).ToNot(gomega.BeEmpty(), "pod node name is empty")

	nodeMemoryCapacityQuantity, _ := getMemoryCapacity(f, pod.Spec.NodeName)
	nodeMemoryCapacity := nodeMemoryCapacityQuantity.Value()
	nodeSwapCapacity := getSwapCapacity(f, pod).Value()
	containerMemoryRequest := pod.Spec.Containers[0].Resources.Requests.Memory().Value()

	containerMemoryProportion := float64(containerMemoryRequest) / float64(nodeMemoryCapacity)
	swapAllocation := containerMemoryProportion * float64(nodeSwapCapacity)
	ginkgo.By(fmt.Sprintf("Calculating swap for burstable pods: nodeMemoryCapacity: %d, nodeSwapCapacity: %d, containerMemoryRequest: %d, swapAllocation: %d",
		nodeMemoryCapacity, nodeSwapCapacity, containerMemoryRequest, int64(swapAllocation)))

	return int64(swapAllocation)
}

func isLimitedSwap() bool {
	kubeletCfg, err := getCurrentKubeletConfig(context.Background())
	framework.ExpectNoError(err, "cannot get kubelet config")

	return kubeletCfg.MemorySwap.SwapBehavior == types.LimitedSwap
}

func cloneQuantity(resource resource.Quantity) *resource.Quantity {
	clone := resource.DeepCopy()
	return &clone
}

func getSwapTestPod(name string, qosClass v1.PodQOSClass, memoryRequestEqualLimit bool) *v1.Pod {
	podMemoryAmount := resource.MustParse("128Mi")

	var resources v1.ResourceRequirements
	switch qosClass {
	case v1.PodQOSBestEffort:
		// nothing to do in this case
	case v1.PodQOSBurstable:
		resources = v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceMemory: podMemoryAmount,
			},
		}

		if memoryRequestEqualLimit {
			resources.Limits = resources.Requests
		}
	case v1.PodQOSGuaranteed:
		resources = v1.ResourceRequirements{
			Limits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("200m"),
				v1.ResourceMemory: podMemoryAmount,
			},
		}
		resources.Requests = resources.Limits
	}

	pod := getSleepingPod(name)
	pod.Spec.Containers[0].Resources = resources

	return pod
}
