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
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	cgroupBasePath        = "/sys/fs/cgroup/"
	cgroupV1SwapLimitFile = "/memory/memory.memsw.limit_in_bytes"
	cgroupV2SwapLimitFile = "memory.swap.max"
	cgroupV1MemLimitFile  = "/memory/memory.limit_in_bytes"
)

var _ = SIGDescribe("Swap", framework.WithNodeConformance(), "[LinuxOnly]", func() {
	f := framework.NewDefaultFramework("swap-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	ginkgo.DescribeTable("with configuration", func(qosClass v1.PodQOSClass, memoryRequestEqualLimit bool) {
		ginkgo.By(fmt.Sprintf("Creating a pod of QOS class %s. memoryRequestEqualLimit: %t", qosClass, memoryRequestEqualLimit))
		pod := getSwapTestPod(f, qosClass, memoryRequestEqualLimit)
		pod = runPodAndWaitUntilScheduled(f, pod)

		isCgroupV2 := isPodCgroupV2(f, pod)
		isLimitedSwap := isLimitedSwap(f, pod)

		if !isSwapFeatureGateEnabled() || !isCgroupV2 || (isLimitedSwap && (qosClass != v1.PodQOSBurstable || memoryRequestEqualLimit)) {
			ginkgo.By(fmt.Sprintf("Expecting no swap. feature gate on? %t isCgroupV2? %t is QoS burstable? %t", isSwapFeatureGateEnabled(), isCgroupV2, qosClass == v1.PodQOSBurstable))
			expectNoSwap(f, pod, isCgroupV2)
			return
		}

		if !isLimitedSwap {
			ginkgo.By("expecting unlimited swap")
			expectUnlimitedSwap(f, pod, isCgroupV2)
			return
		}

		ginkgo.By("expecting limited swap")
		expectedSwapLimit := calcSwapForBurstablePod(f, pod)
		expectLimitedSwap(f, pod, expectedSwapLimit)
	},
		ginkgo.Entry("QOS Best-effort", v1.PodQOSBestEffort, false),
		ginkgo.Entry("QOS Burstable", v1.PodQOSBurstable, false),
		ginkgo.Entry("QOS Burstable with memory request equals to limit", v1.PodQOSBurstable, true),
		ginkgo.Entry("QOS Guaranteed", v1.PodQOSGuaranteed, false),
	)
})

// Note that memoryRequestEqualLimit is effective only when qosClass is PodQOSBestEffort.
func getSwapTestPod(f *framework.Framework, qosClass v1.PodQOSClass, memoryRequestEqualLimit bool) *v1.Pod {
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

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod-swap-" + rand.String(5),
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyAlways,
			Containers: []v1.Container{
				{
					Name:      "busybox-container",
					Image:     busyboxImage,
					Command:   []string{"sleep", "600"},
					Resources: resources,
				},
			},
		},
	}

	return pod
}

func runPodAndWaitUntilScheduled(f *framework.Framework, pod *v1.Pod) *v1.Pod {
	ginkgo.By("running swap test pod")
	podClient := e2epod.NewPodClient(f)

	pod = podClient.CreateSync(context.Background(), pod)
	pod, err := podClient.Get(context.Background(), pod.Name, metav1.GetOptions{})

	framework.ExpectNoError(err)
	isReady, err := testutils.PodRunningReady(pod)
	framework.ExpectNoError(err)
	gomega.ExpectWithOffset(1, isReady).To(gomega.BeTrue(), "pod should be ready")

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

func expectNoSwap(f *framework.Framework, pod *v1.Pod, isCgroupV2 bool) {
	if isCgroupV2 {
		swapLimit := readCgroupFile(f, pod, cgroupV2SwapLimitFile)
		gomega.ExpectWithOffset(1, swapLimit).To(gomega.Equal("0"), "max swap allowed should be zero")
	} else {
		swapPlusMemLimit := readCgroupFile(f, pod, cgroupV1SwapLimitFile)
		memLimit := readCgroupFile(f, pod, cgroupV1MemLimitFile)
		gomega.ExpectWithOffset(1, swapPlusMemLimit).ToNot(gomega.BeEmpty())
		gomega.ExpectWithOffset(1, swapPlusMemLimit).To(gomega.Equal(memLimit))
	}
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

func getSwapCapacity(f *framework.Framework, pod *v1.Pod) int64 {
	output := e2epod.ExecCommandInContainer(f, pod.Name, pod.Spec.Containers[0].Name, "/bin/sh", "-ec", "free -b | grep Swap | xargs | cut -d\" \" -f2")

	swapCapacity, err := strconv.Atoi(output)
	framework.ExpectNoError(err, "cannot convert swap size to int")

	ginkgo.By(fmt.Sprintf("providing swap capacity: %d", swapCapacity))

	return int64(swapCapacity)
}

func getMemoryCapacity(f *framework.Framework, pod *v1.Pod) int64 {
	nodes, err := f.ClientSet.CoreV1().Nodes().List(context.Background(), metav1.ListOptions{})
	framework.ExpectNoError(err, "failed listing nodes")

	for _, node := range nodes.Items {
		if node.Name != pod.Spec.NodeName {
			continue
		}

		memCapacity := node.Status.Capacity[v1.ResourceMemory]
		return memCapacity.Value()
	}

	framework.ExpectNoError(fmt.Errorf("node %s wasn't found", pod.Spec.NodeName))
	return 0
}

func calcSwapForBurstablePod(f *framework.Framework, pod *v1.Pod) int64 {
	nodeMemoryCapacity := getMemoryCapacity(f, pod)
	nodeSwapCapacity := getSwapCapacity(f, pod)
	containerMemoryRequest := pod.Spec.Containers[0].Resources.Requests.Memory().Value()

	containerMemoryProportion := float64(containerMemoryRequest) / float64(nodeMemoryCapacity)
	swapAllocation := containerMemoryProportion * float64(nodeSwapCapacity)
	ginkgo.By(fmt.Sprintf("Calculating swap for burstable pods: nodeMemoryCapacity: %d, nodeSwapCapacity: %d, containerMemoryRequest: %d, swapAllocation: %d",
		nodeMemoryCapacity, nodeSwapCapacity, containerMemoryRequest, int64(swapAllocation)))

	return int64(swapAllocation)
}

func isLimitedSwap(f *framework.Framework, pod *v1.Pod) bool {
	kubeletCfg, err := getCurrentKubeletConfig(context.Background())
	framework.ExpectNoError(err, "cannot get kubelet config")

	return kubeletCfg.MemorySwap.SwapBehavior == types.LimitedSwap
}
