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
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	kubeletconfig2 "k8s.io/kubernetes/test/e2e_node/kubeletconfig"
	testutils "k8s.io/kubernetes/test/utils"
	admissionapi "k8s.io/pod-security-admission/api"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	cgroupBasePath        = "/sys/fs/cgroup/"
	cgroupV1SwapLimitFile = "/memory/memory.memsw.limit_in_bytes"
	cgroupV2SwapLimitFile = "memory.swap.max"
	cgroupV1MemLimitFile  = "/memory/memory.limit_in_bytes"
	swapUsageCgroupFile   = "memory.swap.current"
)

var _ = SIGDescribe("Swap [LinuxOnly]", func() {
	f := framework.NewDefaultFramework("swap-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	ginkgo.Context("[NodeConformance]", func() {
		// Note that memoryRequestEqualLimit is effective only when qosClass is PodQOSBestEffort.
		getSwapTestPod := func(qosClass v1.PodQOSClass, memoryRequestEqualLimit bool) *v1.Pod {
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

			pod := getSleepingPod(f.Namespace.Name)
			pod.Spec.Containers[0].Resources = resources

			return pod
		}

		ginkgo.DescribeTable("with configuration", func(qosClass v1.PodQOSClass, memoryRequestEqualLimit bool) {
			ginkgo.By(fmt.Sprintf("Creating a pod of QOS class %s. memoryRequestEqualLimit: %t", qosClass, memoryRequestEqualLimit))
			pod := getSwapTestPod(qosClass, memoryRequestEqualLimit)
			pod = runPodAndWaitUntilScheduled(f, pod)

			isCgroupV2 := isPodCgroupV2(f, pod)
			isLimitedSwap := isLimitedSwap()

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

	// These tests assume the following, and will fail otherwise:
	// - The node is provisioned with swap
	// - The node is configured with cgroup v2
	// - The swap feature gate is enabled
	// - The node has no more than 15GB of memory
	ginkgo.Context("[SwapConformance]", ginkgo.Serial, func() {
		var (
			nodeName        string
			nodeTotalMemory *resource.Quantity
			nodeUsedMemory  *resource.Quantity
			swapCapacity    *resource.Quantity
			podClient       *e2epod.PodClient
		)

		ginkgo.BeforeEach(func() {
			gomega.Expect(isSwapFeatureGateEnabled()).To(gomega.BeTrue(), "swap feature gate is not enabled")
			podClient = e2epod.NewPodClient(f)

			sleepingPod := getSleepingPod(f.Namespace.Name)
			sleepingPod = runPodAndWaitUntilScheduled(f, sleepingPod)

			gomega.Expect(isPodCgroupV2(f, sleepingPod)).To(gomega.BeTrue(), "node uses cgroup v1")

			nodeName = sleepingPod.Spec.NodeName
			gomega.Expect(nodeName).ToNot(gomega.BeEmpty(), "node name is empty")

			swapCapacity = getSwapCapacity(f, sleepingPod)
			gomega.Expect(swapCapacity.IsZero()).To(gomega.BeFalse(), "node swap capacity is zero")

			err := podClient.Delete(context.Background(), sleepingPod.Name, metav1.DeleteOptions{})
			gomega.Expect(err).ToNot(gomega.HaveOccurred())

			nodeTotalMemory, nodeUsedMemory = getMemoryCapacity(f, nodeName)
			gomega.Expect(nodeTotalMemory.IsZero()).To(gomega.BeFalse(), "node memory capacity is zero")
			gomega.Expect(nodeUsedMemory.IsZero()).To(gomega.BeFalse(), "node used memory is zero")

			ginkgo.By(fmt.Sprintf("Setting node values. nodeName: %s, nodeTotalMemory: %d, nodeUsedMemory: %d, swapCapacity: %d",
				nodeName, nodeTotalMemory.Value(), nodeUsedMemory.Value(), swapCapacity.Value()))
		})

		ginkgo.Context("LimitedSwap", func() {
			tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
				//msg := "swap behavior is already set to LimitedSwap"
				//
				//if swapBehavior := initialConfig.MemorySwap.SwapBehavior; swapBehavior != types.LimitedSwap {
				//	initialConfig.MemorySwap.SwapBehavior = types.LimitedSwap
				//	msg = "setting swap behavior to LimitedSwap"
				//}

				// Clear out the eviction config to make sure it doesn't interfere with the test.
				// TODO(iholder101): Consider changing that after evictions are explored w.r.t. swap
				//evidtionThreshold := map[string]string{"memory.available": "500Gi"}
				initialConfig.EvictionHard = map[string]string{string(evictionapi.SignalMemoryAvailable): "0%", string(evictionapi.SignalAllocatableMemoryAvailable): "0%"}
				initialConfig.EvictionSoft = map[string]string{string(evictionapi.SignalMemoryAvailable): "0%", string(evictionapi.SignalAllocatableMemoryAvailable): "0%"}
				initialConfig.EvictionMinimumReclaim = map[string]string{}

				initialConfig.MemorySwap.SwapBehavior = types.LimitedSwap

				//ginkgo.By(msg)
			})

			getStressSize := func(swapSizeToUse *resource.Quantity) *resource.Quantity {
				stressSize := cloneQuantity(*nodeTotalMemory)
				stressSize.Sub(*nodeUsedMemory)
				stressSize.Add(*swapSizeToUse)

				return stressSize
			}

			getStressPod := func(stressSizeBytes int64, limitSwapPercentage float64) *v1.Pod {
				gomega.ExpectWithOffset(1, limitSwapPercentage).To(gomega.And(gomega.BeNumerically(">=", 0.0), gomega.BeNumerically("<=", 1.0)), "limitSwapPercentage is not between 0 and 1")

				pod := getStressPod(f, stressSizeBytes)
				pod.Spec.NodeName = nodeName
				pod.Spec.Containers[0].Resources = v1.ResourceRequirements{
					Requests: v1.ResourceList{
						// With LimitedSwap, this means that the pod will be limited to a swapPercentageToUse of the node swap capacity.
						v1.ResourceMemory: *resource.NewQuantity(int64(float64(nodeTotalMemory.Value())*limitSwapPercentage), resource.BinarySI),
					},
				}

				return pod
			}

			getSwapSizeByPercentage := func(swapPercentage float64) *resource.Quantity {
				return resource.NewQuantity(int64(float64(swapCapacity.Value())*swapPercentage), resource.BinarySI)
			}

			getPageAlignedQuantities := func(quantity *resource.Quantity) (alignedDownwards, alignedUpwards *resource.Quantity) {
				alignedDownwards = cloneQuantity(*quantity)
				alignedUpwards = cloneQuantity(*quantity)

				alignedDownwards.Sub(resource.MustParse("4Ki"))
				alignedUpwards.Add(resource.MustParse("4Ki"))

				return
			}

			ginkgo.It("should be able over-commit the node memory up to the auto-calculated swap limit", func() {
				config, err := kubeletconfig2.GetCurrentKubeletConfigFromFile()
				gomega.Expect(err).ToNot(gomega.HaveOccurred())
				gomega.Expect(config.MemorySwap.SwapBehavior).To(gomega.Equal(types.LimitedSwap), "swap behavior is not LimitedSwap")

				const swapPercentageToUse = 0.2
				//var err error

				// Since cgroup sizes are page aligned, make sure we're within a page size of the expected value
				swapToUse := getSwapSizeByPercentage(swapPercentageToUse)
				minSwapSizeToUse, maxSwapSizeToUse := getPageAlignedQuantities(swapToUse)

				stressSize := getStressSize(swapToUse)
				stressPod := getStressPod(stressSize.Value(), swapPercentageToUse)

				ginkgo.By(fmt.Sprintf("creating a stress pod with stress size %d and swap percentage %g which is of size %d", stressSize.Value(), swapPercentageToUse, swapToUse.Value()))
				stressPod = runPodAndWaitUntilScheduled(f, stressPod)

				swapLimit, err := readCgroupFileFromHost(stressPod, cgroupV2SwapLimitFile)
				gomega.Expect(err).ToNot(gomega.HaveOccurred())

				gomega.Expect(swapLimit).ToNot(gomega.Equal("max"), "swap limit is not set")
				gomega.Expect(swapLimit).ToNot(gomega.Equal("0"), "swap limit is zero")

				ginkgo.By(fmt.Sprintf("Expecting the swap usage to grow to at least %d", minSwapSizeToUse.Value()))
				gomega.Eventually(func() error {
					stressPod, err = getUpdatedPod(f, stressPod)
					if err != nil {
						return err
					}

					expectPodRunningReady(stressPod)

					swapUsage, err := checkSwapUsage(stressPod)
					if err != nil {
						return err
					}

					//gomega.Expect(swapUsage.Cmp(*maxSwapSizeToUse) == 1).To(gomega.BeFalse(), "swap usage is greater than expected: %v > %v", swapUsage.Value(), maxSwapSizeToUse.Value())

					if swapUsage.Cmp(*minSwapSizeToUse) == -1 {
						return fmt.Errorf("swap usage is smaller than expected: %v < %v", swapUsage.Value(), minSwapSizeToUse.Value())
					}

					return nil
				}, 5*time.Minute, 1*time.Second).Should(gomega.Succeed(), "swap usage should be between %v and %v", minSwapSizeToUse.Value(), maxSwapSizeToUse.Value())
			})
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

func getStressPod(f *framework.Framework, stressSizeBytes int64) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "stress-pod-" + rand.String(5),
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:            "stress-container",
					Image:           "registry.k8s.io/stress:v1",
					ImagePullPolicy: v1.PullAlways,
					Args:            []string{"-mem-alloc-size", "4Mi", "-mem-alloc-sleep", "10ms", "-mem-total", strconv.Itoa(int(stressSizeBytes))},
				},
			},
		},
	}
}

func getUpdatedPod(f *framework.Framework, pod *v1.Pod) (*v1.Pod, error) {
	podClient := e2epod.NewPodClient(f)
	pod, err := podClient.Get(context.Background(), pod.Name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	return pod, nil
}

func expectPodRunningReady(pod *v1.Pod) {
	const expectOffset = 1

	isReady, err := testutils.PodRunningReady(pod)
	gomega.ExpectWithOffset(expectOffset, err).ToNot(gomega.HaveOccurred())
	gomega.ExpectWithOffset(expectOffset, isReady).To(gomega.BeTrue(), "pod should be ready")
}

func runPodAndWaitUntilScheduled(f *framework.Framework, pod *v1.Pod) *v1.Pod {
	ginkgo.By("running swap test pod")
	podClient := e2epod.NewPodClient(f)
	pod = podClient.CreateSync(context.Background(), pod)
	//pod = getUpdatedPod(f, pod)
	expectPodRunningReady(pod)

	return pod
}

func isSwapFeatureGateEnabled() bool {
	ginkgo.By("figuring if NodeSwap feature gate is turned on")
	return isFeatureGateEnabled(features.NodeSwap)
}

func readCgroupFile(f *framework.Framework, pod *v1.Pod, filename string) string {
	filePath := filepath.Join(cgroupBasePath, filename)

	ginkgo.By("reading cgroup file " + filePath)
	output := e2epod.ExecCommandInContainer(f, pod.Name, pod.Spec.Containers[0].Name, "/bin/sh", "-ec", "cat "+filePath)

	return output
}

func readCgroupFileFromHost(pod *v1.Pod, cgroupFile string) (string, error) {
	cgroupPath := getPodCgroupPath(pod)
	cgroupPath = filepath.Join(cgroupPath, cgroupFile)

	ginkgo.By("Reading file " + cgroupPath)
	cmd := "cat " + cgroupPath
	outputBytes, err := exec.Command("sudo", "sh", "-c", cmd).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("error running cmd %s: %w", cmd, err)
	}

	outputStr := strings.TrimSpace(string(outputBytes))
	return outputStr, nil
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

func getSwapCapacity(f *framework.Framework, pod *v1.Pod) *resource.Quantity {
	output := e2epod.ExecCommandInContainer(f, pod.Name, pod.Spec.Containers[0].Name, "/bin/sh", "-ec", "free -b | grep Swap | xargs | cut -d\" \" -f2")

	swapCapacityBytes, err := strconv.Atoi(output)
	framework.ExpectNoError(err, "cannot convert swap size to int")

	ginkgo.By(fmt.Sprintf("providing swap capacity: %d", swapCapacityBytes))

	return resource.NewQuantity(int64(swapCapacityBytes), resource.BinarySI)
}

func getMemoryCapacity(f *framework.Framework, nodeName string) (memCapacity, usedMemory *resource.Quantity) {
	node, err := f.ClientSet.CoreV1().Nodes().Get(context.Background(), nodeName, metav1.GetOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), fmt.Sprintf("failed getting node %s", nodeName))

	nodeOrigCapacity := node.Status.Capacity[v1.ResourceMemory]
	memCapacity = cloneQuantity(nodeOrigCapacity)
	usedMemory = cloneQuantity(nodeOrigCapacity)

	usedMemory.Sub(node.Status.Allocatable[v1.ResourceMemory])
	return
}

func getPodCgroupPath(pod *v1.Pod) string {
	podQos := qos.GetPodQOS(pod)
	cgroupQosComponent := ""

	switch podQos {
	case v1.PodQOSBestEffort:
		cgroupQosComponent = bestEffortCgroup
	case v1.PodQOSBurstable:
		cgroupQosComponent = burstableCgroup
	}

	var rootCgroupName cm.CgroupName
	if cgroupQosComponent != "" {
		rootCgroupName = cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup, cgroupQosComponent)
	} else {
		rootCgroupName = cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup)
	}

	cgroupsToVerify := "pod" + string(pod.UID)
	cgroupName := cm.NewCgroupName(rootCgroupName, cgroupsToVerify)
	cgroupFsPath := toCgroupFsName(cgroupName)

	return filepath.Join(cgroupBasePath, cgroupFsPath)
}

func checkSwapUsage(pod *v1.Pod) (*resource.Quantity, error) {
	const expectOffset = 1
	ginkgo.By("Checking swap usage for pod " + pod.Name)
	cgroupPath := getPodCgroupPath(pod)
	swapUsagePath := filepath.Join(cgroupPath, swapUsageCgroupFile)

	ginkgo.By("Reading file " + swapUsagePath)
	cmd := "cat " + swapUsagePath
	outputBytes, err := exec.Command("sudo", "sh", "-c", cmd).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("error running cmd %s: %w", cmd, err)
	}

	outputStr := strings.TrimSpace(string(outputBytes))
	ginkgo.By("swap usage found: " + outputStr + " bytes")

	swapUsage, err := strconv.ParseInt(outputStr, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("error parsing swap usage %s to int: %w", outputStr, err)
	}

	return resource.NewQuantity(swapUsage, resource.BinarySI), nil
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
