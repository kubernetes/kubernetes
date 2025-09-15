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
	"math/big"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/feature"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	cgroupBasePath                 = "/sys/fs/cgroup/"
	cgroupV2SwapLimitFile          = "memory.swap.max"
	cgroupV2swapCurrentUsageFile   = "memory.swap.current"
	cgroupV2MemoryCurrentUsageFile = "memory.current"
)

var (
	noLimits *resource.Quantity = nil
)

var _ = SIGDescribe("Swap", "[LinuxOnly]", ginkgo.Ordered, feature.Swap, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("swap-qos")
	addAfterEachForCleaningUpPods(f)
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.BeforeEach(func() {
		gomega.Expect(isSwapFeatureGateEnabled()).To(gomega.BeTrueBecause("NodeSwap feature should be on"))
	})

	f.Context(framework.WithNodeConformance(), func() {

		ginkgo.DescribeTable("with configuration", func(qosClass v1.PodQOSClass, memoryRequestEqualLimit bool) {
			ginkgo.By(fmt.Sprintf("Creating a pod of QOS class %s. memoryRequestEqualLimit: %t", qosClass, memoryRequestEqualLimit))
			pod := getSwapTestPod(f, qosClass, memoryRequestEqualLimit)
			pod = runPodAndWaitUntilScheduled(f, pod)

			if !isPodCgroupV2(f, pod) {
				e2eskipper.Skipf("swap tests require cgroup v2")
			}
			gomega.Expect(getSwapBehavior()).To(gomega.Or(gomega.Equal(string(types.NoSwap)), gomega.BeEmpty()), "NodeConformance is expected to run with NoSwap")

			expectNoSwap(f, pod)
		},
			ginkgo.Entry("QOS Best-effort", v1.PodQOSBestEffort, false),
			ginkgo.Entry("QOS Burstable", v1.PodQOSBurstable, false),
			ginkgo.Entry("QOS Burstable with memory request equals to limit", v1.PodQOSBurstable, true),
			ginkgo.Entry("QOS Guaranteed", v1.PodQOSGuaranteed, false),
		)

		ginkgo.It("with a critical pod - should avoid swap", func() {
			ginkgo.By("Creating a critical pod")
			const memoryRequestEqualLimit = false
			pod := getSwapTestPod(f, v1.PodQOSBurstable, memoryRequestEqualLimit)
			pod.Spec.PriorityClassName = "system-node-critical"

			pod = runPodAndWaitUntilScheduled(f, pod)
			gomega.Expect(types.IsCriticalPod(pod)).To(gomega.BeTrueBecause("pod should be critical"))

			ginkgo.By("expecting pod to not have swap access")
			expectNoSwap(f, pod)
		})
	})

	f.Context(framework.WithSerial(), func() {

		enableLimitedSwap := func(ctx context.Context, initialConfig *config.KubeletConfiguration) {
			msg := "swap behavior is already set to LimitedSwap"

			if swapBehavior := initialConfig.MemorySwap.SwapBehavior; swapBehavior != string(types.LimitedSwap) {
				initialConfig.MemorySwap.SwapBehavior = string(types.LimitedSwap)
				msg = "setting swap behavior to LimitedSwap"
			}

			ginkgo.By(msg)
		}

		f.Context("Basic functionality", func() {
			tempSetCurrentKubeletConfig(f, enableLimitedSwap)

			ginkgo.DescribeTable("with configuration", func(qosClass v1.PodQOSClass, memoryRequestEqualLimit bool) {
				ginkgo.By(fmt.Sprintf("Creating a pod of QOS class %s. memoryRequestEqualLimit: %t", qosClass, memoryRequestEqualLimit))
				pod := getSwapTestPod(f, qosClass, memoryRequestEqualLimit)
				pod = runPodAndWaitUntilScheduled(f, pod)

				if !isPodCgroupV2(f, pod) {
					e2eskipper.Skipf("swap tests require cgroup v2")
				}
				gomega.Expect(getSwapBehavior()).To(gomega.Equal(string(types.LimitedSwap)))

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
		f.Context("with swap stress", func() {
			var (
				nodeName           string
				nodeTotalMemory    *resource.Quantity
				nodeUsedMemory     *resource.Quantity
				swapCapacity       *resource.Quantity
				stressMemAllocSize *resource.Quantity
				podClient          *e2epod.PodClient
			)

			ginkgo.BeforeEach(func() {
				podClient = e2epod.NewPodClient(f)

				sleepingPod := getSleepingPod(f.Namespace.Name)
				sleepingPod = runPodAndWaitUntilScheduled(f, sleepingPod)
				if !isPodCgroupV2(f, sleepingPod) {
					e2eskipper.Skipf("swap tests require cgroup v2")
				}

				nodeName = sleepingPod.Spec.NodeName
				gomega.Expect(nodeName).ToNot(gomega.BeEmpty(), "node name is empty")

				nodeTotalMemory, nodeUsedMemory = getMemoryCapacity(f, nodeName)
				gomega.Expect(nodeTotalMemory.IsZero()).To(gomega.BeFalseBecause("node memory capacity is zero"))
				gomega.Expect(nodeUsedMemory.IsZero()).To(gomega.BeFalseBecause("node used memory is zero"))

				swapCapacity = getSwapCapacity(f, sleepingPod)
				if swapCapacity.IsZero() {
					e2eskipper.Skipf("swap is not provisioned on the node")
				}

				err := podClient.Delete(context.Background(), sleepingPod.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err)

				stressMemAllocSize = multiplyQuantity(divideQuantity(nodeTotalMemory, 1000), 4)

				ginkgo.By(fmt.Sprintf("Setting node values. nodeName: %s, nodeTotalMemory: %s, nodeUsedMemory: %s, swapCapacity: %s, stressMemAllocSize: %s",
					nodeName, nodeTotalMemory.String(), nodeUsedMemory.String(), swapCapacity.String(), stressMemAllocSize.String()))
			})

			ginkgo.Context("LimitedSwap", func() {
				tempSetCurrentKubeletConfig(f, enableLimitedSwap)

				getRequestBySwapLimit := func(swapPercentage int64) *resource.Quantity {
					gomega.ExpectWithOffset(1, swapPercentage).To(gomega.And(
						gomega.BeNumerically(">=", 1),
						gomega.BeNumerically("<=", 100),
					), "percentage has to be between 1 and 100")

					// <swap-limit> = <memory-request>*(<swap-capacity>/<node-capacity>).
					// if x is the percentage, and <memory-request> == (x/100)*<node-capacity>, then:
					// <swap-limit> = (x/100)*<node-capacity>*(<swap-capacity>/<node-capacity>) = (x/100)*<swap-capacity>.
					return multiplyQuantity(divideQuantity(nodeTotalMemory, 100), swapPercentage)
				}

				getStressPod := func(stressSize *resource.Quantity) *v1.Pod {
					pod := getStressPod(f, stressSize, stressMemAllocSize)
					pod.Spec.NodeName = nodeName

					return pod
				}

				ginkgo.It("should be able to use more than the node memory capacity", func() {
					stressSize := cloneQuantity(nodeTotalMemory)

					stressPod := getStressPod(stressSize)
					// Request will use a lot more swap memory than needed, since we don't test swap limits in this test
					memRequest := getRequestBySwapLimit(30)
					setPodMemoryResources(stressPod, memRequest, noLimits)
					gomega.Expect(qos.GetPodQOS(stressPod)).To(gomega.Equal(v1.PodQOSBurstable))

					ginkgo.By(fmt.Sprintf("creating a stress pod with stress size %s and request of %s", stressSize.String(), memRequest.String()))
					stressPod = runPodAndWaitUntilScheduled(f, stressPod)

					ginkgo.By("Expecting the swap usage to be non-zero")
					var swapUsage *resource.Quantity
					gomega.Eventually(func() error {
						stressPod = getUpdatedPod(f, stressPod)
						gomega.Expect(stressPod.Status.Phase).To(gomega.Equal(v1.PodRunning), "pod should be running")

						var err error
						swapUsage, err = getSwapUsage(f, stressPod)
						if err != nil {
							return err
						}

						if swapUsage.IsZero() {
							return fmt.Errorf("swap usage is zero")
						}

						return nil
					}, 5*time.Minute, 1*time.Second).Should(gomega.Succeed(), "swap usage is above zero: %s", swapUsage.String())

					// Better to delete the stress pod ASAP to avoid node failures
					err := podClient.Delete(context.Background(), stressPod.Name, metav1.DeleteOptions{})
					framework.ExpectNoError(err)
				})

				ginkgo.It("should be able to use more memory than memory limits", func() {
					stressSize := divideQuantity(nodeTotalMemory, 5)
					ginkgo.By("Creating a stress pod with stress size: " + stressSize.String())
					stressPod := getStressPod(stressSize)

					memoryLimit := cloneQuantity(stressSize)
					memoryLimit.Sub(resource.MustParse("50Mi"))
					memoryRequest := divideQuantity(memoryLimit, 2)
					ginkgo.By("Adding memory request of " + memoryRequest.String() + " and memory limit of " + memoryLimit.String())
					setPodMemoryResources(stressPod, memoryRequest, memoryLimit)
					gomega.Expect(qos.GetPodQOS(stressPod)).To(gomega.Equal(v1.PodQOSBurstable))

					var swapUsage, memoryUsage *resource.Quantity
					// This is sanity check to ensure that swap usage is not caused by a system-level pressure, but
					// due to a container-level (cgroup-level) pressure that's caused because of the memory limits.
					minExpectedMemoryUsage := multiplyQuantity(divideQuantity(memoryLimit, 4), 3)

					stressPod = runPodAndWaitUntilScheduled(f, stressPod)

					ginkgo.By("Expecting the pod exceed limits and avoid an OOM kill since it would use swap")
					gomega.Eventually(func() error {
						stressPod = getUpdatedPod(f, stressPod)
						gomega.Expect(stressPod.Status.Phase).To(gomega.Equal(v1.PodRunning), "pod should be running")

						var err error
						swapUsage, err = getSwapUsage(f, stressPod)
						if err != nil {
							return err
						}

						memoryUsage, err = getMemoryUsage(f, stressPod)
						if err != nil {
							return err
						}

						if memoryUsage.Cmp(*minExpectedMemoryUsage) == -1 {
							return fmt.Errorf("memory usage (%s) is smaller than minimum expected memory usage (%s)", memoryUsage.String(), minExpectedMemoryUsage.String())
						}
						if swapUsage.IsZero() {
							return fmt.Errorf("swap usage is zero")
						}

						return nil
					}, 5*time.Minute, 1*time.Second).Should(gomega.Succeed())

					// Better to delete the stress pod ASAP to avoid node failures
					err := podClient.Delete(context.Background(), stressPod.Name, metav1.DeleteOptions{})
					framework.ExpectNoError(err)
				})

				ginkgo.It("ensure summary API properly reports swap", func() {
					stressSize := divideQuantity(nodeTotalMemory, 5)
					ginkgo.By("Creating a stress pod with stress size: " + stressSize.String())
					stressPod := getStressPod(stressSize)

					memoryLimit := cloneQuantity(stressSize)
					memoryLimit.Sub(resource.MustParse("50Mi"))
					memoryRequest := divideQuantity(memoryLimit, 2)
					ginkgo.By("Adding memory request of " + memoryRequest.String() + " and memory limit of " + memoryLimit.String())
					setPodMemoryResources(stressPod, memoryRequest, memoryLimit)
					gomega.Expect(qos.GetPodQOS(stressPod)).To(gomega.Equal(v1.PodQOSBurstable))

					stressPod = runPodAndWaitUntilScheduled(f, stressPod)

					ginkgo.By("Ensuring the pod is using swap")
					var swapUsage *resource.Quantity
					gomega.Eventually(func() error {
						stressPod = getUpdatedPod(f, stressPod)
						gomega.Expect(stressPod.Status.Phase).To(gomega.Equal(v1.PodRunning), "pod should be running")

						var err error
						swapUsage, err = getSwapUsage(f, stressPod)
						if err != nil {
							return err
						}

						if swapUsage.IsZero() {
							return fmt.Errorf("swap usage is zero")
						}

						return nil
					}, 5*time.Minute, 1*time.Second).Should(gomega.Succeed())

					ginkgo.By("Waiting 15 seconds for cAdvisor to collect 2 stats points")
					time.Sleep(15 * time.Second)

					getSwapExpectation := func() gomega.OmegaMatcher {
						return gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
							"Time":           recent(maxStatsAge),
							"SwapUsageBytes": bounded(1, memoryLimit.Value()),
						}))
					}

					matchExpectations := gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
						"Pods": gstruct.MatchElements(summaryObjectID, gstruct.IgnoreExtras, gstruct.Elements{
							fmt.Sprintf("%s::%s", f.Namespace.Name, stressPod.Name): gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
								"Containers": gstruct.MatchElements(summaryObjectID, gstruct.IgnoreExtras, gstruct.Elements{
									stressPod.Spec.Containers[0].Name: gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
										"Swap": getSwapExpectation(),
									}),
								}),
								"Swap": getSwapExpectation(),
							}),
						}),
					}))

					ginkgo.By("Validating /stats/summary")
					// Give pods a minute to actually start up.
					gomega.Eventually(context.Background(), getNodeSummary, 180*time.Second, 15*time.Second).Should(matchExpectations)
					// Then the summary should match the expectations a few more times.
					gomega.Consistently(context.Background(), getNodeSummary, 30*time.Second, 15*time.Second).Should(matchExpectations)
				})
			})
		})
	})
})

// Note that memoryRequestEqualLimit is effective only when qosClass is not PodQOSBestEffort.
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

	pod := getSleepingPod(f.Namespace.Name)

	return pod
}

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

func getStressPod(f *framework.Framework, stressSize, memAllocSize *resource.Quantity) *v1.Pod {
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
					Image:           imageutils.GetE2EImage(imageutils.Agnhost),
					ImagePullPolicy: v1.PullAlways,
					Args:            []string{"stress", "--mem-alloc-size", memAllocSize.String(), "--mem-alloc-sleep", "1000ms", "--mem-total", strconv.Itoa(int(stressSize.Value()))},
				},
			},
		},
	}
}

func getUpdatedPod(f *framework.Framework, pod *v1.Pod) *v1.Pod {
	podClient := e2epod.NewPodClient(f)
	pod, err := podClient.Get(context.Background(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	return pod
}

func runPodAndWaitUntilScheduled(f *framework.Framework, pod *v1.Pod) *v1.Pod {
	ginkgo.By("running swap test pod")
	podClient := e2epod.NewPodClient(f)

	pod = podClient.CreateSync(context.Background(), pod)
	pod = getUpdatedPod(f, pod)

	isReady, err := testutils.PodRunningReady(pod)
	framework.ExpectNoError(err)
	gomega.ExpectWithOffset(1, isReady).To(gomega.BeTrueBecause("pod %+v was expected to be ready", pod))

	return pod
}

func isSwapFeatureGateEnabled() bool {
	ginkgo.By("figuring if NodeSwap feature gate is turned on")
	return e2eskipper.IsFeatureGateEnabled(features.NodeSwap)
}

func isPodCgroupV2(f *framework.Framework, pod *v1.Pod) bool {
	ginkgo.By("figuring is test pod runs cgroup v2")
	output := e2epod.ExecCommandInContainer(f, pod.Name, pod.Spec.Containers[0].Name, "/bin/sh", "-ec", `if test -f "/sys/fs/cgroup/cgroup.controllers"; then echo "true"; else echo "false"; fi`)

	return output == "true"
}

func expectNoSwap(f *framework.Framework, pod *v1.Pod) {
	ginkgo.By("expecting no swap")
	const offest = 1

	swapLimit, err := readCgroupFile(f, pod, cgroupV2SwapLimitFile)
	gomega.ExpectWithOffset(offest, err).ToNot(gomega.HaveOccurred())
	gomega.ExpectWithOffset(offest, swapLimit).To(gomega.Equal("0"), "max swap allowed should be zero")
}

// supports v2 only as v1 shouldn't support LimitedSwap
func expectLimitedSwap(f *framework.Framework, pod *v1.Pod, expectedSwapLimit int64) {
	ginkgo.By("expecting limited swap")
	swapLimitStr, err := readCgroupFile(f, pod, cgroupV2SwapLimitFile)
	framework.ExpectNoError(err)

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
	memCapacity = cloneQuantity(&nodeOrigCapacity)
	usedMemory = cloneQuantity(&nodeOrigCapacity)

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

func getSwapBehavior() string {
	kubeletCfg, err := getCurrentKubeletConfig(context.Background())
	framework.ExpectNoError(err, "cannot get kubelet config")

	swapBehavior := kubeletCfg.MemorySwap.SwapBehavior
	ginkgo.By("Figuring out swap behavior: " + swapBehavior)
	return swapBehavior
}

func cloneQuantity(resource *resource.Quantity) *resource.Quantity {
	clone := resource.DeepCopy()
	return &clone
}

func divideQuantity(quantity *resource.Quantity, divideBy int64) *resource.Quantity {
	dividedBigInt := new(big.Int).Div(
		quantity.AsDec().UnscaledBig(),
		big.NewInt(divideBy),
	)

	return resource.NewQuantity(dividedBigInt.Int64(), quantity.Format)
}

func multiplyQuantity(quantity *resource.Quantity, multiplier int64) *resource.Quantity {
	product := new(big.Int).Mul(
		quantity.AsDec().UnscaledBig(),
		big.NewInt(multiplier),
	)

	return resource.NewQuantity(product.Int64(), quantity.Format)
}

func readCgroupFile(f *framework.Framework, pod *v1.Pod, cgroupFile string) (string, error) {
	cgroupPath := filepath.Join(cgroupBasePath, cgroupFile)

	outputStr := e2epod.ExecCommandInContainer(f, pod.Name, pod.Spec.Containers[0].Name, "sh", "-c", "cat "+cgroupPath)
	outputStr = strings.TrimSpace(outputStr)
	ginkgo.By("cgroup found value: " + outputStr)

	return outputStr, nil
}

func parseBytesStrToQuantity(bytesStr string) (*resource.Quantity, error) {
	bytesInt, err := strconv.ParseInt(bytesStr, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("error parsing swap usage %s to int: %w", bytesStr, err)
	}

	return resource.NewQuantity(bytesInt, resource.BinarySI), nil
}

func getSwapUsage(f *framework.Framework, pod *v1.Pod) (*resource.Quantity, error) {
	outputStr, err := readCgroupFile(f, pod, cgroupV2swapCurrentUsageFile)
	if err != nil {
		return nil, err
	}

	ginkgo.By("swap usage found: " + outputStr + " bytes")

	return parseBytesStrToQuantity(outputStr)
}

func getMemoryUsage(f *framework.Framework, pod *v1.Pod) (*resource.Quantity, error) {
	outputStr, err := readCgroupFile(f, pod, cgroupV2MemoryCurrentUsageFile)
	if err != nil {
		return nil, err
	}

	ginkgo.By("memory usage found: " + outputStr + " bytes")

	return parseBytesStrToQuantity(outputStr)
}

// Sets memory request or limit can be null, then it's dismissed.
// Sets the same value for all containers.
func setPodMemoryResources(pod *v1.Pod, memoryRequest, memoryLimit *resource.Quantity) {
	for i := range pod.Spec.Containers {
		resources := &pod.Spec.Containers[i].Resources

		if memoryRequest != nil {
			if resources.Requests == nil {
				resources.Requests = make(map[v1.ResourceName]resource.Quantity)
			}

			resources.Requests[v1.ResourceMemory] = *memoryRequest
		}

		if memoryLimit != nil {
			if resources.Limits == nil {
				resources.Limits = make(map[v1.ResourceName]resource.Quantity)
			}

			resources.Limits[v1.ResourceMemory] = *memoryLimit
		}
	}
}
