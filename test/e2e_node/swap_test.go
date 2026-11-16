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
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/feature"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/ptr"

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
	cgroupV2MaxSwapLimit           = "max"
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

			// Under LimitedSwap only Burstable pods whose memory request differs from their limit get swap.
			ginkgo.DescribeTable("with configuration", func(qosClass v1.PodQOSClass, memoryRequestEqualLimit, expectSwap bool) {
				ginkgo.By(fmt.Sprintf("Creating a pod of QOS class %s. memoryRequestEqualLimit: %t", qosClass, memoryRequestEqualLimit))
				pod := getSwapTestPod(f, qosClass, memoryRequestEqualLimit)
				pod = runPodAndWaitUntilScheduled(f, pod)

				if !isPodCgroupV2(f, pod) {
					e2eskipper.Skipf("swap tests require cgroup v2")
				}
				gomega.Expect(getSwapBehavior()).To(gomega.Equal(string(types.LimitedSwap)))

				if !expectSwap {
					expectNoSwap(f, pod)
					return
				}

				expectedSwapLimit := calcSwapForBurstablePod(f, pod)
				gomega.Expect(expectedSwapLimit).To(gomega.BeNumerically(">", 0), "expected a non-zero swap limit for a swap-eligible pod, is swap provisioned on the node?")
				expectLimitedSwap(f, pod, expectedSwapLimit)
			},
				ginkgo.Entry("QOS Best-effort", v1.PodQOSBestEffort, false, false),
				ginkgo.Entry("QOS Burstable", v1.PodQOSBurstable, false, true),
				ginkgo.Entry("QOS Burstable with memory request equals to limit", v1.PodQOSBurstable, true, false),
				ginkgo.Entry("QOS Guaranteed", v1.PodQOSGuaranteed, false, false),
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

		f.Context("with PodLevelResources", framework.WithFeatureGate(features.PodLevelResources), func() {
			ginkgo.BeforeEach(func() {
				e2eskipper.SkipUnlessFeatureGateEnabled(features.PodLevelResources)
			})

			enableNoSwap := func(ctx context.Context, initialConfig *config.KubeletConfiguration) {
				initialConfig.MemorySwap.SwapBehavior = string(types.NoSwap)
			}

			ginkgo.Context("NoSwap", func() {
				tempSetCurrentKubeletConfig(f, enableNoSwap)

				ginkgo.It("disables swap on pod and container cgroups for pod-level specs while leaving container-only pod cgroup at max", func() {
					podOnly := getPodLevelSwapTestPod(f.Namespace.Name,
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
							Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
						},
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{}),
							newSleepingContainer("c2", v1.ResourceRequirements{}),
						},
					)
					podOnly = runPodAndWaitUntilScheduled(f, podOnly)
					if !isPodCgroupV2(f, podOnly) {
						e2eskipper.Skipf("swap tests require cgroup v2")
					}
					expectPodCgroupNoSwap(podOnly)
					expectContainerNoSwap(f, podOnly, "c1")
					expectContainerNoSwap(f, podOnly, "c2")

					mixedPod := getPodLevelSwapTestPod(f.Namespace.Name,
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
							Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("512Mi")},
						},
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
								Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
							}),
							newSleepingContainer("c2", v1.ResourceRequirements{}),
						},
					)
					mixedPod = runPodAndWaitUntilScheduled(f, mixedPod)
					expectPodCgroupNoSwap(mixedPod)
					expectContainerNoSwap(f, mixedPod, "c1")
					expectContainerNoSwap(f, mixedPod, "c2")

					containerOnlyPod := getPodLevelSwapTestPod(f.Namespace.Name,
						nil,
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
								Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
							}),
						},
					)
					containerOnlyPod = runPodAndWaitUntilScheduled(f, containerOnlyPod)
					expectPodCgroupMaxSwap(containerOnlyPod)
					expectContainerNoSwap(f, containerOnlyPod, "c1")
				})
			})

			ginkgo.Context("LimitedSwap", func() {
				tempSetCurrentKubeletConfig(f, enableLimitedSwap)

				type swapExpectation struct {
					mode       string // "0", "max", or "proportional"
					requestMem resource.Quantity
				}

				ginkgo.DescribeTable("configures pod and container cgroup memory.swap.max",
					func(podResources *v1.ResourceRequirements, initContainers, containers []v1.Container, expectedPodSwap swapExpectation, expectedContainers map[string]swapExpectation) {
						pod := getPodLevelSwapTestPod(f.Namespace.Name, podResources, initContainers, containers)
						pod = runPodAndWaitUntilScheduled(f, pod)
						if !isPodCgroupV2(f, pod) {
							e2eskipper.Skipf("swap tests require cgroup v2")
						}
						if getSwapCapacity(f, pod).IsZero() {
							e2eskipper.Skipf("swap is not provisioned on the node")
						}
						gomega.Expect(getSwapBehavior()).To(gomega.Equal(string(types.LimitedSwap)))

						switch expectedPodSwap.mode {
						case "0":
							expectPodCgroupNoSwap(pod)
						case "max":
							expectPodCgroupMaxSwap(pod)
						case "proportional":
							expectedSwap := calcSwapForMemoryRequest(f, pod, expectedPodSwap.requestMem.Value())
							gomega.Expect(expectedSwap).To(gomega.BeNumerically(">", 0))
							expectPodCgroupLimitedSwap(pod, expectedSwap)
						}

						for ctrName, exp := range expectedContainers {
							switch exp.mode {
							case "0":
								expectContainerNoSwap(f, pod, ctrName)
							case "max":
								expectContainerMaxSwap(f, pod, ctrName)
							case "proportional":
								expectedCtrSwap := calcSwapForMemoryRequest(f, pod, exp.requestMem.Value())
								gomega.Expect(expectedCtrSwap).To(gomega.BeNumerically(">", 0))
								expectContainerLimitedSwap(f, pod, ctrName, expectedCtrSwap)
							}
						}
					},
					ginkgo.Entry("Case 1: pod-level resources only sets container swap to max",
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
							Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
						},
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{}),
							newSleepingContainer("c2", v1.ResourceRequirements{}),
						},
						swapExpectation{mode: "proportional", requestMem: resource.MustParse("128Mi")},
						map[string]swapExpectation{
							"c1": {mode: "max"},
							"c2": {mode: "max"},
						},
					),
					ginkgo.Entry("Case 2: mixed pod with sum of container requests < pod request",
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
							Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("512Mi")},
						},
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
								Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("192Mi")},
							}),
							newSleepingContainer("c2", v1.ResourceRequirements{}),
						},
						swapExpectation{mode: "proportional", requestMem: resource.MustParse("256Mi")},
						map[string]swapExpectation{
							"c1": {mode: "proportional", requestMem: resource.MustParse("128Mi")},
							"c2": {mode: "max"},
						},
					),
					ginkgo.Entry("Case 3: mixed pod with sum of container requests == pod request",
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
							Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
						},
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
								Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("192Mi")},
							}),
							newSleepingContainer("c2", v1.ResourceRequirements{}),
						},
						swapExpectation{mode: "proportional", requestMem: resource.MustParse("128Mi")},
						map[string]swapExpectation{
							"c1": {mode: "proportional", requestMem: resource.MustParse("128Mi")},
							"c2": {mode: "max"},
						},
					),
					ginkgo.Entry("Case 4: container with request == limit opts out inside eligible pod",
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
							Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("512Mi")},
						},
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
								Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
							}),
							newSleepingContainer("c2", v1.ResourceRequirements{}),
						},
						swapExpectation{mode: "proportional", requestMem: resource.MustParse("256Mi")},
						map[string]swapExpectation{
							"c1": {mode: "0"},
							"c2": {mode: "max"},
						},
					),
					ginkgo.Entry("Case 5: restartable sidecar and regular containers follow uniform rules",
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
							Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("512Mi")},
						},
						[]v1.Container{
							func() v1.Container {
								sc := newSleepingContainer("sidecar", v1.ResourceRequirements{
									Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("64Mi")},
									Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
								})
								sc.RestartPolicy = ptr.To(v1.ContainerRestartPolicyAlways)
								return sc
							}(),
						},
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{}),
						},
						swapExpectation{mode: "proportional", requestMem: resource.MustParse("256Mi")},
						map[string]swapExpectation{
							"sidecar": {mode: "proportional", requestMem: resource.MustParse("64Mi")},
							"c1":      {mode: "max"},
						},
					),
					ginkgo.Entry("Case 6a: Guaranteed pod with pod-level resources disables swap",
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("200m"),
								v1.ResourceMemory: resource.MustParse("128Mi"),
							},
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("200m"),
								v1.ResourceMemory: resource.MustParse("128Mi"),
							},
						},
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{}),
						},
						swapExpectation{mode: "0"},
						map[string]swapExpectation{
							"c1": {mode: "0"},
						},
					),
					ginkgo.Entry("Case 6b: Burstable pod with PodMemoryRequest == PodMemoryLimit disables container swap",
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: resource.MustParse("256Mi"),
							},
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("200m"),
								v1.ResourceMemory: resource.MustParse("256Mi"),
							},
						},
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
								Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("192Mi")},
							}),
						},
						swapExpectation{mode: "0"},
						map[string]swapExpectation{
							"c1": {mode: "0"},
						},
					),
					ginkgo.Entry("container-only Burstable pod leaves pod cgroup at max and container without memory at 0",
						nil,
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
								Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
							}),
							newSleepingContainer("c2", v1.ResourceRequirements{}),
						},
						swapExpectation{mode: "max"},
						map[string]swapExpectation{
							"c1": {mode: "proportional", requestMem: resource.MustParse("128Mi")},
							"c2": {mode: "0"},
						},
					),
					ginkgo.Entry("pod-level CPU only with no memory disables swap on pod and containers",
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
							Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("200m")},
						},
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{}),
						},
						swapExpectation{mode: "0"},
						map[string]swapExpectation{
							"c1": {mode: "0"},
						},
					),
					ginkgo.Entry("pod-level CPU limits with container c1 req < lim and c2 unbounded (API-defaulted pod memory request) aggregates pod swap and sets c2 to max",
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
							Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("200m")},
						},
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
								Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
							}),
							newSleepingContainer("c2", v1.ResourceRequirements{}),
						},
						swapExpectation{mode: "proportional", requestMem: resource.MustParse("128Mi")},
						map[string]swapExpectation{
							"c1": {mode: "proportional", requestMem: resource.MustParse("128Mi")},
							"c2": {mode: "max"},
						},
					),
					ginkgo.Entry("pod-level CPU limits with c1 req == lim and c2 unbounded (API-defaulted pod memory request) aggregates pod swap, opts out c1, and sets c2 to max",
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
							Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("200m")},
						},
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
								Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
							}),
							newSleepingContainer("c2", v1.ResourceRequirements{}),
						},
						swapExpectation{mode: "proportional", requestMem: resource.MustParse("128Mi")},
						map[string]swapExpectation{
							"c1": {mode: "0"},
							"c2": {mode: "max"},
						},
					),
					ginkgo.Entry("pod-level CPU with single container req == lim disables swap because PodMemoryRequest == PodMemoryLimit",
						&v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
							Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("200m")},
						},
						nil,
						[]v1.Container{
							newSleepingContainer("c1", v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
								Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
							}),
						},
						swapExpectation{mode: "0"},
						map[string]swapExpectation{
							"c1": {mode: "0"},
						},
					),
				)

				ginkgo.It("allows container without container-level resources to use swap bounded by pod-level limit", func() {
					podClient := e2epod.NewPodClient(f)
					probePod := runPodAndWaitUntilScheduled(f, getSleepingPod(f.Namespace.Name))
					if !isPodCgroupV2(f, probePod) {
						e2eskipper.Skipf("swap tests require cgroup v2")
					}
					nodeName := probePod.Spec.NodeName
					nodeTotalMemory, _ := getMemoryCapacity(f, nodeName)
					swapCapacity := getSwapCapacity(f, probePod)
					framework.ExpectNoError(podClient.Delete(context.Background(), probePod.Name, metav1.DeleteOptions{}))
					if swapCapacity.IsZero() {
						e2eskipper.Skipf("swap is not provisioned on the node")
					}

					stressMemAllocSize := multiplyQuantity(divideQuantity(nodeTotalMemory, 1000), 4)
					stressSize := divideQuantity(nodeTotalMemory, 5)
					memoryLimit := cloneQuantity(stressSize)
					memoryLimit.Sub(resource.MustParse("50Mi"))
					memoryRequest := divideQuantity(memoryLimit, 2)

					stressPod := getStressPod(f, stressSize, stressMemAllocSize)
					stressPod.Spec.NodeName = nodeName
					stressPod.Spec.Resources = &v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceMemory: *memoryRequest},
						Limits:   v1.ResourceList{v1.ResourceMemory: *memoryLimit},
					}
					gomega.Expect(qos.GetPodQOS(stressPod)).To(gomega.Equal(v1.PodQOSBurstable))

					stressPod = runPodAndWaitUntilScheduled(f, stressPod)
					expectContainerMaxSwap(f, stressPod, stressPod.Spec.Containers[0].Name)
					expectPodCgroupLimitedSwap(stressPod, calcSwapForMemoryRequest(f, stressPod, memoryRequest.Value()))

					minExpectedMemoryUsage := multiplyQuantity(divideQuantity(memoryLimit, 4), 3)
					gomega.Eventually(func() error {
						stressPod = getUpdatedPod(f, stressPod)
						gomega.Expect(stressPod.Status.Phase).To(gomega.Equal(v1.PodRunning), "pod should be running")

						swapUsage, err := getSwapUsage(f, stressPod)
						if err != nil {
							return err
						}
						memoryUsage, err := getMemoryUsage(f, stressPod)
						if err != nil {
							return err
						}
						if memoryUsage.Cmp(*minExpectedMemoryUsage) == -1 {
							return fmt.Errorf("memory usage (%s) is smaller than minimum expected (%s)", memoryUsage.String(), minExpectedMemoryUsage.String())
						}
						if swapUsage.IsZero() {
							return fmt.Errorf("swap usage is zero")
						}
						return nil
					}, 5*time.Minute, 1*time.Second).Should(gomega.Succeed())

					framework.ExpectNoError(podClient.Delete(context.Background(), stressPod.Name, metav1.DeleteOptions{}))
				})
			})
		})
	})
})

// Note that memoryRequestEqualLimit is effective only when qosClass is PodQOSBurstable.
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
			resources.Limits = resources.Requests.DeepCopy()
		}
	case v1.PodQOSGuaranteed:
		resources = v1.ResourceRequirements{
			Limits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("200m"),
				v1.ResourceMemory: podMemoryAmount,
			},
		}
		resources.Requests = resources.Limits.DeepCopy()
	}

	pod := getSleepingPod(f.Namespace.Name)
	pod.Spec.Containers[0].Resources = resources
	gomega.ExpectWithOffset(1, qos.GetPodQOS(pod)).To(gomega.Equal(qosClass), "swap test pod does not have the requested QoS class")

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
	expectContainerLimitedSwap(f, pod, pod.Spec.Containers[0].Name, expectedSwapLimit)
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
	return calcSwapForMemoryRequest(f, pod, pod.Spec.Containers[0].Resources.Requests.Memory().Value())
}

// calcSwapForMemoryRequest computes expected proportional swap allocation independently
// of cm.CalcSwapForBurstablePods so the E2E test does not share implementation with the kubelet code under test.
func calcSwapForMemoryRequest(f *framework.Framework, pod *v1.Pod, memoryRequest int64) int64 {
	gomega.Expect(pod.Spec.NodeName).ToNot(gomega.BeEmpty(), "pod node name is empty")

	nodeMemoryCapacityQuantity, _ := getMemoryCapacity(f, pod.Spec.NodeName)
	nodeMemoryCapacity := nodeMemoryCapacityQuantity.Value()
	nodeSwapCapacity := getSwapCapacity(f, pod).Value()

	memoryProportion := float64(memoryRequest) / float64(nodeMemoryCapacity)
	swapAllocation := memoryProportion * float64(nodeSwapCapacity)
	ginkgo.By(fmt.Sprintf("Calculating swap for burstable pods: nodeMemoryCapacity: %d, nodeSwapCapacity: %d, memoryRequest: %d, swapAllocation: %d",
		nodeMemoryCapacity, nodeSwapCapacity, memoryRequest, int64(swapAllocation)))
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
	return readContainerCgroupFile(f, pod, pod.Spec.Containers[0].Name, cgroupFile)
}

func readContainerCgroupFile(f *framework.Framework, pod *v1.Pod, containerName, cgroupFile string) (string, error) {
	cgroupPath := filepath.Join(cgroupBasePath, cgroupFile)

	outputStr := e2epod.ExecCommandInContainer(f, pod.Name, containerName, "sh", "-c", "cat "+cgroupPath)
	outputStr = strings.TrimSpace(outputStr)
	ginkgo.By(fmt.Sprintf("container %s cgroup %s found value: %s", containerName, cgroupFile, outputStr))

	return outputStr, nil
}

func newSleepingContainer(name string, resources v1.ResourceRequirements) v1.Container {
	return v1.Container{
		Name:      name,
		Image:     busyboxImage,
		Command:   []string{"sleep", "600"},
		Resources: resources,
	}
}

func getPodLevelSwapTestPod(namespace string, podResources *v1.ResourceRequirements, initContainers, containers []v1.Container) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "plr-swap-test-pod-" + rand.String(5),
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			RestartPolicy:                 v1.RestartPolicyAlways,
			TerminationGracePeriodSeconds: ptr.To[int64](0),
			Resources:                     podResources,
			InitContainers:                initContainers,
			Containers:                    containers,
		},
	}
}

func getPodCgroupHostPath(pod *v1.Pod) string {
	kubeletCfg, err := getCurrentKubeletConfig(context.Background())
	framework.ExpectNoError(err, "cannot get kubelet config")

	base := cm.ParseCgroupfsToCgroupName(kubeletCfg.CgroupRoot)
	if kubeletCfg.CgroupsPerQOS {
		base = cm.NewCgroupName(base, defaultNodeAllocatableCgroup)
		switch pod.Status.QOSClass {
		case v1.PodQOSBurstable:
			base = cm.NewCgroupName(base, strings.ToLower(string(v1.PodQOSBurstable)))
		case v1.PodQOSBestEffort:
			base = cm.NewCgroupName(base, strings.ToLower(string(v1.PodQOSBestEffort)))
		}
	}
	podCgroup := cm.NewCgroupName(base, cm.GetPodCgroupNameSuffix(pod.UID))
	return filepath.Join(cgroupBasePath, toCgroupFsName(podCgroup))
}

func readPodCgroupSwapMax(pod *v1.Pod) string {
	podCgroupPath := getPodCgroupHostPath(pod)
	data, err := os.ReadFile(filepath.Join(podCgroupPath, cgroupV2SwapLimitFile))
	framework.ExpectNoError(err, "failed reading pod cgroup memory.swap.max at %s", podCgroupPath)
	val := strings.TrimSpace(string(data))
	ginkgo.By(fmt.Sprintf("pod cgroup %s/%s found value: %s", podCgroupPath, cgroupV2SwapLimitFile, val))
	return val
}

func expectPodCgroupNoSwap(pod *v1.Pod) {
	gomega.ExpectWithOffset(1, readPodCgroupSwapMax(pod)).To(gomega.Equal("0"), "pod cgroup memory.swap.max should be zero")
}

func expectPodCgroupMaxSwap(pod *v1.Pod) {
	gomega.ExpectWithOffset(1, readPodCgroupSwapMax(pod)).To(gomega.Equal(cgroupV2MaxSwapLimit), "pod cgroup memory.swap.max should be max")
}

func expectPodCgroupLimitedSwap(pod *v1.Pod, expectedSwapLimit int64) {
	expectSwapLimitWithinPageTolerance(readPodCgroupSwapMax(pod), expectedSwapLimit, "pod cgroup")
}

func expectContainerNoSwap(f *framework.Framework, pod *v1.Pod, containerName string) {
	swapLimit, err := readContainerCgroupFile(f, pod, containerName, cgroupV2SwapLimitFile)
	gomega.ExpectWithOffset(1, err).ToNot(gomega.HaveOccurred())
	gomega.ExpectWithOffset(1, swapLimit).To(gomega.Equal("0"), "container %s memory.swap.max should be zero", containerName)
}

func expectContainerMaxSwap(f *framework.Framework, pod *v1.Pod, containerName string) {
	swapLimit, err := readContainerCgroupFile(f, pod, containerName, cgroupV2SwapLimitFile)
	gomega.ExpectWithOffset(1, err).ToNot(gomega.HaveOccurred())
	gomega.ExpectWithOffset(1, swapLimit).To(gomega.Equal(cgroupV2MaxSwapLimit), "container %s memory.swap.max should be max", containerName)
}

func expectContainerLimitedSwap(f *framework.Framework, pod *v1.Pod, containerName string, expectedSwapLimit int64) {
	swapLimitStr, err := readContainerCgroupFile(f, pod, containerName, cgroupV2SwapLimitFile)
	framework.ExpectNoError(err)

	expectSwapLimitWithinPageTolerance(swapLimitStr, expectedSwapLimit, fmt.Sprintf("container %s", containerName))
}

func expectSwapLimitWithinPageTolerance(swapLimitStr string, expectedSwapLimit int64, target string) {
	swapLimit, err := strconv.ParseInt(swapLimitStr, 10, 64)
	framework.ExpectNoErrorWithOffset(2, err, "cannot parse %s memory.swap.max %q as int64", target, swapLimitStr)

	// cgroup values are always aligned w.r.t. the page size, which is usually 4Ki
	const cgroupAlignment int64 = 4 * 1024
	gomega.ExpectWithOffset(2, swapLimit).To(
		gomega.And(
			gomega.BeNumerically(">=", expectedSwapLimit-cgroupAlignment),
			gomega.BeNumerically("<=", expectedSwapLimit+cgroupAlignment),
		),
		"%s memory.swap.max is not within page-aligned tolerance of expected swap limit", target,
	)
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
