/*
Copyright 2024 The Kubernetes Authors.

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

package node

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/test/e2e/common/node/framework/cgroups"
	"k8s.io/kubernetes/test/e2e/common/node/framework/podresize"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	fakeExtendedResource = "dummy.com/dummy"

	originalCPU       = "20m"
	originalCPULimit  = "30m"
	reducedCPU        = "15m"
	reducedCPULimit   = "25m"
	increasedCPU      = "25m"
	increasedCPULimit = "35m"

	originalMem       = "35Mi"
	originalMemLimit  = "45Mi"
	reducedMem        = "30Mi"
	reducedMemLimit   = "40Mi"
	increasedMem      = "40Mi"
	increasedMemLimit = "50Mi"
)

func offsetCPU(index int, value string) string {
	if value == "" {
		return ""
	}
	val := resource.MustParse(value)
	ptr := &val
	ptr.Add(resource.MustParse(fmt.Sprintf("%dm", 2*index)))
	return ptr.String()
}

func offsetMemory(index int64, value string) string {
	if value == "" {
		return ""
	}
	val := resource.MustParse(value)
	ptr := &val
	ptr.Add(resource.MustParse(fmt.Sprintf("%dMi", 2*index)))
	return ptr.String()
}

func doGuaranteedPodResizeTests(f *framework.Framework) {
	ginkgo.DescribeTableSubtree("guaranteed qos - 1 container with resize policy", func(cpuPolicy, memPolicy v1.ResourceResizeRestartPolicy, resizeInitCtrs bool) {
		ginkgo.DescribeTable("resizing", func(ctx context.Context, desiredCPU, desiredMem string) {

			// The tests for guaranteed pods include extended resources.
			nodes, err := e2enode.GetReadySchedulableNodes(context.Background(), f.ClientSet)
			framework.ExpectNoError(err)
			for _, node := range nodes.Items {
				e2enode.AddExtendedResource(ctx, f.ClientSet, node.Name, fakeExtendedResource, resource.MustParse("123"))
			}
			defer func() {
				for _, node := range nodes.Items {
					e2enode.RemoveExtendedResource(ctx, f.ClientSet, node.Name, fakeExtendedResource)
				}
			}()

			originalContainers := makeGuaranteedContainers(1, cpuPolicy, memPolicy, true, true, originalCPU, originalMem)
			expectedContainers := makeGuaranteedContainers(1, cpuPolicy, memPolicy, true, true, desiredCPU, desiredMem)
			for i, c := range expectedContainers {
				// If the pod has init containers, but we are not resizing them, keep the original resources.
				if c.InitCtr && !resizeInitCtrs {
					c.Resources = originalContainers[i].Resources
					expectedContainers[i] = c
					continue
				}
				// For containers where the resize policy is "restart", we expect a restart.
				expectRestart := int32(0)
				if cpuPolicy == v1.RestartContainer && desiredCPU != originalCPU {
					expectRestart = 1
				}
				if memPolicy == v1.RestartContainer && desiredMem != originalMem {
					expectRestart = 1
				}
				c.RestartCount = expectRestart
				expectedContainers[i] = c
			}

			doPatchAndRollback(ctx, f, originalContainers, expectedContainers, nil, nil, true, false)
		},
			// All tests will perform the requested resize, and once completed, will roll back the change.
			// This results in the coverage of both increase and decrease of resources.
			ginkgo.Entry("cpu", increasedCPU, originalMem),
			ginkgo.Entry("mem", originalCPU, increasedMem),
			ginkgo.Entry("cpu & mem in the same direction", increasedCPU, increasedMem),
			ginkgo.Entry("cpu & mem in opposite directions", increasedCPU, reducedMem),
		)
	},
		ginkgo.Entry("no restart", v1.NotRequired, v1.NotRequired, false),
		ginkgo.Entry("no restart + resize initContainers", v1.NotRequired, v1.NotRequired, true),
		ginkgo.Entry("mem restart", v1.NotRequired, v1.RestartContainer, false),
		ginkgo.Entry("cpu restart", v1.RestartContainer, v1.NotRequired, false),
		ginkgo.Entry("cpu & mem restart", v1.RestartContainer, v1.RestartContainer, false),
		ginkgo.Entry("cpu & mem restart + resize initContainers", v1.RestartContainer, v1.RestartContainer, true),
	)

	// All tests will perform the requested resize, and once completed, will roll back the change.
	// This results in coverage of both the operation as described, and its reverse.
	ginkgo.Describe("guaranteed pods with multiple containers", func() {
		/*
			Release: v1.35
			Testname: In-place Pod Resize, guaranteed pods with multiple containers, net increase
			Description: Issuing an in-place Pod Resize request via the Pod Resize subresource patch endpoint to modify CPU and memory requests and limits for a guaranteed pod with 3 containers with a net increase MUST result in the Pod resources being updated as expected.
		*/
		framework.ConformanceIt("3 containers - increase cpu & mem on c1, c2, decrease cpu & mem on c3 - net increase [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
			originalContainers := makeGuaranteedContainers(3, v1.NotRequired, v1.NotRequired, false, false, originalCPU, originalMem)
			for i := range originalContainers {
				originalContainers[i].CPUPolicy = nil
				originalContainers[i].MemPolicy = nil
			}

			expectedContainers := []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: offsetCPU(0, increasedCPU), CPULim: offsetCPU(0, increasedCPU), MemReq: offsetMemory(0, increasedMem), MemLim: offsetMemory(0, increasedMem)},
				},
				{
					Name:      "c2",
					Resources: &cgroups.ContainerResources{CPUReq: offsetCPU(1, increasedCPU), CPULim: offsetCPU(1, increasedCPU), MemReq: offsetMemory(1, increasedMem), MemLim: offsetMemory(1, increasedMem)},
				},
				{
					Name:      "c3",
					Resources: &cgroups.ContainerResources{CPUReq: offsetCPU(2, reducedCPU), CPULim: offsetCPU(2, reducedCPU), MemReq: offsetMemory(2, reducedMem), MemLim: offsetMemory(2, reducedMem)},
				},
			}

			doPatchAndRollback(ctx, f, originalContainers, expectedContainers, nil, nil, true, false)
		})

		/*
			Release: v1.35
			Testname: In-place Pod Resize, guaranteed pods with multiple containers, net decrease
			Description: Issuing an in-place Pod Resize request via the Pod Resize subresource patch endpoint to modify CPU and memory requests and limits for a pod with 3 containers with a net decrease MUST result in the Pod resources being updated as expected.
		*/
		framework.ConformanceIt("3 containers - increase cpu & mem on c1, decrease cpu & mem on c2, c3 - net decrease [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
			originalContainers := makeGuaranteedContainers(3, v1.NotRequired, v1.NotRequired, false, false, originalCPU, originalMem)
			for i := range originalContainers {
				originalContainers[i].CPUPolicy = nil
				originalContainers[i].MemPolicy = nil
			}

			expectedContainers := []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: offsetCPU(0, increasedCPU), CPULim: offsetCPU(0, increasedCPU), MemReq: offsetMemory(0, increasedMem), MemLim: offsetMemory(0, increasedMem)},
				},
				{
					Name:      "c2",
					Resources: &cgroups.ContainerResources{CPUReq: offsetCPU(1, reducedCPU), CPULim: offsetCPU(1, reducedCPU), MemReq: offsetMemory(1, reducedMem), MemLim: offsetMemory(1, reducedMem)},
				},
				{
					Name:      "c3",
					Resources: &cgroups.ContainerResources{CPUReq: offsetCPU(2, reducedCPU), CPULim: offsetCPU(2, reducedCPU), MemReq: offsetMemory(2, reducedMem), MemLim: offsetMemory(2, reducedMem)},
				},
			}

			doPatchAndRollback(ctx, f, originalContainers, expectedContainers, nil, nil, true, false)
		})

		/*
			Release: v1.35
			Testname: In-place Pod Resize, guaranteed pods with multiple containers, various operations
			Description: Issuing an in-place Pod Resize request via the Pod Resize subresource patch endpoint to modify CPU and memory requests and limits for a pod with 3 containers with various operations MUST result in the Pod resources being updated as expected.
		*/
		framework.ConformanceIt("3 containers - increase: CPU (c1,c3), memory (c2, c3) ; decrease: CPU (c2) [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
			originalContainers := makeGuaranteedContainers(3, v1.NotRequired, v1.NotRequired, false, false, originalCPU, originalMem)
			for i := range originalContainers {
				originalContainers[i].CPUPolicy = nil
				originalContainers[i].MemPolicy = nil
			}

			expectedContainers := []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: offsetCPU(0, increasedCPU), CPULim: offsetCPU(0, increasedCPU), MemReq: offsetMemory(0, originalMem), MemLim: offsetMemory(0, originalMem)},
				},
				{
					Name:      "c2",
					Resources: &cgroups.ContainerResources{CPUReq: offsetCPU(1, reducedCPU), CPULim: offsetCPU(1, reducedCPU), MemReq: offsetMemory(1, increasedMem), MemLim: offsetMemory(1, increasedMem)},
				},
				{
					Name:      "c3",
					Resources: &cgroups.ContainerResources{CPUReq: offsetCPU(2, increasedCPU), CPULim: offsetCPU(2, increasedCPU), MemReq: offsetMemory(2, increasedMem), MemLim: offsetMemory(2, increasedMem)},
				},
			}

			doPatchAndRollback(ctx, f, originalContainers, expectedContainers, nil, nil, true, false)
		})
	})
}

type resourceRequestsLimits struct {
	cpuReq string
	cpuLim string
	memReq string
	memLim string
}

func doBurstablePodResizeTests(f *framework.Framework) {
	ginkgo.DescribeTableSubtree("burstable pods - 1 container with all requests & limits set and resize policy", func(cpuPolicy, memPolicy v1.ResourceResizeRestartPolicy) {
		ginkgo.DescribeTable("resizing", func(ctx context.Context, desiredContainerResources resourceRequestsLimits) {
			originalContainers := makeBurstableContainers(1, cpuPolicy, memPolicy, originalCPU, originalCPULimit, originalMem, originalMemLimit)
			expectedContainers := makeBurstableContainers(1, cpuPolicy, memPolicy, desiredContainerResources.cpuReq, desiredContainerResources.cpuLim, desiredContainerResources.memReq, desiredContainerResources.memLim)
			for i, c := range expectedContainers {
				// For containers where the resize policy is "restart", we expect a restart.
				expectRestart := int32(0)
				if cpuPolicy == v1.RestartContainer && (desiredContainerResources.cpuReq != originalCPU || desiredContainerResources.cpuLim != originalCPULimit) {
					expectRestart = 1
				}
				if memPolicy == v1.RestartContainer && (desiredContainerResources.memReq != originalMem || desiredContainerResources.memLim != originalMemLimit) {
					expectRestart = 1
				}
				c.RestartCount = expectRestart
				expectedContainers[i] = c
			}

			doPatchAndRollback(ctx, f, originalContainers, expectedContainers, nil, nil, true, false)
		},
			// All tests will perform the requested resize, and once completed, will roll back the change.
			// This results in the coverage of both increase and decrease of resources.
			ginkgo.Entry("cpu requests", resourceRequestsLimits{increasedCPU, originalCPULimit, originalMem, originalMemLimit}),
			ginkgo.Entry("cpu limits", resourceRequestsLimits{originalCPU, increasedCPULimit, originalMem, originalMemLimit}),
			ginkgo.Entry("mem requests", resourceRequestsLimits{originalCPU, originalCPULimit, increasedMem, originalMemLimit}),
			ginkgo.Entry("mem limits", resourceRequestsLimits{originalCPU, originalCPULimit, originalMem, increasedMemLimit}),
			ginkgo.Entry("all resources in the same direction", resourceRequestsLimits{increasedCPU, increasedCPULimit, increasedMem, increasedMemLimit}),
			ginkgo.Entry("cpu & mem in opposite directions", resourceRequestsLimits{increasedCPU, increasedCPULimit, reducedMem, reducedMemLimit}),
			ginkgo.Entry("requests & limits in opposite directions", resourceRequestsLimits{reducedCPU, increasedCPULimit, increasedMem, reducedMemLimit}),
		)
	},
		ginkgo.Entry("no restart", v1.NotRequired, v1.NotRequired),
		ginkgo.Entry("cpu restart", v1.RestartContainer, v1.NotRequired),
		ginkgo.Entry("mem restart", v1.NotRequired, v1.RestartContainer),
		ginkgo.Entry("cpu & mem restart", v1.RestartContainer, v1.RestartContainer),
	)

	// The following tests cover the remaining burstable pod scenarios:
	// - multiple containers
	// - adding limits where only requests were previously set
	// - adding requests where none were previously set
	// - resizing with equivalents (e.g. 2m -> 1m)
	ginkgo.Describe("burstable pods - extended", func() {
		/*
			Release: v1.35
			Testname: In-place Pod Resize, burtable pod with multiple containers and various operations
			Description: Issuing a Pod Resize request via the Pod Resize subresource patch endpoint to modify CPU and memory requests and limits on a 6-container pod with various operations MUST result in the Pod resources being updated as expected.
		*/
		framework.ConformanceIt("6 containers - various operations performed (including adding limits and requests) [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
			originalContainers := []podresize.ResizableContainerInfo{
				{
					// c1 starts with CPU requests only; increase CPU requests + add CPU limits
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU},
				},
				{
					// c2 starts with memory requests only; increase memory requests + add memory limits
					Name:      "c2",
					Resources: &cgroups.ContainerResources{MemReq: originalMem},
				},
				{
					// c3 starts with CPU and memory requests; decrease memory requests only
					Name:      "c3",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, MemReq: originalMem},
				},
				{
					// c4 starts with CPU requests only; decrease CPU requests + add memory requests
					Name:      "c4",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU},
				},
				{
					// c5 starts with memory requests only; increase memory requests + add CPU requests
					Name:      "c5",
					Resources: &cgroups.ContainerResources{MemReq: originalMem},
				},
			}
			expectedContainers := []podresize.ResizableContainerInfo{
				{
					// c1 starts with CPU requests only; increase CPU requests + add CPU limits
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: increasedCPU, CPULim: increasedCPULimit},
				},
				{
					// c2 starts with memory requests only; decrease memory requests + add memory limits
					Name:      "c2",
					Resources: &cgroups.ContainerResources{MemReq: reducedMem, MemLim: increasedMemLimit},
				},
				{
					// c3 starts with CPU and memory requests; decrease memory requests onloy
					Name:      "c3",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, MemReq: reducedMem},
				},
				{
					// c4 starts with CPU requests only; decrease CPU requests + add memory requests
					Name:      "c4",
					Resources: &cgroups.ContainerResources{CPUReq: reducedCPU, MemReq: originalMem},
				},
				{
					// c5 starts with memory requests only; increase memory requests + add CPU requests
					Name:      "c5",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, MemReq: increasedMem},
				},
			}
			doPatchAndRollback(ctx, f, originalContainers, expectedContainers, nil, nil, false, false)
		})

		/*
			Release: v1.35
			Testname: In-place Pod Resize, burstable pod resized with equivalents
			Description: Issuing an in-place Pod Resize request via the Pod Resize subresource patch endpoint to modify CPU requests and limits using equivalent values (e.g. 2m -> 1m) MUST result in the updated Pod resources displayed correctly in the status.
		*/
		framework.ConformanceIt("resize with equivalents [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
			originalContainers := []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "2m", CPULim: "10m"},
				},
			}
			expectedContainers := []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "1m", CPULim: "5m"},
				},
			}
			doPatchAndRollback(ctx, f, originalContainers, expectedContainers, nil, nil, true, false)
		})
	})
}

func doPodResizePatchErrorTests(f *framework.Framework) {
	ginkgo.DescribeTable("apply invalid resize patch requests", func(ctx context.Context, originalContainers, desiredContainers []podresize.ResizableContainerInfo, patchError string, waitForStart bool) {
		podClient := e2epod.NewPodClient(f)
		var newPod *v1.Pod

		ginkgo.By("creating and verifying pod")
		if waitForStart {
			newPod = createAndVerifyPod(ctx, f, podClient, originalContainers, nil, false)
		} else {
			tStamp := strconv.Itoa(time.Now().Nanosecond())
			testPod := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod", tStamp, originalContainers, nil)
			testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)
			newPod = podClient.Create(ctx, testPod)
		}

		ginkgo.By("patching pod for resize")
		patch := podresize.MakeResizePatch(originalContainers, desiredContainers, nil, nil)
		_, pErr := f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
			types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
		gomega.Expect(pErr).To(gomega.HaveOccurred())
		gomega.Expect(pErr.Error()).To(gomega.ContainSubstring(patchError))

		patchedPod, getErr := f.ClientSet.CoreV1().Pods(newPod.Namespace).Get(ctx, newPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(getErr)

		ginkgo.By("verifying pod resources after patch")
		podresize.VerifyPodResources(patchedPod, originalContainers, nil)

		if waitForStart {
			ginkgo.By("verifying pod status resources after patch")
			framework.ExpectNoError(podresize.VerifyPodStatusResources(patchedPod, originalContainers))
		}

		ginkgo.By("deleting pod")
		podClient.DeleteSync(ctx, newPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
	},
		ginkgo.Entry("BestEffort pod - request memory",
			[]podresize.ResizableContainerInfo{
				{
					Name: "c1",
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{MemReq: originalMem},
				},
			},
			"Pod QOS Class may not change as a result of resizing",
			true,
		),
		ginkgo.Entry("BestEffort pod - request cpu",
			[]podresize.ResizableContainerInfo{
				{
					Name: "c1",
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU},
				},
			},
			"Pod QOS Class may not change as a result of resizing",
			true,
		),
		ginkgo.Entry("Guaranteed pod - remove cpu & memory limits",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, MemReq: originalMem},
				},
			},
			"resource limits cannot be removed",
			true,
		),
		ginkgo.Entry("Burstable pod - remove cpu & memory limits + increase requests",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, CPULim: increasedCPULimit, MemReq: originalMem, MemLim: increasedMemLimit},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: increasedCPU, MemReq: increasedMem},
				},
			},
			"resource limits cannot be removed",
			true,
		),
		ginkgo.Entry("Burstable pod - remove memory requests",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{MemReq: originalMem},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{},
				},
			},
			"resource requests cannot be removed",
			true,
		),
		ginkgo.Entry("Burstable pod - remove cpu requests",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{},
				},
			},
			"resource requests cannot be removed",
			true,
		),
		ginkgo.Entry("Burstable pod - reorder containers",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU},
				},
				{
					Name:      "c2",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c2",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU},
				},
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU},
				},
			},
			"spec.containers[0].name: Forbidden: containers may not be renamed or reordered on resize, spec.containers[1].name: Forbidden: containers may not be renamed or reordered on resize",
			true,
		),
		ginkgo.Entry("Guaranteed pod - rename containers",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1-old",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
				{
					Name:      "c2-old",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1-new",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
				{
					Name:      "c2-new",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
			},
			"spec.containers[0].name: Forbidden: containers may not be renamed or reordered on resize, spec.containers[1].name: Forbidden: containers may not be renamed or reordered on resize",
			true,
		),
		ginkgo.Entry("Burstable pod - set requests == limits",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, CPULim: increasedCPULimit, MemReq: originalMem, MemLim: increasedMemLimit},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: increasedCPULimit, CPULim: increasedCPULimit, MemReq: increasedMemLimit, MemLim: increasedMemLimit},
				},
			},
			"Pod QOS Class may not change as a result of resizing",
			true,
		),
		ginkgo.Entry("Burstable pod - resize ephemeral storage",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, MemReq: originalMem, EphStorReq: "1"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: originalCPU, MemReq: originalMem, EphStorReq: "3"},
				},
			},
			"only cpu and memory resources are mutable",
			true,
		),
		ginkgo.Entry("Burstable pod - nonrestartable initContainer",
			[]podresize.ResizableContainerInfo{
				{
					Name:          "c1-init",
					InitCtr:       true,
					Resources:     &cgroups.ContainerResources{CPUReq: originalCPU, MemReq: originalMem},
					RestartPolicy: v1.ContainerRestartPolicyNever,
				},
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:          "c1-init",
					InitCtr:       true,
					Resources:     &cgroups.ContainerResources{CPUReq: increasedCPU, MemReq: increasedMem},
					RestartPolicy: v1.ContainerRestartPolicyNever,
				},
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{},
				},
			},
			"resources for non-sidecar init containers are immutable",
			false,
		),
	)
}

func doPodResizeMemoryLimitDecreaseTest(f *framework.Framework) {
	// Tests the behavior when decreasing memory limit:
	// 1. Decrease the limit a little bit - should succeed
	// 2. Decrease the limit down to a tiny amount - should fail
	// 3. Revert the limit back to the original value - should succeed
	ginkgo.It("decrease memory limit below usage", func(ctx context.Context) {
		podClient := e2epod.NewPodClient(f)
		original := []podresize.ResizableContainerInfo{{
			Name:      "c1",
			Resources: &cgroups.ContainerResources{MemReq: originalMem, MemLim: originalMem},
		}}

		ginkgo.By("creating and verifying pod")
		testPod := createAndVerifyPod(ctx, f, podClient, original, nil, false)

		// 1. Decrease the limit a little bit - should succeed
		ginkgo.By("Patching pod with a slightly lowered memory limit")
		viableLoweredLimit := []podresize.ResizableContainerInfo{{
			Name:      "c1",
			Resources: &cgroups.ContainerResources{MemReq: reducedMem, MemLim: reducedMem},
		}}
		patch := podresize.MakeResizePatch(original, viableLoweredLimit, nil, nil)
		testPod, pErr := f.ClientSet.CoreV1().Pods(testPod.Namespace).Patch(ctx, testPod.Name,
			types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(pErr, "failed to patch pod for viable lowered limit")

		ginkgo.By("verifying pod patched for viable lowered limit")
		podresize.VerifyPodResources(testPod, viableLoweredLimit, nil)

		ginkgo.By("waiting for viable lowered limit to be actuated")
		resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod, viableLoweredLimit)
		podresize.ExpectPodResized(ctx, f, resizedPod, viableLoweredLimit)

		// There is some latency after container startup before memory usage is scraped. On CRI-O
		// this latency is much higher, so wait enough time for cAdvisor to scrape metrics twice.
		ginkgo.By("Waiting for stats scraping")
		const scrapingDelay = 30 * time.Second // 2 * maxHousekeepingInterval
		startTime := testPod.Status.StartTime
		time.Sleep(time.Until(startTime.Add(scrapingDelay)))

		// 2. Decrease the limit down to a tiny amount - should fail
		const nonViableMemoryLimit = "10Ki"
		ginkgo.By("Patching pod with a greatly lowered memory limit")
		nonViableLoweredLimit := []podresize.ResizableContainerInfo{{
			Name:      "c1",
			Resources: &cgroups.ContainerResources{MemReq: nonViableMemoryLimit, MemLim: nonViableMemoryLimit},
		}}
		patch = podresize.MakeResizePatch(viableLoweredLimit, nonViableLoweredLimit, nil, nil)
		testPod, pErr = f.ClientSet.CoreV1().Pods(testPod.Namespace).Patch(ctx, testPod.Name,
			types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(pErr, "failed to patch pod for viable lowered limit")

		framework.ExpectNoError(framework.Gomega().
			Eventually(ctx, framework.RetryNotFound(framework.GetObject(f.ClientSet.CoreV1().Pods(testPod.Namespace).Get, testPod.Name, metav1.GetOptions{}))).
			WithTimeout(f.Timeouts.PodStart).
			Should(framework.MakeMatcher(func(pod *v1.Pod) (func() string, error) {
				// If VerifyPodStatusResources succeeds, it means the resize completed.
				if podresize.VerifyPodStatusResources(pod, nonViableLoweredLimit) == nil {
					return nil, gomega.StopTrying("non-viable resize unexpectedly completed")
				}

				var inProgressCondition *v1.PodCondition
				for i, condition := range pod.Status.Conditions {
					switch condition.Type {
					case v1.PodResizeInProgress:
						inProgressCondition = &pod.Status.Conditions[i]
					case v1.PodResizePending:
						return func() string {
							return fmt.Sprintf("pod should not be pending, got reason=%s, message=%q", condition.Reason, condition.Message)
						}, nil
					}
				}
				if inProgressCondition == nil {
					return func() string { return "resize is not in progress" }, nil
				}

				if inProgressCondition.Reason != v1.PodReasonError {
					return func() string { return "in-progress reason is not error" }, nil
				}

				expectedMsg := regexp.MustCompile(`memory limit \(\d+\) below current usage`)
				if !expectedMsg.MatchString(inProgressCondition.Message) {
					return func() string {
						return fmt.Sprintf("Expected %q to contain %q", inProgressCondition.Message, expectedMsg)
					}, nil
				}
				return nil, nil
			})),
		)
		ginkgo.By("verifying pod status resources still match the viable resize")
		framework.ExpectNoError(podresize.VerifyPodStatusResources(testPod, viableLoweredLimit))

		// 3. Revert the limit back to the original value - should succeed
		ginkgo.By("Patching pod to revert to original state")
		patch = podresize.MakeResizePatch(nonViableLoweredLimit, original, nil, nil)
		testPod, pErr = f.ClientSet.CoreV1().Pods(testPod.Namespace).Patch(ctx, testPod.Name,
			types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(pErr, "failed to patch pod back to original values")

		ginkgo.By("verifying pod patched for original values")
		podresize.VerifyPodResources(testPod, original, nil)

		ginkgo.By("waiting for the original values to be actuated")
		resizedPod = podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod, original)
		podresize.ExpectPodResized(ctx, f, resizedPod, original)

		ginkgo.By("deleting pod")
		podClient.DeleteSync(ctx, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
	})
}

func doPodResizeReadAndReplaceTests(f *framework.Framework) {
	/*
		Release: v1.35
		Testname: In-place Pod Resize, read and replace endpoints
		Description: Issuing a Pod Resize request via the Pod Resize subresource replace endpoint MUST result in the Pod resources being updated as expected. The Pod object fetched from the Pod Resize subresource MUST be equivalent to the Pod object fetched from the main Pod endpoint after the resize is completed.
	*/
	framework.ConformanceIt("resize pod via the replace endpoint [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
		podClient := e2epod.NewPodClient(f)
		original := []podresize.ResizableContainerInfo{{
			Name:      "c1",
			Resources: &cgroups.ContainerResources{CPUReq: originalCPU, MemReq: originalMem},
		}}
		desiredContainers := []podresize.ResizableContainerInfo{{
			Name:      "c1",
			Resources: &cgroups.ContainerResources{CPUReq: increasedCPU, MemReq: increasedMem},
		}}
		desired := v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse(increasedCPU),
				v1.ResourceMemory: resource.MustParse(increasedMem),
			},
		}

		ginkgo.By("creating and verifying pod")
		pod := createAndVerifyPod(ctx, f, podClient, original, nil, false)
		gomega.Expect(pod.Generation).To(gomega.BeEquivalentTo(1))

		podToUpdate, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get pod")
		podToUpdate.Spec.Containers[0].Resources = desired

		ginkgo.By("updating the pod resources")
		_, err = podClient.UpdateResize(ctx, pod.Name, podToUpdate, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "failed to resize pod")

		ginkgo.By("fetching updated pod")
		updatedPod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get pod")

		ginkgo.By("verifying pod resources")
		podresize.VerifyPodResources(updatedPod, desiredContainers, nil)
		gomega.Expect(updatedPod.Generation).To(gomega.BeEquivalentTo(2))

		ginkgo.By("verifying pod resources after patch")
		expected := podresize.UpdateExpectedContainerRestarts(ctx, updatedPod, desiredContainers)
		resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, updatedPod, expected)
		podresize.ExpectPodResized(ctx, f, resizedPod, expected)

		ginkgo.By("verifying pod fetched from resize subresource")
		framework.ExpectNoError(framework.Gomega().
			Eventually(ctx, framework.RetryNotFound(framework.GetObject(f.ClientSet.CoreV1().Pods(pod.Namespace).Get, pod.Name, metav1.GetOptions{}))).
			WithTimeout(f.Timeouts.PodStart).
			Should(framework.MakeMatcher(func(pod *v1.Pod) (func() string, error) {
				// For some reason, the pod object returned from the client doesn't have the GVK populated.
				pod.Kind = "Pod"
				pod.APIVersion = "v1"

				ginkgo.By("verifying pod fetched from resize subresource")
				podResource := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
				unstruct, err := f.DynamicClient.Resource(podResource).Namespace(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{}, "resize")
				if err != nil {
					return func() string {
						return fmt.Sprintf("failed to fetch pod after resize: %v", err.Error())
					}, nil
				}
				updatedPodFromResize, err := unstructuredToPod(unstruct)
				if err != nil {
					return func() string {
						return fmt.Sprintf("couldn't convert result to pod object: %v", err.Error())
					}, nil
				}
				if !apiequality.Semantic.DeepEqual(pod, updatedPodFromResize) {
					return func() string {
						return fmt.Sprintf("pod from resize subresource not equivalent to pod; %s", cmp.Diff(pod, updatedPodFromResize))
					}, nil
				}

				return nil, nil
			})),
		)
	})
}

// NOTE: Pod resize scheduler, resource quota, limit ranger, deferred resize tests are out of scope in e2e_node tests,
//
//	because in e2e_node tests
//	   a) scheduler and controller manager is not running by the Node e2e
//	   b) api-server in services doesn't start with --enable-admission-plugins=ResourceQuota
//	      and is not possible to start it from TEST_ARGS
//	Above tests in test/e2e/node/pod_resize.go
var _ = SIGDescribe("Pod InPlace Resize Container", func() {
	f := framework.NewDefaultFramework("pod-resize-tests")

	ginkgo.BeforeEach(func(ctx context.Context) {
		_, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		if framework.NodeOSDistroIs("windows") {
			e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
		}
	})

	doGuaranteedPodResizeTests(f)
	doBurstablePodResizeTests(f)
	doPodResizeReadAndReplaceTests(f)
	doPodResizePatchErrorTests(f)
	doPodResizeMemoryLimitDecreaseTest(f)
})

func doPatchAndRollback(ctx context.Context, f *framework.Framework, originalContainers, expectedContainers []podresize.ResizableContainerInfo, originalPodResources, expectedPodResources *v1.ResourceRequirements, doRollback bool, mountPodCgroup bool) {
	ginkgo.By("creating and verifying pod")
	podClient := e2epod.NewPodClient(f)
	newPod := createAndVerifyPod(ctx, f, podClient, originalContainers, originalPodResources, mountPodCgroup)

	ginkgo.By("patching and verifying pod for resize")
	patchAndVerify(ctx, f, podClient, newPod, originalContainers, expectedContainers, originalPodResources, expectedPodResources, "resize")
	if doRollback {
		// Resize has been actuated, test the reverse operation.
		rollbackContainers := createRollbackContainers(originalContainers, expectedContainers)
		ginkgo.By("patching and verifying pod for rollback")
		patchAndVerify(ctx, f, podClient, newPod, expectedContainers, rollbackContainers, expectedPodResources, originalPodResources, "rollback")
	}

	ginkgo.By("deleting pod")
	podClient.DeleteSync(ctx, newPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
}

func patchAndVerify(ctx context.Context, f *framework.Framework, podClient *e2epod.PodClient, newPod *v1.Pod, originalContainers, expectedContainers []podresize.ResizableContainerInfo, originalPodResources, expectedPodResources *v1.ResourceRequirements, opStr string) {
	patch := podresize.MakeResizePatch(originalContainers, expectedContainers, originalPodResources, expectedPodResources)
	patchedPod, pErr := f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
		types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
	framework.ExpectNoError(pErr, fmt.Sprintf("failed to patch pod for %s", opStr))

	expected := podresize.UpdateExpectedContainerRestarts(ctx, patchedPod, expectedContainers)

	podresize.VerifyPodResources(patchedPod, expected, expectedPodResources)
	resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, newPod, expected)

	podresize.ExpectPodResized(ctx, f, resizedPod, expected)
	if expectedPodResources != nil {
		framework.ExpectNoError(podresize.VerifyPodCgroupValues(ctx, f, resizedPod))
	}
}

func createAndVerifyPod(ctx context.Context, f *framework.Framework, podClient *e2epod.PodClient, originalContainers []podresize.ResizableContainerInfo, podResources *v1.ResourceRequirements, mountPodCgroup bool) *v1.Pod {
	tStamp := strconv.Itoa(time.Now().Nanosecond())
	testPod := podresize.MakePodWithResizableContainers(f.Namespace.Name, "", tStamp, originalContainers, podResources)
	testPod.GenerateName = "resize-test-"
	testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)

	// TODO: mount pod cgroup path for all pod resize related tests and remove
	// dependency on mountPodCgroup
	if mountPodCgroup {
		// mount pod cgroup in first container that can be used for pod cgroup values verification
		cgroups.ConfigureHostPathForPodCgroup(testPod)
	}

	newPod := podClient.CreateSync(ctx, testPod)
	podresize.VerifyPodResources(newPod, originalContainers, podResources)

	podresize.VerifyPodResizePolicy(newPod, originalContainers)
	framework.ExpectNoError(podresize.VerifyPodStatusResources(newPod, originalContainers))
	framework.ExpectNoError(podresize.VerifyPodContainersCgroupValues(ctx, f, newPod, originalContainers))
	if podResources != nil {
		framework.ExpectNoError(podresize.VerifyPodCgroupValues(ctx, f, newPod))
	}
	return newPod
}

func makeGuaranteedContainers(numContainers int,
	cpuPolicy, memPolicy v1.ResourceResizeRestartPolicy,
	initContainers, extendedResources bool,
	cpu, mem string) []podresize.ResizableContainerInfo {

	containers := []podresize.ResizableContainerInfo{}
	for i := range numContainers {
		// Offset the resources a bit so that in multi-container pods, not all containers have the same resources.
		resources := &cgroups.ContainerResources{
			CPUReq: offsetCPU(i, cpu),
			CPULim: offsetCPU(i, cpu),
			MemReq: offsetMemory(int64(i), mem),
			MemLim: offsetMemory(int64(i), mem),
		}

		if extendedResources {
			resources.ExtendedResourceReq = "1"
			resources.ExtendedResourceLim = "1"
		}

		containers = append(containers, podresize.ResizableContainerInfo{
			Name:      fmt.Sprintf("c%d", i+1),
			Resources: resources,
			CPUPolicy: &cpuPolicy,
			MemPolicy: &memPolicy,
		})
	}

	if initContainers {
		containers = append(containers, podresize.ResizableContainerInfo{
			Name:          "c1-init",
			Resources:     &cgroups.ContainerResources{CPUReq: cpu, CPULim: cpu, MemReq: mem, MemLim: mem},
			InitCtr:       true,
			RestartPolicy: v1.ContainerRestartPolicyAlways,
			CPUPolicy:     &cpuPolicy,
			MemPolicy:     &memPolicy,
		})
	}

	return containers
}

func makeBurstableContainers(numContainers int,
	cpuPolicy, memPolicy v1.ResourceResizeRestartPolicy,
	cpu, cpuLimit, mem, memLimit string) []podresize.ResizableContainerInfo {

	containers := []podresize.ResizableContainerInfo{}
	for i := range numContainers {
		// Offset the resources a bit so that in multi-container pods, not all containers have the same resources.
		resources := &cgroups.ContainerResources{CPUReq: offsetCPU(i, cpu), CPULim: offsetCPU(i, cpuLimit), MemReq: offsetMemory(int64(i), mem), MemLim: offsetMemory(int64(i), memLimit)}
		containers = append(containers, podresize.ResizableContainerInfo{
			Name:      fmt.Sprintf("c%d", i+1),
			Resources: resources,
			CPUPolicy: &cpuPolicy,
			MemPolicy: &memPolicy,
		})
	}
	return containers
}

func createRollbackContainers(originalContainers, expectedContainers []podresize.ResizableContainerInfo) []podresize.ResizableContainerInfo {
	rollbackContainers := make([]podresize.ResizableContainerInfo, len(originalContainers))
	copy(rollbackContainers, originalContainers)
	for i, c := range rollbackContainers {
		gomega.Expect(c.Name).To(gomega.Equal(expectedContainers[i].Name),
			"test case containers & expectations should be in the same order")
		// Resizes that trigger a restart should trigger a second restart when rolling back.
		rollbackContainers[i].RestartCount = expectedContainers[i].RestartCount
	}
	return rollbackContainers
}
