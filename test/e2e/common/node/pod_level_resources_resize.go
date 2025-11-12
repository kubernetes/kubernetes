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

package node

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"time"

	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/common/node/framework/cgroups"
	"k8s.io/kubernetes/test/e2e/common/node/framework/podresize"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

// TODO(ndixita): de-dup tests and common helpers from test/e2e/common/node/pod_resize.go
func doGuaranteedPodLevelResizeTests(f *framework.Framework) {
	ginkgo.DescribeTableSubtree("PLR guaranteed qos - 1 container with resize policy", func(cpuPolicy, memPolicy v1.ResourceResizeRestartPolicy, resizeInitCtrs bool) {
		ginkgo.DescribeTable("resizing", func(ctx context.Context, desiredCtrCPU, desiredCtrMem, desiredPodCPU, desiredPodMem string) {

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

			originalCtrCPU := originalCPU
			if desiredCtrCPU == "" {
				originalCtrCPU = ""
			}
			originalCtrMem := originalMem
			if desiredCtrMem == "" {
				originalCtrMem = ""
			}

			originalContainers := makeGuaranteedContainers(1, cpuPolicy, memPolicy, true, true, originalCtrCPU, originalCtrMem)
			expectedContainers := makeGuaranteedContainers(1, cpuPolicy, memPolicy, true, true, desiredCtrCPU, desiredCtrMem)
			for i, c := range expectedContainers {
				// If the pod has init containers, but we are not resizing them, keep the original resources.
				if c.InitCtr && !resizeInitCtrs {
					c.Resources = originalContainers[i].Resources
					expectedContainers[i] = c
					continue
				}
				// For containers where the resize policy is "restart", we expect a restart.
				expectRestart := int32(0)
				if cpuPolicy == v1.RestartContainer && desiredCtrCPU != originalCtrCPU {
					expectRestart = 1
				}
				if memPolicy == v1.RestartContainer && desiredCtrMem != originalCtrMem {
					expectRestart = 1
				}
				c.RestartCount = expectRestart
				expectedContainers[i] = c
			}

			var originalPodResources, desiredPodResources *v1.ResourceRequirements
			if desiredPodCPU != "" || desiredPodMem != "" {
				originalPodResources = makePodResources(offsetCPU(15, originalCPU), offsetCPU(15, originalCPU), offsetMemory(15, originalMem), offsetMemory(15, originalMem))
				desiredPodResources = makePodResources(offsetCPU(15, desiredPodCPU), offsetCPU(15, desiredPodCPU), offsetMemory(15, desiredPodMem), offsetMemory(15, desiredPodMem))
			}

			doPatchAndRollbackPLR(ctx, f, originalContainers, expectedContainers, originalPodResources, desiredPodResources, true, true)
		},
			// All tests will perform the requested resize, and once completed, will roll back the change.
			// This results in the coverage of both increase and decrease of
			// resources.
			// TODO: de-dup tests in pod_resize.go and this file
			ginkgo.Entry("cpu", increasedCPU, originalMem, "", ""),
			ginkgo.Entry("mem", originalCPU, increasedMem, "", ""),
			ginkgo.Entry("cpu & mem in the same direction", increasedCPU, increasedMem, "", ""),
			ginkgo.Entry("cpu & mem in opposite directions", increasedCPU, reducedMem, "", ""),
			ginkgo.Entry("pod-level cpu", originalCPU, originalMem, increasedCPU, originalMem),
			ginkgo.Entry("pod-level mem", originalCPU, originalMem, originalCPU, offsetMemory(10, increasedMem)),
			ginkgo.Entry("only pod-level cpu", "", "", increasedCPU, originalMem),
			ginkgo.Entry("only pod-level mem", "", "", originalCPU, increasedMem),
			ginkgo.Entry("pod-level cpu & mem in the same direction", originalCPU, originalMem, increasedCPU, increasedMem),
			ginkgo.Entry("pod-level cpu & mem in opposite directions", originalCPU, originalMem, increasedCPU, reducedMem),
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
	ginkgo.Describe("pod-level guaranteed pods with multiple containers", func() {
		/*
			Release: v1.35
			Testname: In-place Pod Resize, guaranteed pods with multiple containers, net increase
			Description: Issuing an in-place Pod Resize request via the Pod Resize subresource patch endpoint to modify CPU and memory requests and limits for a guaranteed pod with 3 containers with a net increase MUST result in the Pod resources being updated as expected.
		*/
		framework.It("3 containers - increase cpu & mem on c1, c2, decrease cpu & mem on c3 - net increase [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
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

			doPatchAndRollbackPLR(ctx, f, originalContainers, expectedContainers, nil, nil, true, true)
		})

		/*
			Release: v1.35
			Testname: In-place Pod Resize, guaranteed pods with multiple containers, net decrease
			Description: Issuing an in-place Pod Resize request via the Pod Resize subresource patch endpoint to modify CPU and memory requests and limits for a pod with 3 containers with a net decrease MUST result in the Pod resources being updated as expected.
		*/
		framework.It("3 containers - increase cpu & mem on c1, decrease cpu & mem on c2, c3 - net decrease [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
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

			doPatchAndRollbackPLR(ctx, f, originalContainers, expectedContainers, nil, nil, true, true)
		})

		/*
			Release: v1.35
			Testname: In-place Pod Resize, guaranteed pods with multiple containers, various operations
			Description: Issuing an in-place Pod Resize request via the Pod Resize subresource patch endpoint to modify CPU and memory requests and limits for a pod with 3 containers with various operations MUST result in the Pod resources being updated as expected.
		*/
		framework.It("3 containers - increase: CPU (c1,c3), memory (c2, c3) ; decrease: CPU (c2) [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
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

			doPatchAndRollbackPLR(ctx, f, originalContainers, expectedContainers, nil, nil, true, true)
		})
	})
}

func doBurstablePodLevelResizeTests(f *framework.Framework) {
	ginkgo.DescribeTableSubtree("PLR pod-level burstable pods - 1 container with all requests & limits set and resize policy", func(cpuPolicy, memPolicy v1.ResourceResizeRestartPolicy) {
		ginkgo.DescribeTable("resizing", func(ctx context.Context, desiredContainerResources, desiredPodLevelResources resourceRequestsLimits) {
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

			var originalPodResources, desiredPodResources *v1.ResourceRequirements
			if desiredPodLevelResources != (resourceRequestsLimits{}) {
				originalPodResources = makePodResources(offsetCPU(30, originalCPU), offsetCPU(30, originalCPULimit), offsetMemory(30, originalMem), offsetMemory(30, originalMemLimit))
				desiredPodResources = makePodResources(offsetCPU(30, desiredPodLevelResources.cpuReq), offsetCPU(30, desiredPodLevelResources.cpuLim), offsetMemory(30, desiredPodLevelResources.memReq), offsetMemory(30, desiredPodLevelResources.memLim))
			}
			doPatchAndRollbackPLR(ctx, f, originalContainers, expectedContainers, originalPodResources, desiredPodResources, true, true)
		},
			// All tests will perform the requested resize, and once completed, will roll back the change.
			// This results in the coverage of both increase and decrease of resources.
			ginkgo.Entry("cpu requests", resourceRequestsLimits{increasedCPU, originalCPULimit, originalMem, originalMemLimit}, resourceRequestsLimits{}),
			ginkgo.Entry("cpu limits", resourceRequestsLimits{originalCPU, increasedCPULimit, originalMem, originalMemLimit}, resourceRequestsLimits{}),
			ginkgo.Entry("mem requests", resourceRequestsLimits{originalCPU, originalCPULimit, increasedMem, originalMemLimit}, resourceRequestsLimits{}),
			ginkgo.Entry("mem limits", resourceRequestsLimits{originalCPU, originalCPULimit, originalMem, increasedMemLimit}, resourceRequestsLimits{}),
			ginkgo.Entry("all resources in the same direction", resourceRequestsLimits{increasedCPU, increasedCPULimit, increasedMem, increasedMemLimit}, resourceRequestsLimits{}),
			ginkgo.Entry("cpu & mem in opposite directions", resourceRequestsLimits{increasedCPU, increasedCPULimit, reducedMem, reducedMemLimit}, resourceRequestsLimits{}),
			ginkgo.Entry("requests & limits in opposite directions", resourceRequestsLimits{reducedCPU, increasedCPULimit, increasedMem, reducedMemLimit}, resourceRequestsLimits{}),
			ginkgo.Entry("pod-level cpu requests", resourceRequestsLimits{originalCPU, originalCPULimit, originalMem, originalMemLimit}, resourceRequestsLimits{increasedCPU, originalCPULimit, originalMem, originalMemLimit}),
			ginkgo.Entry("pod-level cpu limits", resourceRequestsLimits{originalCPU, originalCPULimit, originalMem, originalMemLimit}, resourceRequestsLimits{originalCPU, increasedCPULimit, originalMem, originalMemLimit}),
			ginkgo.Entry("pod-level mem requests", resourceRequestsLimits{originalCPU, originalCPULimit, originalMem, originalMemLimit}, resourceRequestsLimits{originalCPU, originalCPULimit, increasedMem, originalMemLimit}),
			ginkgo.Entry("pod-level mem limits", resourceRequestsLimits{originalCPU, originalCPULimit, originalMem, originalMemLimit}, resourceRequestsLimits{originalCPU, originalCPULimit, originalMem, increasedMemLimit}),
			ginkgo.Entry("pod-level all resources in the same direction", resourceRequestsLimits{originalCPU, originalCPULimit, originalMem, originalMemLimit}, resourceRequestsLimits{increasedCPU, increasedCPULimit, increasedMem, increasedMemLimit}),
			ginkgo.Entry("pod-level cpu & mem in opposite directions", resourceRequestsLimits{originalCPU, originalCPULimit, originalMem, originalMemLimit}, resourceRequestsLimits{increasedCPU, increasedCPULimit, reducedMem, reducedMemLimit}),
			ginkgo.Entry("pod-level requests & limits in opposite directions", resourceRequestsLimits{originalCPU, originalCPULimit, originalMem, originalMemLimit}, resourceRequestsLimits{reducedCPU, increasedCPULimit, increasedMem, reducedMemLimit}),
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
			Testname: In-place Pod Resize, burstable pod with multiple containers and various operations
			Description: Issuing a Pod Resize request via the Pod Resize subresource patch endpoint to modify CPU and memory requests and limits on a 6-container pod with various operations MUST result in the Pod resources being updated as expected.
		*/
		framework.It("6 containers - various operations performed (including adding limits and requests) [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
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
			doPatchAndRollbackPLR(ctx, f, originalContainers, expectedContainers, nil, nil, false, true)
		})

		/*
			Release: v1.35
			Testname: In-place Pod Resize, burstable pod resized with equivalents
			Description: Issuing an in-place Pod Resize request via the Pod Resize subresource patch endpoint to modify CPU requests and limits using equivalent values (e.g. 2m -> 1m) MUST result in the updated Pod resources displayed correctly in the status.
		*/
		framework.It("resize with equivalents [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
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
			doPatchAndRollbackPLR(ctx, f, originalContainers, expectedContainers, nil, nil, true, true)
		})
	})

	ginkgo.DescribeTable("burstable pods - pod-level resources", func(ctx context.Context, originalContainers, expectedContainers []podresize.ResizableContainerInfo, originalPodLevelResources, expectedPodLevelResources resourceRequestsLimits, doRollback bool) {

		originalPodResources := makePodResources(originalPodLevelResources.cpuReq, originalPodLevelResources.cpuLim, originalPodLevelResources.memReq, originalPodLevelResources.memLim)
		desiredPodResources := makePodResources(expectedPodLevelResources.cpuReq, expectedPodLevelResources.cpuLim, expectedPodLevelResources.memReq, expectedPodLevelResources.memLim)

		doPatchAndRollbackPLR(ctx, f, originalContainers, expectedContainers, originalPodResources, desiredPodResources, doRollback, true)
	},
		ginkgo.Entry("pod-level resize with equivalents",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "2m", CPULim: "10m"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "2m", CPULim: "10m"},
				},
			},
			resourceRequestsLimits{cpuReq: "4m", cpuLim: "20m"},
			resourceRequestsLimits{cpuReq: "5m", cpuLim: "25m"},
			true,
		),
		ginkgo.Entry("pod-level resize with limits add",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "2m", CPULim: "10m"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "2m", CPULim: "10m"},
				},
			},
			resourceRequestsLimits{cpuReq: "4m"},
			resourceRequestsLimits{cpuReq: "5m", cpuLim: "25m"},
			false,
		),
		ginkgo.Entry("pod-level resize with requests and limits add",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "2m", CPULim: "10m"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "2m", CPULim: "10m"},
				},
			},
			resourceRequestsLimits{},
			resourceRequestsLimits{cpuReq: "5m", cpuLim: "25m"},
			false,
		),
		ginkgo.Entry("pod-level resize with no container requests and limits",
			[]podresize.ResizableContainerInfo{
				{
					Name: "c1",
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name: "c1",
				},
			},
			resourceRequestsLimits{cpuReq: "4m", memReq: "10m"},
			resourceRequestsLimits{cpuReq: "5m", memReq: "25m"},
			false,
		),
	)
}

var _ = SIGDescribe("PLR Pod InPlace Resize", framework.WithFeatureGate(features.InPlacePodLevelResourcesVerticalScaling), func() {
	f := framework.NewDefaultFramework("pod-level-resources-resize-tests")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.BeforeEach(func(ctx context.Context) {
		_, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		if framework.NodeOSDistroIs("windows") {
			e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
		}
	})

	doGuaranteedPodLevelResizeTests(f)
	doBurstablePodLevelResizeTests(f)
	doPodLevelResourcesMemoryLimitDecreaseTest(f)
})

func makePodResources(cpuReq, cpuLim, memReq, memLim string) *v1.ResourceRequirements {
	if cpuReq == "" && memReq == "" && cpuLim == "" && memLim == "" {
		return nil
	}

	resources := &v1.ResourceRequirements{}

	if cpuLim != "" || memLim != "" {
		resources.Limits = v1.ResourceList{}
	}

	if cpuLim != "" {
		resources.Limits[v1.ResourceCPU] = resource.MustParse(cpuLim)
	}

	if memLim != "" {
		resources.Limits[v1.ResourceMemory] = resource.MustParse(memLim)

	}

	if cpuReq != "" || memReq != "" {
		resources.Requests = v1.ResourceList{}
	}

	if cpuReq != "" {
		resources.Requests[v1.ResourceCPU] = resource.MustParse(cpuReq)
	}

	if memReq != "" {
		resources.Requests[v1.ResourceMemory] = resource.MustParse(memReq)

	}
	return resources
}

func VerifyPodLevelStatus(gotPod *v1.Pod) error {
	var errs []error
	var wantStatusResources *v1.ResourceRequirements

	if gotPod.Status.Phase != v1.PodRunning {
		wantStatusResources = gotPod.Spec.Resources
		if err := framework.Gomega().Expect(gotPod.Status.Resources).To(gomega.BeComparableTo(wantStatusResources)); err != nil {
			errs = append(errs, fmt.Errorf("pod.status.resources mismatch: %w", err))
		}
	}
	// TODO(ndixita) - add tests when to compare Pod.Status.Resources with pod
	// cgroup config values. Converting cgroup values -> Pod.Status.Resources ->
	// cgroup values is resulting in indeterministic rounded off values

	var wantAllocatedResources v1.ResourceList
	aggrReq, _ := podresize.AggregateContainerResources(gotPod.Spec)
	wantAllocatedResources = aggrReq

	if gotPod.Spec.Resources != nil {
		wantAllocatedResources = gotPod.Spec.Resources.Requests
	}

	if err := framework.Gomega().Expect(gotPod.Status.AllocatedResources).To(gomega.BeComparableTo(wantAllocatedResources)); err != nil {
		errs = append(errs, fmt.Errorf("pod.status.allocatedResources mismatch: %w", err))
	}

	return utilerrors.NewAggregate(errs)
}

func doPodLevelResourcesMemoryLimitDecreaseTest(f *framework.Framework) {
	// Tests the behavior when decreasing pod-level memory limit:
	// 1. Decrease the limit a little bit - should succeed
	// 2. Decrease the limit down to a tiny amount - should fail
	// 3. Revert the limit back to the original value - should succeed
	ginkgo.It("decrease pod-level memory limit below usage", func(ctx context.Context) {
		podClient := e2epod.NewPodClient(f)
		originalPLR := &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse(originalMem),
			},
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse(originalMem),
			},
		}

		containers := []podresize.ResizableContainerInfo{{
			Name: "c1",
		}}

		ginkgo.By("creating and verifying pod")
		testPod := createAndVerifyPodPLR(ctx, f, podClient, containers, originalPLR, true)

		// 1. Decrease the limit a little bit - should succeed
		ginkgo.By("Patching pod with a slightly lowered memory limit")
		viableLoweredLimitPLR := &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse(reducedMem),
			},
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse(reducedMem),
			},
		}
		patch := podresize.MakeResizePatch(containers, containers, originalPLR, viableLoweredLimitPLR)
		testPod, pErr := f.ClientSet.CoreV1().Pods(testPod.Namespace).Patch(ctx, testPod.Name,
			types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(pErr, "failed to patch pod for viable lowered limit")

		ginkgo.By("verifying pod patched for viable lowered limit")
		podresize.VerifyPodResources(testPod, containers, viableLoweredLimitPLR)

		ginkgo.By("waiting for viable lowered limit to be actuated")
		resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod, containers)
		podresize.ExpectPodResized(ctx, f, resizedPod, containers)

		// There is some latency after container startup before memory usage is scraped. On CRI-O
		// this latency is much higher, so wait enough time for cAdvisor to scrape metrics twice.
		ginkgo.By("Waiting for stats scraping")
		const scrapingDelay = 30 * time.Second // 2 * maxHousekeepingInterval
		startTime := testPod.Status.StartTime
		time.Sleep(time.Until(startTime.Add(scrapingDelay)))

		// 2. Decrease the limit down to a tiny amount - should fail
		const nonViableMemoryLimit = "10Ki"
		ginkgo.By("Patching pod with a greatly lowered memory limit")
		nonViableLoweredLimitPLR := &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse(nonViableMemoryLimit),
			},
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse(nonViableMemoryLimit),
			},
		}

		patch = podresize.MakeResizePatch(containers, containers, viableLoweredLimitPLR, nonViableLoweredLimitPLR)
		testPod, pErr = f.ClientSet.CoreV1().Pods(testPod.Namespace).Patch(ctx, testPod.Name,
			types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(pErr, "failed to patch pod for viable lowered limit")

		framework.ExpectNoError(framework.Gomega().
			Eventually(ctx, framework.RetryNotFound(framework.GetObject(f.ClientSet.CoreV1().Pods(testPod.Namespace).Get, testPod.Name, metav1.GetOptions{}))).
			WithTimeout(f.Timeouts.PodStart).
			Should(framework.MakeMatcher(func(pod *v1.Pod) (func() string, error) {
				// If VerifyPodLevelStatusResources succeeds, it means the resize completed.
				if podresize.VerifyPodLevelStatusResources(pod, nonViableLoweredLimitPLR) == nil {
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
		framework.ExpectNoError(podresize.VerifyPodStatusResources(testPod, containers))

		// 3. Revert the limit back to the original value - should succeed
		ginkgo.By("Patching pod to revert to original state")
		patch = podresize.MakeResizePatch(containers, containers, viableLoweredLimitPLR, originalPLR)
		testPod, pErr = f.ClientSet.CoreV1().Pods(testPod.Namespace).Patch(ctx, testPod.Name,
			types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(pErr, "failed to patch pod back to original values")

		ginkgo.By("verifying pod patched for original values")
		podresize.VerifyPodResources(testPod, containers, originalPLR)

		ginkgo.By("waiting for the original values to be actuated")
		resizedPod = podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod, containers)
		podresize.ExpectPodResized(ctx, f, resizedPod, containers)

		ginkgo.By("deleting pod")
		podClient.DeleteSync(ctx, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
	})
}

func doPatchAndRollbackPLR(ctx context.Context, f *framework.Framework, originalContainers, expectedContainers []podresize.ResizableContainerInfo, originalPodResources, expectedPodResources *v1.ResourceRequirements, doRollback bool, mountPodCgroup bool) {
	ginkgo.By("creating and verifying pod")
	podClient := e2epod.NewPodClient(f)
	newPod := createAndVerifyPodPLR(ctx, f, podClient, originalContainers, originalPodResources, mountPodCgroup)

	if expectedPodResources != nil {
		framework.ExpectNoError(VerifyPodLevelStatus(newPod))
	}
	ginkgo.By(fmt.Sprintf("patching and verifying pod for resize %s: %v", newPod.Name, newPod.UID))
	patchAndVerifyPLR(ctx, f, podClient, newPod, originalContainers, expectedContainers, originalPodResources, expectedPodResources, "resize")
	if doRollback {
		// Resize has been actuated, test the reverse operation.
		rollbackContainers := createRollbackContainers(originalContainers, expectedContainers)
		ginkgo.By("patching and verifying pod for rollback")
		patchAndVerify(ctx, f, podClient, newPod, expectedContainers, rollbackContainers, expectedPodResources, originalPodResources, "rollback")
	}

	ginkgo.By("deleting pod")
	podClient.DeleteSync(ctx, newPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
}

func patchAndVerifyPLR(ctx context.Context, f *framework.Framework, podClient *e2epod.PodClient, newPod *v1.Pod, originalContainers, expectedContainers []podresize.ResizableContainerInfo, originalPodResources, expectedPodResources *v1.ResourceRequirements, opStr string) {
	patch := podresize.MakeResizePatch(originalContainers, expectedContainers, originalPodResources, expectedPodResources)
	patchedPod, pErr := f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
		types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
	framework.ExpectNoError(pErr, fmt.Sprintf("failed to patch pod for %s", opStr))

	expected := podresize.UpdateExpectedContainerRestarts(ctx, patchedPod, expectedContainers)

	podresize.VerifyPodResources(patchedPod, expected, expectedPodResources)
	resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, newPod, expected)
	podresize.ExpectPodResized(ctx, f, resizedPod, expected)
	// Uncomment pod-level status verification after patch in 1.36 release.
	// convesion of cgroup values -> Pod.Status.Resources -> cgroup values is
	// resulting in values off by a small number.
	// framework.ExpectNoError(VerifyPodLevelStatus(resizedPod))
	if expectedPodResources != nil {
		framework.ExpectNoError(podresize.VerifyPodCgroupValues(ctx, f, resizedPod))
	}
}

func createAndVerifyPodPLR(ctx context.Context, f *framework.Framework, podClient *e2epod.PodClient, originalContainers []podresize.ResizableContainerInfo, podResources *v1.ResourceRequirements, mountPodCgroup bool) *v1.Pod {
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
