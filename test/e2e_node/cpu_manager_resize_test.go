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

package e2enode

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gcustom"
	"github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
	helpers "k8s.io/component-helpers/resource"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/test/e2e/common/node/framework/cgroups"
	"k8s.io/kubernetes/test/e2e/common/node/framework/podresize"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"

	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
)

type containerCPUInfo struct {
	Name       string
	InitCtr    bool
	SidecarCtr bool
	cpuCount   int
}

var _ = SIGDescribe("CPU Manager with InPlacePodVerticalScalingExclusiveCPUs enabled", ginkgo.Ordered, ginkgo.ContinueOnFailure, framework.WithSerial(), feature.CPUManager, feature.InPlacePodVerticalScaling, feature.InPlacePodVerticalScalingExclusiveCPUs, framework.WithFeatureGate(features.InPlacePodVerticalScalingExclusiveCPUs), func() {
	f := framework.NewDefaultFramework("cpu-manager-pod-resize-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// original kubeletconfig before the context start, to be restored
	var oldCfg *kubeletconfig.KubeletConfiguration
	var reservedCPUs cpuset.CPUSet
	var onlineCPUs cpuset.CPUSet
	var smtLevel int
	var uncoreGroupSize int
	// tracks all the pods created by a It() block. Best would be a namespace per It block
	var podMap map[string]*v1.Pod
	var skipIfAllocatableCPUsLessThan func(node *v1.Node, cpuReq int)

	var createPodSync func(ctx context.Context, pod *v1.Pod) *v1.Pod

	var testPodResize func(ctx context.Context,
		baselineContainers []podresize.ResizableContainerInfo,
		baselineCpuInfo []containerCPUInfo,
		desiredContainersFirstPatch []podresize.ResizableContainerInfo,
		expectedContainersFirstPatch []podresize.ResizableContainerInfo,
		expectedCpuInfoFirstPatch []containerCPUInfo,
		wantPatchErrorFirstPatch string,
		wantActuationErrorFirstPatch string,
		desiredContainersSecondPatch []podresize.ResizableContainerInfo,
		expectedContainersSecondPatch []podresize.ResizableContainerInfo,
		expectedCpuInfoSecondPatch []containerCPUInfo,
		wantPatchErrorSecondPatch string,
		wantActuationErrorSecondPatch string,
		podMap map[string]*v1.Pod)

	ginkgo.BeforeAll(func(ctx context.Context) {
		var err error
		oldCfg, err = getCurrentKubeletConfig(ctx)
		framework.ExpectNoError(err)

		onlineCPUs, err = getOnlineCPUs() // this should not change at all, at least during this suite lifetime
		framework.ExpectNoError(err)
		framework.Logf("Online CPUs: %s", onlineCPUs)

		uncoreGroupSize = getUncoreCPUGroupSize()
		framework.Logf("Uncore Group Size: %d", uncoreGroupSize)

		e2enodeCgroupV2Enabled = IsCgroup2UnifiedMode()
		framework.Logf("cgroup V2 enabled: %v", e2enodeCgroupV2Enabled)

		e2enodeCgroupDriver = oldCfg.CgroupDriver
		framework.Logf("cgroup driver: %s", e2enodeCgroupDriver)

		runtime, _, err := getCRIClient(ctx)
		framework.ExpectNoError(err, "Failed to get CRI client")

		version, err := runtime.Version(ctx, "")
		framework.ExpectNoError(err, "Failed to get runtime version")

		e2enodeRuntimeName = version.GetRuntimeName()
		framework.Logf("runtime: %s", e2enodeRuntimeName)
	})

	ginkgo.AfterAll(func(ctx context.Context) {
		updateKubeletConfig(ctx, f, oldCfg, true)
	})

	ginkgo.BeforeEach(func(ctx context.Context) {
		// note intentionally NOT set reservedCPUs -  this must be initialized on a test-by-test basis
		reservedCPUs = cpuset.New()
		podMap = make(map[string]*v1.Pod)
		createPodSync = func(ctx context.Context, pod *v1.Pod) *v1.Pod {
			newPod := e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(newPod.UID)] = newPod
			return newPod
		}

		testPodResize = func(ctx context.Context,
			baselineContainers []podresize.ResizableContainerInfo,
			baselineCpuInfo []containerCPUInfo,
			desiredContainersFirstPatch []podresize.ResizableContainerInfo,
			expectedContainersFirstPatch []podresize.ResizableContainerInfo,
			expectedCpuInfoFirstPatch []containerCPUInfo,
			wantPatchErrorFirstPatch string,
			wantActuationErrorFirstPatch string,
			desiredContainersSecondPatch []podresize.ResizableContainerInfo,
			expectedContainersSecondPatch []podresize.ResizableContainerInfo,
			expectedCpuInfoSecondPatch []containerCPUInfo,
			wantPatchErrorSecondPatch string,
			wantActuationErrorSecondPatch string,
			podMap map[string]*v1.Pod) {

			// The API server rejects a resize when the pod CPU requests exceed the node allocatable CPUs
			// (plugin/pkg/admission/podresize). Skip if the node is big enough to accept the patch.
			skipIfPatchFitsAllocatable := func(node *v1.Node, desiredContainers []podresize.ResizableContainerInfo, wantPatchError string) {
				ginkgo.GinkgoHelper()
				if wantPatchError == "" || desiredContainers == nil {
					return
				}
				desiredPod := podresize.MakePodWithResizableContainers(f.Namespace.Name, "desired", "", desiredContainers, nil)
				podCPURequests := helpers.PodRequests(desiredPod, helpers.PodResourcesOptions{})[v1.ResourceCPU]
				allocatableCPUs := node.Status.Allocatable[v1.ResourceCPU]
				if podCPURequests.Cmp(allocatableCPUs) <= 0 {
					e2eskipper.Skipf("Skipping test: expected patch error %q, but pod CPU requests %s fit within node allocatable CPUs %s",
						wantPatchError, podCPURequests.String(), allocatableCPUs.String())
				}
			}

			nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
			framework.ExpectNoError(err, "failed to get running nodes")
			gomega.Expect(nodes.Items).ShouldNot(gomega.BeEmpty())
			framework.Logf("Found %d schedulable nodes", len(nodes.Items))

			ginkgo.By("Find node CPU resources available for allocation!")
			node := getLocalNode(ctx, f)
			cpuCap, cpuAlloc, _ := getLocalNodeCPUDetails(ctx, f)
			framework.Logf("Node '%s': NodeAllocatableCPUs = %d, currently available to allocate = %d.",
				node.Name, cpuCap, cpuAlloc)

			skipIfPatchFitsAllocatable(node, desiredContainersFirstPatch, wantPatchErrorFirstPatch)
			skipIfPatchFitsAllocatable(node, desiredContainersSecondPatch, wantPatchErrorSecondPatch)

			testPod1 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod1", strconv.Itoa(time.Now().Nanosecond()), baselineContainers, nil)
			testPod1 = e2epod.MustMixinRestrictedPodSecurity(testPod1)
			ginkgo.By("creating test pod 1 with multiple containers")
			testPod1 = createPodSync(ctx, testPod1)
			podClient := e2epod.PodClientNS(f, testPod1.Namespace)

			ginkgo.By("verifying baseline pod resources, allocations and policy are as expected")
			podresize.VerifyPodResources(testPod1, baselineContainers, nil)

			ginkgo.By("verifying baseline pod cpusets are as expected")
			checkAllocatedCPUsForResize(ctx, f, baselineCpuInfo, testPod1)

			cpuCapAfterPodCreate, cpuAllocAfterPodCreate, _ := getLocalNodeCPUDetails(ctx, f)
			framework.Logf("Node '%s': NodeAllocatableCPUs = %d, currently available to allocate = %d.",
				node.Name, cpuCapAfterPodCreate, cpuAllocAfterPodCreate)

			ginkgo.By("patching pod for resize")
			patchString := podresize.MakeResizePatch(baselineContainers, desiredContainersFirstPatch, nil, nil)

			patchedPod, pErr := podClient.Patch(ctx,
				testPod1.Name, apimachinerytypes.StrategicMergePatchType, []byte(patchString), metav1.PatchOptions{}, "resize")

			if wantPatchErrorFirstPatch != "" {
				ginkgo.By("ensuring the testing pod is failed for the expected reason for first patch")
				gomega.Expect(pErr).To(gomega.MatchError(gomega.MatchRegexp(wantPatchErrorFirstPatch)))

				ginkgo.By("verifying pod cpusets after first non patched resize")
				checkAllocatedCPUsForResize(ctx, f, expectedCpuInfoFirstPatch, testPod1)
				return
			}

			framework.ExpectNoError(pErr, "failed to patch pod for resize")

			if wantActuationErrorFirstPatch != "" {
				ginkgo.By("waiting for testing pod first resize status to be infeasible")
				WaitForPodResizeInfeasible(ctx, f, patchedPod)

				infeasibleResizePod, err := framework.GetObject(podClient.Get, patchedPod.Name, metav1.GetOptions{})(ctx)
				framework.ExpectNoError(err, "failed to get infeasible to resize pod")

				ginkgo.By("ensuring the testing pod is failed for the expected reason")
				gomega.Expect(infeasibleResizePod).To(HaveStatusConditionsMatchingRegex(wantActuationErrorFirstPatch))

				ginkgo.By("verifying pod cpusets after infeasible resize, are as expected")
				checkAllocatedCPUsForResize(ctx, f, expectedCpuInfoFirstPatch, testPod1)
				return
			}

			expected := podresize.UpdateExpectedContainerRestarts(ctx, patchedPod, expectedContainersFirstPatch)
			ginkgo.By("verifying pod resources before first resize are as expected")
			podresize.VerifyPodResources(patchedPod, expected, nil)

			ginkgo.By("waiting for first resize to be complete and the pod to become ready")
			resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod1, nil)

			ginkgo.By("verifying pod resources after first resize are as expected")
			podresize.VerifyPodResources(resizedPod, expected, nil)

			ginkgo.By("verifying pod in use cpusets after first resize are as expected")
			checkAllocatedCPUsForResize(ctx, f, expectedCpuInfoFirstPatch, testPod1)

			cpuCapAfterFirstPatch, cpuAllocAfterFirstPatch, _ := getLocalNodeCPUDetails(ctx, f)
			framework.Logf("Node '%s': NodeAllocatableCPUs = %d, currently available to allocate = %d.",
				node.Name, cpuCapAfterFirstPatch, cpuAllocAfterFirstPatch)

			if desiredContainersSecondPatch == nil {
				return
			}

			ginkgo.By("patching again pod for resize")
			secondPatchString := podresize.MakeResizePatch(expected, desiredContainersSecondPatch, nil, nil)

			patchedPod, pErr = podClient.Patch(ctx,
				testPod1.Name, apimachinerytypes.StrategicMergePatchType, []byte(secondPatchString), metav1.PatchOptions{}, "resize")

			if wantPatchErrorSecondPatch != "" {
				ginkgo.By("ensuring the testing pod is failed for the expected reason for second patch")
				gomega.Expect(pErr).To(gomega.MatchError(gomega.MatchRegexp(wantPatchErrorSecondPatch)))

				ginkgo.By("verifying pod cpusets after second non patched resize")
				checkAllocatedCPUsForResize(ctx, f, expectedCpuInfoSecondPatch, testPod1)
				return
			}

			framework.ExpectNoError(pErr, "failed to patch again pod for resize")

			ginkgo.By("waiting for second resize to be complete and the pod to become ready")
			if wantActuationErrorSecondPatch != "" {
				ginkgo.By("waiting for second resize to become infeasible")
				WaitForPodResizeInfeasible(ctx, f, patchedPod)

				infeasibleResizePod, err := framework.GetObject(podClient.Get, patchedPod.Name, metav1.GetOptions{})(ctx)
				framework.ExpectNoError(err, "failed to get infeasible to resize pod")

				ginkgo.By("ensuring the testing pod is failed for the expected reason for second patch")
				gomega.Expect(infeasibleResizePod).To(HaveStatusConditionsMatchingRegex(wantActuationErrorSecondPatch))

				ginkgo.By("verifying pod in use cpusets, after infeasible second resize, are as expected")
				checkAllocatedCPUsForResize(ctx, f, expectedCpuInfoSecondPatch, testPod1)
				return
			}

			cpuCapAfterSecondPatch, cpuAllocAfterSecondPatch, _ := getLocalNodeCPUDetails(ctx, f)
			framework.Logf("Node '%s': NodeAllocatableCPUs = %d, currently available to allocate = %d.",
				node.Name, cpuCapAfterSecondPatch, cpuAllocAfterSecondPatch)

			expected = podresize.UpdateExpectedContainerRestarts(ctx, patchedPod, expectedContainersSecondPatch)
			ginkgo.By("verifying pod resources are as expected post second patch, pre-actuation")
			podresize.VerifyPodResources(patchedPod, expected, nil)

			ginkgo.By("waiting for second patch resize to be actuated")
			resizedPod = podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod1, nil)

			ginkgo.By("verifying pod resources after second resize")
			podresize.VerifyPodResources(resizedPod, expected, nil)

			ginkgo.By("verifying pod cpusets after second actuated resize")
			checkAllocatedCPUsForResize(ctx, f, expectedCpuInfoSecondPatch, testPod1)

		}
	})

	ginkgo.JustBeforeEach(func(ctx context.Context) {
		if !e2enodeCgroupV2Enabled {
			e2eskipper.Skipf("Skipping since CgroupV2 not used")
		}
		// use a closure to minimize the arguments, to make the usage more straightforward
		skipIfAllocatableCPUsLessThan = func(node *v1.Node, val int) {
			ginkgo.GinkgoHelper()
			checkAllocatableCPUs(node, val, reservedCPUs, onlineCPUs, smtLevel)
		}
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		deletePodsAsync(ctx, f, podMap)
	})

	ginkgo.When("resizing a Guaranteed Pod with a single container and integer CPU requests", ginkgo.Label("single container guaranteed pod with integer CPU requests resize", "exclusive-cpus"), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			reservedCPUs = cpuset.New(0)
		})
		ginkgo.DescribeTable("",
			func(ctx context.Context,
				baselineContainers []podresize.ResizableContainerInfo,
				baselineCpuInfo []containerCPUInfo,
				desiredContainers []podresize.ResizableContainerInfo,
				expectedContainers []podresize.ResizableContainerInfo,
				expectedCpuInfo []containerCPUInfo,
				wantPatchError string,
				wantActuationError string,
			) {

				baselineExpectedCPUCount := 0
				for i := range baselineCpuInfo {
					baselineExpectedCPUCount += baselineCpuInfo[i].cpuCount
				}
				firstPatchExpectedCPUCount := 0
				for i := range expectedCpuInfo {
					firstPatchExpectedCPUCount += expectedCpuInfo[i].cpuCount
				}

				skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), max(baselineExpectedCPUCount, firstPatchExpectedCPUCount))

				updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
					policyName:         string(cpumanager.PolicyStatic),
					reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
					enableInPlacePodVerticalScalingExclusiveCPUs: true,
				}))

				testPodResize(ctx,
					baselineContainers,
					baselineCpuInfo,
					desiredContainers,
					expectedContainers,
					expectedCpuInfo,
					wantPatchError,
					wantActuationError,
					nil,
					nil,
					nil,
					"",
					"",
					podMap)
			},
			ginkgo.Entry("should increase exclusively CPUs, within available capacity",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "300Mi", MemLim: "300Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "3000m", CPULim: "3000m", MemReq: "300Mi", MemLim: "300Mi"},
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "3000m", CPULim: "3000m", MemReq: "300Mi", MemLim: "300Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 3,
					},
				},
				"",
				"",
			),
			ginkgo.Entry("should increase the CPU request/limit, decrease memory request/limit, within available capacity",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "400Mi", MemLim: "400Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 4,
					},
				},
				"",
				"",
			),
			ginkgo.Entry("should increase the CPU request/limit, increase memory request/limit, within available capacity",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "400Mi", MemLim: "400Mi"},
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "400Mi", MemLim: "400Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 4,
					},
				},
				"",
				"",
			),
			ginkgo.Entry("should not decrease allocated exclusively CPUs, below baseline cpuset",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "300Mi", MemLim: "300Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "300Mi", MemLim: "300Mi"},
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "300Mi", MemLim: "300Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
				},
				"",
				"Infeasible.*not allowed to reduce container exclusively allocated pool below baseline.*",
			),
			ginkgo.Entry("should not increase allocated exclusively CPUs, beyond available capacity",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "300Mi", MemLim: "300Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "1000000m", CPULim: "1000000m", MemReq: "300Mi", MemLim: "300Mi"},
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "300Mi", MemLim: "300Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
				},
				".*node didn't have enough allocatable resources.*",
				"",
			),
		)
	})

	ginkgo.When("topologyManagerPolicy is set to none, resizing a Guaranteed multiple containers Pod with integer CPU requests", ginkgo.Label("guaranteed multiple container pod with integer CPU requests resize", "exclusive-cpus"), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			reservedCPUs = cpuset.New(0)
		})
		ginkgo.DescribeTable("",
			func(ctx context.Context,
				baselineContainers []podresize.ResizableContainerInfo,
				baselineCpuInfo []containerCPUInfo,
				desiredContainersFirstPatch []podresize.ResizableContainerInfo,
				expectedContainersFirstPatch []podresize.ResizableContainerInfo,
				expectedCpuInfoFirstPatch []containerCPUInfo,
				wantPatchErrorFirstPatch string,
				wantActuationErrorFirstPatch string,
				desiredContainersSecondPatch []podresize.ResizableContainerInfo,
				expectedContainersSecondPatch []podresize.ResizableContainerInfo,
				expectedCpuInfoSecondPatch []containerCPUInfo,
				wantPatchErrorSecondPatch string,
				wantActuationErrorSecondPatch string,
			) {

				baselineExpectedCPUCount := 0
				for i := range baselineCpuInfo {
					baselineExpectedCPUCount += baselineCpuInfo[i].cpuCount
				}
				firstPatchExpectedCPUCount := 0
				for i := range expectedCpuInfoFirstPatch {
					firstPatchExpectedCPUCount += expectedCpuInfoFirstPatch[i].cpuCount
				}
				secondPatchExpectedCPUCount := 0
				for i := range expectedCpuInfoSecondPatch {
					secondPatchExpectedCPUCount += expectedCpuInfoSecondPatch[i].cpuCount
				}
				skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), max(baselineExpectedCPUCount, firstPatchExpectedCPUCount, secondPatchExpectedCPUCount))

				updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
					policyName:         string(cpumanager.PolicyStatic),
					reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
					enableInPlacePodVerticalScalingExclusiveCPUs: true,
					topologyManagerPolicy:                        "none",
				}))
				testPodResize(ctx,
					baselineContainers,
					baselineCpuInfo,
					desiredContainersFirstPatch,
					expectedContainersFirstPatch,
					expectedCpuInfoFirstPatch,
					wantPatchErrorFirstPatch,
					wantActuationErrorFirstPatch,
					desiredContainersSecondPatch,
					expectedContainersSecondPatch,
					expectedCpuInfoSecondPatch,
					wantPatchErrorSecondPatch,
					wantActuationErrorSecondPatch,
					podMap)
			},
			ginkgo.Entry("should succeed to scale up both a (gu-container-1) and a non-restartable init container using shared pool",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "init-container-1",
						Resources:     &cgroups.ContainerResources{CPUReq: "2500m", CPULim: "2500m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyNever,
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "init-container-1",
						Resources:     &cgroups.ContainerResources{CPUReq: "3500m", CPULim: "3500m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyNever,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "init-container-1",
						Resources:     &cgroups.ContainerResources{CPUReq: "3500m", CPULim: "3500m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyNever,
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 4,
					},
				},
				"",
				"",
				nil,
				nil,
				nil,
				"",
				"",
			),
			ginkgo.Entry("should succeed to scale up (gu-container-1) without resizing a non-restartable init container using shared pool",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "init-container-1",
						Resources:     &cgroups.ContainerResources{CPUReq: "2500m", CPULim: "2500m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyNever,
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "init-container-1",
						Resources:     &cgroups.ContainerResources{CPUReq: "2500m", CPULim: "2500m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyNever,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "init-container-1",
						Resources:     &cgroups.ContainerResources{CPUReq: "2500m", CPULim: "2500m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyNever,
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 4,
					},
				},
				"",
				"",
				nil,
				nil,
				nil,
				"",
				"",
			),
			ginkgo.Entry("should succeed to scale up (gu-container-1) without resizing a non-restartable init container with exclusive CPUs",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "init-container-1-with-exclusive-cpus",
						Resources:     &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyNever,
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
					{
						Name:     "init-container-1-with-exclusive-cpus",
						InitCtr:  true,
						cpuCount: 2,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "init-container-1-with-exclusive-cpus",
						Resources:     &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyNever,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "init-container-1-with-exclusive-cpus",
						Resources:     &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyNever,
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 4,
					},
					{
						Name:     "init-container-1-with-exclusive-cpus",
						InitCtr:  true,
						cpuCount: 2,
					},
				},
				"",
				"",
				nil,
				nil,
				nil,
				"",
				"",
			),
			ginkgo.Entry("should fail to resize of non restartable init container with exclusive CPUs",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "init-container-1-with-exclusive-cpus",
						Resources:     &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyNever,
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
					{
						Name:     "init-container-1-with-exclusive-cpus",
						InitCtr:  true,
						cpuCount: 2,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "init-container-1-with-exclusive-cpus",
						Resources:     &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyNever,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "init-container-1-with-exclusive-cpus",
						Resources:     &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyNever,
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
					{
						Name:     "init-container-1-with-exclusive-cpus",
						InitCtr:  true,
						cpuCount: 2,
					},
				},
				"",
				"Infeasible.*",
				nil,
				nil,
				nil,
				"",
				"",
			),
			ginkgo.Entry("should first increase CPU (gu-container-1) request and limit, afterwards decrease CPU (gu-container-1) request and limit within available capacity without impacting sidecar CPUs",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "sidecar-container-1-with-exclusive-cpus",
						Resources:     &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyAlways,
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
					{
						Name:       "sidecar-container-1-with-exclusive-cpus",
						cpuCount:   2,
						SidecarCtr: true,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "sidecar-container-1-with-exclusive-cpus",
						Resources:     &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyAlways,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:      "sidecar-container-1-with-exclusive-cpus",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 4,
					},
					{
						Name:       "sidecar-container-1-with-exclusive-cpus",
						cpuCount:   2,
						SidecarCtr: true,
					},
				},
				"",
				"",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:      "sidecar-container-1-with-exclusive-cpus",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:      "sidecar-container-1-with-exclusive-cpus",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
					{
						Name:       "sidecar-container-1-with-exclusive-cpus",
						cpuCount:   2,
						SidecarCtr: true,
					},
				},
				"",
				"",
			),
			ginkgo.Entry("should first increase CPU (sidecar-container-1) request and limit, afterwards decrease CPU (sidecar-container-1) and increase CPU (gu-container-1) request and limit within available capacity",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "sidecar-container-1",
						Resources:     &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyAlways,
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
					{
						Name:       "sidecar-container-1",
						SidecarCtr: true,
						cpuCount:   2,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "sidecar-container-1",
						Resources:     &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyAlways,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:      "sidecar-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
					{
						Name:       "sidecar-container-1",
						SidecarCtr: true,
						cpuCount:   4,
					},
				},
				"",
				"",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:          "sidecar-container-1",
						Resources:     &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						InitCtr:       true,
						RestartPolicy: v1.ContainerRestartPolicyAlways,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:      "sidecar-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 4,
					},
					{
						Name:       "sidecar-container-1",
						SidecarCtr: true,
						cpuCount:   2,
					},
				},
				"",
				"",
			),
		)
	})

	ginkgo.When("topologyManagerPolicy is set to best-effort, resizing a Guaranteed multiple containers Pod with integer CPU request", ginkgo.Label("guaranteed pod with integer CPU requests resize", "exclusive-cpus"), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			reservedCPUs = cpuset.New(0)
		})
		ginkgo.DescribeTable("",
			func(ctx context.Context,
				baselineContainers []podresize.ResizableContainerInfo,
				baselineCpuInfo []containerCPUInfo,
				desiredContainersFirstPatch []podresize.ResizableContainerInfo,
				expectedContainersFirstPatch []podresize.ResizableContainerInfo,
				expectedCpuInfoFirstPatch []containerCPUInfo,
				wantPatchErrorFirstPatch string,
				wantActuationErrorFirstPatch string,
				desiredContainersSecondPatch []podresize.ResizableContainerInfo,
				expectedContainersSecondPatch []podresize.ResizableContainerInfo,
				expectedCpuInfoSecondPatch []containerCPUInfo,
				wantPatchErrorSecondPatch string,
				wantActuationErrorSecondPatch string,
			) {

				baselineExpectedCPUCount := 0
				for i := range baselineCpuInfo {
					baselineExpectedCPUCount += baselineCpuInfo[i].cpuCount
				}
				firstPatchExpectedCPUCount := 0
				for i := range expectedCpuInfoFirstPatch {
					firstPatchExpectedCPUCount += expectedCpuInfoFirstPatch[i].cpuCount
				}
				secondPatchExpectedCPUCount := 0
				for i := range expectedCpuInfoSecondPatch {
					secondPatchExpectedCPUCount += expectedCpuInfoSecondPatch[i].cpuCount
				}
				skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), max(baselineExpectedCPUCount, firstPatchExpectedCPUCount, secondPatchExpectedCPUCount))

				updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
					policyName:         string(cpumanager.PolicyStatic),
					reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
					enableInPlacePodVerticalScalingExclusiveCPUs: true,
					topologyManagerPolicy:                        "best-effort",
					topologyManagerScope:                         "container",
				}))
				testPodResize(ctx,
					baselineContainers,
					baselineCpuInfo,
					desiredContainersFirstPatch,
					expectedContainersFirstPatch,
					expectedCpuInfoFirstPatch,
					wantPatchErrorFirstPatch,
					wantActuationErrorFirstPatch,
					desiredContainersSecondPatch,
					expectedContainersSecondPatch,
					expectedCpuInfoSecondPatch,
					wantPatchErrorSecondPatch,
					wantActuationErrorSecondPatch,
					podMap)

			},
			ginkgo.Entry("should first increase (gu-container-1) and (gu-container-2) CPU request/limit, afterwards decrease (gu-container-1) and (gu-container-2) CPU request/limit, within available capacity",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:      "gu-container-2",
						Resources: &cgroups.ContainerResources{CPUReq: "3000m", CPULim: "3000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
					{
						Name:     "gu-container-2",
						cpuCount: 3,
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:      "gu-container-2",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:      "gu-container-2",
						Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 4,
					},
					{
						Name:     "gu-container-2",
						cpuCount: 4,
					},
				},
				"",
				"",
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:      "gu-container-2",
						Resources: &cgroups.ContainerResources{CPUReq: "3000m", CPULim: "3000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]podresize.ResizableContainerInfo{
					{
						Name:      "gu-container-1",
						Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
					{
						Name:      "gu-container-2",
						Resources: &cgroups.ContainerResources{CPUReq: "3000m", CPULim: "3000m", MemReq: "200Mi", MemLim: "200Mi"},
					},
				},
				[]containerCPUInfo{
					{
						Name:     "gu-container-1",
						cpuCount: 2,
					},
					{
						Name:     "gu-container-2",
						cpuCount: 3,
					},
				},
				"",
				"",
			),
		)
	})

	ginkgo.When("topologyManagerPolicy option is set to restricted, resizing a Guaranteed multiple containers Pod, with integer CPU request", ginkgo.Label("guaranteed multiple containers pod with integer CPU requests resize", "exclusive-cpus"), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			reservedCPUs = cpuset.New(0)
		})
		smtLevel = smtLevelFromSysFS() // this should not change at all, at least during this suite lifetime
		framework.Logf("SMT level: %d", smtLevel)

		if smtLevel >= minSMTLevel {
			ginkgo.DescribeTable("",
				func(ctx context.Context,
					baselineContainers []podresize.ResizableContainerInfo,
					baselineCpuInfo []containerCPUInfo,
					desiredContainersFirstPatch []podresize.ResizableContainerInfo,
					expectedContainersFirstPatch []podresize.ResizableContainerInfo,
					expectedCpuInfoFirstPatch []containerCPUInfo,
					wantPatchErrorFirstPatch string,
					wantActuationErrorFirstPatch string,
					desiredContainersSecondPatch []podresize.ResizableContainerInfo,
					expectedContainersSecondPatch []podresize.ResizableContainerInfo,
					expectedCpuInfoSecondPatch []containerCPUInfo,
					wantPatchErrorSecondPatch string,
					wantActuationErrorSecondPatch string,
				) {

					baselineExpectedCPUCount := 0
					for i := range baselineCpuInfo {
						baselineExpectedCPUCount += baselineCpuInfo[i].cpuCount
					}
					firstPatchExpectedCPUCount := 0
					for i := range expectedCpuInfoFirstPatch {
						firstPatchExpectedCPUCount += expectedCpuInfoFirstPatch[i].cpuCount
					}
					secondPatchExpectedCPUCount := 0
					for i := range expectedCpuInfoSecondPatch {
						secondPatchExpectedCPUCount += expectedCpuInfoSecondPatch[i].cpuCount
					}
					skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), max(baselineExpectedCPUCount, firstPatchExpectedCPUCount, secondPatchExpectedCPUCount))

					updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
						policyName:         string(cpumanager.PolicyStatic),
						reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
						enableInPlacePodVerticalScalingExclusiveCPUs: true,
						topologyManagerPolicy:                        "restricted",
						topologyManagerScope:                         "container",
						topologyManagerPolicyOptions: map[string]string{
							"max-allowable-numa-nodes":  "8",
							"prefer-closest-numa-nodes": "true",
						},
					}))
					testPodResize(ctx,
						baselineContainers,
						baselineCpuInfo,
						desiredContainersFirstPatch,
						expectedContainersFirstPatch,
						expectedCpuInfoFirstPatch,
						wantPatchErrorFirstPatch,
						wantActuationErrorFirstPatch,
						desiredContainersSecondPatch,
						expectedContainersSecondPatch,
						expectedCpuInfoSecondPatch,
						wantPatchErrorSecondPatch,
						wantActuationErrorSecondPatch,
						podMap)
				},
				ginkgo.Entry("should first increase (gu-container-1) CPU request/limit, afterwards decrease (gu-container-1) CPU request/limit, within available capacity",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 1,
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 2,
						},
					},
					"",
					"",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 1,
						},
					},
					"",
					"",
				),
			)
		}
	})

	ginkgo.When("topology manager policy option is set to single-numa-node, resizing a Guaranteed multiple container pod, with integer CPU request", ginkgo.Label("guaranteed multiple containers pod with integer CPU requests resize", "exclusive-cpus"), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			reservedCPUs = cpuset.New(0)
		})
		smtLevel = smtLevelFromSysFS() // this should not change at all, at least during this suite lifetime
		framework.Logf("SMT level: %d", smtLevel)

		if smtLevel >= minSMTLevel {
			ginkgo.DescribeTable("",
				func(ctx context.Context,
					baselineContainers []podresize.ResizableContainerInfo,
					baselineCpuInfo []containerCPUInfo,
					desiredContainersFirstPatch []podresize.ResizableContainerInfo,
					expectedContainersFirstPatch []podresize.ResizableContainerInfo,
					expectedCpuInfoFirstPatch []containerCPUInfo,
					wantPatchErrorFirstPatch string,
					wantActuationErrorFirstPatch string,
					desiredContainersSecondPatch []podresize.ResizableContainerInfo,
					expectedContainersSecondPatch []podresize.ResizableContainerInfo,
					expectedCpuInfoSecondPatch []containerCPUInfo,
					wantPatchErrorSecondPatch string,
					wantActuationErrorSecondPatch string,
				) {

					baselineExpectedCPUCount := 0
					for i := range baselineCpuInfo {
						baselineExpectedCPUCount += baselineCpuInfo[i].cpuCount
					}
					firstPatchExpectedCPUCount := 0
					for i := range expectedCpuInfoFirstPatch {
						firstPatchExpectedCPUCount += expectedCpuInfoFirstPatch[i].cpuCount
					}
					secondPatchExpectedCPUCount := 0
					for i := range expectedCpuInfoSecondPatch {
						secondPatchExpectedCPUCount += expectedCpuInfoSecondPatch[i].cpuCount
					}
					skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), max(baselineExpectedCPUCount, firstPatchExpectedCPUCount, secondPatchExpectedCPUCount))

					updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
						policyName:         string(cpumanager.PolicyStatic),
						reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
						enableInPlacePodVerticalScalingExclusiveCPUs: true,
						topologyManagerPolicy:                        "single-numa-node",
						topologyManagerScope:                         "container",
						topologyManagerPolicyOptions: map[string]string{
							"max-allowable-numa-nodes":  "8",
							"prefer-closest-numa-nodes": "true",
						},
					}))
					testPodResize(ctx,
						baselineContainers,
						baselineCpuInfo,
						desiredContainersFirstPatch,
						expectedContainersFirstPatch,
						expectedCpuInfoFirstPatch,
						wantPatchErrorFirstPatch,
						wantActuationErrorFirstPatch,
						desiredContainersSecondPatch,
						expectedContainersSecondPatch,
						expectedCpuInfoSecondPatch,
						wantPatchErrorSecondPatch,
						wantActuationErrorSecondPatch,
						podMap)
				},
				ginkgo.Entry("should first increase (gu-container-1) CPU request/limit and afterwards decrease (gu-container-1) CPU request/limit, within available capacity",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 2,
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 4,
						},
					},
					"",
					"",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 2,
						},
					},
					"",
					"",
				),
				ginkgo.Entry("should first increase (gu-container-1) CPU request/limit, then fail to reduce (gu-container-1) CPU request/limit, below baseline and increase (gu-container-2) CPU request/limit, within available capacity",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "3000m", CPULim: "3000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 3,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 2,
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 4,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 2,
						},
					},
					"",
					"",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 4,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 2,
						},
					},
					"",
					"Infeasible.*not allowed to reduce container exclusively allocated pool below baseline.*",
				),
				ginkgo.Entry("should first increase (gu-container-1) CPU request/limit, afterwards keep (gu-container-1) CPU request/limit and increase (gu-container-2) CPU request/limit, within available capacity",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 2,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 2,
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 4,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 2,
						},
					},
					"",
					"",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 4,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 4,
						},
					},
					"",
					"",
				),
				ginkgo.Entry("should first increase (gu-container-1) CPU request/limit, afterwards fail to reduce (gu-container-1) CPU request/limit to baseline and increase (gu-container-2) CPU request/limit, beyond available capacity",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "3000m", CPULim: "3000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 3,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 2,
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 4,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 2,
						},
					},
					"",
					"",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "3000m", CPULim: "3000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "1000000m", CPULim: "1000000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 4,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 2,
						},
					},
					".*node didn't have enough allocatable resources.*",
					"",
				),
				ginkgo.Entry("should first increase (gu-container-1) CPU request/limit and scale down (gu-container-1) CPU request/limit, within available capacity",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 1,
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 2,
						},
					},
					"",
					"",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 1,
						},
					},
					"",
					"",
				),
				ginkgo.Entry("should first increase (gu-container-1) CPU request/limit afterwards scale down (gu-container-1) CPU request/limit and increase (gu-container-2) CPU request/limit, within available capacity",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 1,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 1,
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 2,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 1,
						},
					},
					"",
					"",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "1000m", CPULim: "1000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 1,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 2,
						},
					},
					"",
					"",
				),
				ginkgo.Entry("should first increase (gu-container-1) CPU request/limit, afterwards fail to reduce (gu-container-1) CPU request/limit, below baseline and increase (gu-container-2) CPU request/limit, within available capacity",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "3000m", CPULim: "3000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 3,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 2,
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 4,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 2,
						},
					},
					"",
					"",
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]podresize.ResizableContainerInfo{
						{
							Name:      "gu-container-1",
							Resources: &cgroups.ContainerResources{CPUReq: "4000m", CPULim: "4000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
						{
							Name:      "gu-container-2",
							Resources: &cgroups.ContainerResources{CPUReq: "2000m", CPULim: "2000m", MemReq: "200Mi", MemLim: "200Mi"},
						},
					},
					[]containerCPUInfo{
						{
							Name:     "gu-container-1",
							cpuCount: 4,
						},
						{
							Name:     "gu-container-2",
							cpuCount: 2,
						},
					},
					"",
					"Infeasible.*not allowed to reduce container exclusively allocated pool below baseline.*",
				),
			)
		}
	})

})

func WaitForPodResizeInfeasible(ctx context.Context, f *framework.Framework, testPod *v1.Pod) {
	framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, testPod.Namespace, testPod.Name, "display pod resize status as infeasible", f.Timeouts.PodStart, func(pod *v1.Pod) (bool, error) {
		return helpers.IsPodResizeInfeasible(pod), nil
	}))
}

func HaveStatusConditionsMatchingRegex(expr string) types.GomegaMatcher {
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		re, err := regexp.Compile(expr)
		if err != nil {
			return false, err
		}
		for _, condition := range actual.Status.Conditions {
			if re.MatchString(fmt.Sprintf("%v", condition)) {
				return true, nil
			}
		}
		return false, nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} conditions {{.Actual.Status.Conditions}} does not match regexp {{.Data}}", expr)
}

func HaveSidecarContainerCPUsCount(ctnName string, val int) types.GomegaMatcher {
	md := &msgData{
		Name:  ctnName,
		Count: val,
	}
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		cpus, err := getContainerAllowedCPUs(actual, ctnName, true)
		md.CurrentCPUs = cpus.String()
		if err != nil {
			framework.Logf("getContainerAllowedCPUs(%s) failed: %v", ctnName, err)
			return false, err
		}
		return cpus.Size() == val, nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} has allowed CPUs <{{.Data.CurrentCPUs}}> not matching expected count <{{.Data.Count}}> for container {{.Data.Name}}", md)
}

func HaveInitContainerCPUsCount(ctnName string, val int, logs string) types.GomegaMatcher {
	md := &msgData{
		Name:  ctnName,
		Count: val,
	}
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		cpus := getInitContainerAllowedCPUsFromLogs(actual.Namespace, actual.Name, ctnName, logs)
		md.CurrentCPUs = cpus.String()
		return cpus.Size() == val, nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} has allowed CPUs <{{.Data.CurrentCPUs}}> not matching expected count <{{.Data.Count}}> for container {{.Data.Name}}", md)
}

func getInitContainerAllowedCPUsFromLogs(podNamespace, podName, cntName, logs string) cpuset.CPUSet {
	framework.Logf("got pod logs: <%v>", logs)
	r := strings.NewReplacer("Cpus_allowed_list:", "")
	cpus, err := cpuset.Parse(strings.TrimSpace(r.Replace(logs)))
	framework.Logf("pod %v/%v cnt %v cpuset %v", podNamespace, podName, cntName, cpus)
	framework.ExpectNoError(err, "parsing cpuset from logs for [%s] of pod [%s]", cntName, podName)
	return cpus
}

func checkAllocatedCPUsForResize(ctx context.Context, f *framework.Framework, expectedCpusInfo []containerCPUInfo, testPod *v1.Pod) {
	// CPUManagerReconcilePeriod is set to 1 second during tests,
	// to reduce flakiness this check includes a 5 second timeout.
	const cpusetReconcilePeriodTimeout = 5 * time.Second

	for i := range expectedCpusInfo {
		if expectedCpusInfo[i].InitCtr {
			// when we get there the real initcontainer terminated, so we can only check its logs
			ginkgo.By("checking if the expected cpuset was assigned")
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, testPod.Namespace, testPod.Name, expectedCpusInfo[i].Name)
			framework.ExpectNoError(err, "expected log not found in init container [%s] of pod [%s]", expectedCpusInfo[i].Name, testPod.Name)
			gomega.Expect(testPod).To(HaveInitContainerCPUsCount(expectedCpusInfo[i].Name, expectedCpusInfo[i].cpuCount, logs))
		} else if expectedCpusInfo[i].SidecarCtr {
			gomega.Eventually(ctx, testPod).
				WithTimeout(cpusetReconcilePeriodTimeout).
				WithPolling(f.Timeouts.Poll).
				Should(HaveSidecarContainerCPUsCount(expectedCpusInfo[i].Name, expectedCpusInfo[i].cpuCount))
			gomega.Consistently(ctx, testPod).
				WithTimeout(5 * time.Second).
				WithPolling(f.Timeouts.Poll).
				Should(HaveSidecarContainerCPUsCount(expectedCpusInfo[i].Name, expectedCpusInfo[i].cpuCount))
		} else {
			gomega.Eventually(ctx, testPod).
				WithTimeout(cpusetReconcilePeriodTimeout).
				WithPolling(f.Timeouts.Poll).
				Should(HaveContainerCPUsCount(expectedCpusInfo[i].Name, expectedCpusInfo[i].cpuCount))
			gomega.Consistently(ctx, testPod).
				WithTimeout(5 * time.Second).
				WithPolling(f.Timeouts.Poll).
				Should(HaveContainerCPUsCount(expectedCpusInfo[i].Name, expectedCpusInfo[i].cpuCount))
		}
	}
}
