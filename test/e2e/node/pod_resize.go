/*
Copyright 2021 The Kubernetes Authors.

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
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	helpers "k8s.io/component-helpers/resource"
	resourceapi "k8s.io/kubernetes/pkg/api/v1/resource"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/common/node/framework/cgroups"
	"k8s.io/kubernetes/test/e2e/common/node/framework/podresize"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	gomegatypes "github.com/onsi/gomega/types"
)

func doPodResizeResourceQuotaTests(f *framework.Framework) {
	originalContainers := []podresize.ResizableContainerInfo{
		{
			Name:      "c1",
			Resources: &cgroups.ContainerResources{CPUReq: "300m", CPULim: "300m", MemReq: "300Mi", MemLim: "300Mi"},
		},
	}

	ginkgo.BeforeEach(func(ctx context.Context) {
		resourceQuota := v1.ResourceQuota{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "resize-resource-quota",
				Namespace: f.Namespace.Name,
			},
			Spec: v1.ResourceQuotaSpec{
				Hard: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("800m"),
					v1.ResourceMemory: resource.MustParse("800Mi"),
				},
			},
		}

		ginkgo.By("Creating a ResourceQuota")
		_, rqErr := f.ClientSet.CoreV1().ResourceQuotas(f.Namespace.Name).Create(ctx, &resourceQuota, metav1.CreateOptions{})
		framework.ExpectNoError(rqErr, "failed to create resource quota")
		// pod creation using this quota will fail until the quota status is populated, so we need to wait to
		// prevent races with the resourcequota controller
		ginkgo.By("Waiting for ResourceQuota status to populate")
		quotaStatusErr := waitForResourceQuota(ctx, f.ClientSet, f.Namespace.Name, resourceQuota.Name)
		framework.ExpectNoError(quotaStatusErr, "resource quota status failed to populate")
	})

	ginkgo.DescribeTable("pod-resize-resource-quota-test",
		func(ctx context.Context, desiredContainers []podresize.ResizableContainerInfo, expectedContainers []podresize.ResizableContainerInfo, wantError string) {
			tStamp := strconv.Itoa(time.Now().Nanosecond())
			testPod1 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod1", tStamp, originalContainers, nil)
			testPod1 = e2epod.MustMixinRestrictedPodSecurity(testPod1)
			testPod2 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod2", tStamp, originalContainers, nil)
			testPod2 = e2epod.MustMixinRestrictedPodSecurity(testPod2)

			ginkgo.By("creating pods")
			podClient := e2epod.NewPodClient(f)
			newPods := podClient.CreateBatch(ctx, []*v1.Pod{testPod1, testPod2})

			ginkgo.By("verifying initial pod resources, and policy are as expected")
			podresize.VerifyPodResources(newPods[0], originalContainers, nil)

			ginkgo.By("patching pod for resize with resource-quota")
			patchString := podresize.MakeResizePatch(originalContainers, desiredContainers, nil, nil)

			if wantError == "" {
				patchedPod, pErr := f.ClientSet.CoreV1().Pods(newPods[0].Namespace).Patch(ctx,
					newPods[0].Name, types.StrategicMergePatchType, []byte(patchString), metav1.PatchOptions{}, "resize")
				framework.ExpectNoError(pErr, "failed to patch pod for resize")

				expected := podresize.UpdateExpectedContainerRestarts(ctx, patchedPod, expectedContainers)
				ginkgo.By("verifying pod resources are as expected post patch, pre-actuation")
				podresize.VerifyPodResources(patchedPod, expected, nil)

				ginkgo.By("waiting for resize to be actuated")
				resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, newPods[0], expected)
				podresize.ExpectPodResized(ctx, f, resizedPod, expected)

				ginkgo.By("verifying pod resources after resize")
				podresize.VerifyPodResources(resizedPod, expected, nil)

			} else {
				var patchedPod *v1.Pod
				framework.ExpectNoError(framework.Gomega().
					// Use Eventually because we need to wait for the resource-quota controller to sync.
					Eventually(ctx, func(ctx context.Context) error {
						var pErr error
						patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPods[0].Namespace).Patch(ctx,
							newPods[0].Name, types.StrategicMergePatchType, []byte(patchString), metav1.PatchOptions{}, "resize")
						return pErr
					}).
					WithTimeout(f.Timeouts.PodStart).
					Should(gomega.MatchError(gomega.ContainSubstring(wantError))))

				expected := podresize.UpdateExpectedContainerRestarts(ctx, patchedPod, expectedContainers)
				ginkgo.By("verifying pod patched for resize with error remains unchanged")
				patchedPod, pErrEx2 := podClient.Get(ctx, newPods[0].Name, metav1.GetOptions{})
				framework.ExpectNoError(pErrEx2, "failed to get pod post failed resize")
				podresize.VerifyPodResources(patchedPod, expected, nil)
				framework.ExpectNoError(podresize.VerifyPodStatusResources(patchedPod, expected))
			}
		},

		ginkgo.Entry("exceed maximum CPU",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "600m", CPULim: "600m", MemReq: "400Mi", MemLim: "400Mi"},
				},
			},
			originalContainers,
			"exceeded quota: resize-resource-quota, requested: cpu=300m, used: cpu=600m, limited: cpu=800m",
		),

		ginkgo.Entry("exceed maximum Memory",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "250m", CPULim: "250m", MemReq: "750Mi", MemLim: "750Mi"},
				},
			},
			originalContainers,
			"exceeded quota: resize-resource-quota, requested: memory=450Mi, used: memory=600Mi, limited: memory=800Mi",
		),

		ginkgo.Entry("exceed maximum CPU and Memory",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "600m", CPULim: "600m", MemReq: "750Mi", MemLim: "750Mi"},
				},
			},
			originalContainers,
			"exceeded quota: resize-resource-quota, requested: cpu=300m,memory=450Mi, used: cpu=600m,memory=600Mi, limited: cpu=800m,memory=800Mi",
		),

		ginkgo.Entry("valid increase of CPU",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "350m", CPULim: "350m", MemReq: "300Mi", MemLim: "300Mi"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "350m", CPULim: "350m", MemReq: "300Mi", MemLim: "300Mi"},
				},
			},
			"",
		),

		ginkgo.Entry("valid increase of Memory",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "350m", CPULim: "350m", MemReq: "350Mi", MemLim: "350Mi"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "350m", CPULim: "350m", MemReq: "350Mi", MemLim: "350Mi"},
				},
			},
			"",
		),

		ginkgo.Entry("valid increase for both CPU and Memory",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "400m", CPULim: "400m", MemReq: "400Mi", MemLim: "400Mi"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "400m", CPULim: "400m", MemReq: "400Mi", MemLim: "400Mi"},
				},
			},
			"",
		),
	)
}

func doPodResizeLimitRangerTests(f *framework.Framework) {
	originalContainers := []podresize.ResizableContainerInfo{
		{
			Name:      "c1",
			Resources: &cgroups.ContainerResources{CPUReq: "300m", CPULim: "300m", MemReq: "300Mi", MemLim: "300Mi"},
		},
	}

	ginkgo.BeforeEach(func(ctx context.Context) {
		ginkgo.By("Creating a LimitRanger")
		lr := &v1.LimitRange{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "resize-limit-ranger",
				Namespace: f.Namespace.Name,
			},
			Spec: v1.LimitRangeSpec{
				Limits: []v1.LimitRangeItem{
					{
						Type: v1.LimitTypeContainer,
						Max: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("500m"),
							v1.ResourceMemory: resource.MustParse("500Mi"),
						},
						Min: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("50m"),
							v1.ResourceMemory: resource.MustParse("50Mi"),
						},
						Default: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("100m"),
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
						DefaultRequest: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("50m"),
							v1.ResourceMemory: resource.MustParse("50Mi"),
						},
					},
				},
			},
		}
		_, lrErr := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Create(context.Background(), lr, metav1.CreateOptions{})
		framework.ExpectNoError(lrErr, "failed to create limit ranger")

		ginkgo.By("Fetching the LimitRange to ensure it has proper values")
		gotLr, lrErr := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Get(ctx, lr.Name, metav1.GetOptions{})
		framework.ExpectNoError(lrErr)

		if !apiequality.Semantic.DeepEqual(lr.Spec.Limits, gotLr.Spec.Limits) {
			framework.Failf("retrieved LimitRange does not match created one: got %v; expected %v", gotLr.Spec.Limits, lr.Spec.Limits)
		}
	})

	ginkgo.DescribeTable("pod-resize-limit-ranger-test",
		func(ctx context.Context, desiredContainers []podresize.ResizableContainerInfo, expectedContainers []podresize.ResizableContainerInfo, wantErrors []string) {
			tStamp := strconv.Itoa(time.Now().Nanosecond())
			testPod1 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod1", tStamp, originalContainers, nil)
			testPod1 = e2epod.MustMixinRestrictedPodSecurity(testPod1)
			testPod2 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod2", tStamp, originalContainers, nil)
			testPod2 = e2epod.MustMixinRestrictedPodSecurity(testPod2)

			ginkgo.By("creating pods")
			podClient := e2epod.NewPodClient(f)
			newPods := podClient.CreateBatch(ctx, []*v1.Pod{testPod1, testPod2})

			ginkgo.By("verifying initial pod resources, and policy are as expected")
			podresize.VerifyPodResources(newPods[0], originalContainers, nil)

			ginkgo.By("patching pod for resize with limit-ranger")
			patchString := podresize.MakeResizePatch(originalContainers, desiredContainers, nil, nil)

			if len(wantErrors) == 0 {
				patchedPod, pErr := f.ClientSet.CoreV1().Pods(newPods[0].Namespace).Patch(ctx,
					newPods[0].Name, types.StrategicMergePatchType, []byte(patchString), metav1.PatchOptions{}, "resize")
				framework.ExpectNoError(pErr, "failed to patch pod for resize")

				expected := podresize.UpdateExpectedContainerRestarts(ctx, patchedPod, expectedContainers)
				ginkgo.By("verifying pod resources are as expected post patch, pre-actuation")
				podresize.VerifyPodResources(patchedPod, expected, nil)

				ginkgo.By("waiting for resize to be actuated")
				resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, newPods[0], expected)
				podresize.ExpectPodResized(ctx, f, resizedPod, expected)

				ginkgo.By("verifying pod resources after resize")
				podresize.VerifyPodResources(resizedPod, expected, nil)

			} else {
				var patchedPod *v1.Pod
				var errMatchers []gomegatypes.GomegaMatcher
				for _, wantErr := range wantErrors {
					errMatchers = append(errMatchers, gomega.ContainSubstring(wantErr))
				}

				framework.ExpectNoError(framework.Gomega().
					// Use Eventually because we need to wait for the limit-ranger controller to sync.
					Eventually(ctx, func(ctx context.Context) error {
						var pErr error
						patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPods[0].Namespace).Patch(ctx,
							newPods[0].Name, types.StrategicMergePatchType, []byte(patchString), metav1.PatchOptions{}, "resize")
						return pErr
					}).
					WithTimeout(f.Timeouts.PodStart).
					Should(gomega.MatchError(gomega.And(errMatchers...))))

				expected := podresize.UpdateExpectedContainerRestarts(ctx, patchedPod, expectedContainers)
				ginkgo.By("verifying pod patched for resize with error remains unchanged")
				patchedPod, pErrEx2 := podClient.Get(ctx, newPods[0].Name, metav1.GetOptions{})
				framework.ExpectNoError(pErrEx2, "failed to get pod post failed resize")
				podresize.VerifyPodResources(patchedPod, expected, nil)
				framework.ExpectNoError(podresize.VerifyPodStatusResources(patchedPod, expected))
			}
		},

		ginkgo.Entry("exceed maximum CPU",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "600m", CPULim: "600m", MemReq: "250Mi", MemLim: "250Mi"},
				},
			},
			originalContainers,
			[]string{"forbidden: maximum cpu usage per Container is 500m, but limit is 600m"},
		),

		ginkgo.Entry("exceed maximum Memory",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "250m", CPULim: "250m", MemReq: "750Mi", MemLim: "750Mi"},
				},
			},
			originalContainers,
			[]string{"forbidden: maximum memory usage per Container is 500Mi, but limit is 750Mi"},
		),

		ginkgo.Entry("exceed maximum Memory and CPU",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "600m", CPULim: "600m", MemReq: "600Mi", MemLim: "600Mi"},
				},
			},
			originalContainers,
			[]string{"maximum cpu usage per Container is 500m, but limit is 600m", "maximum memory usage per Container is 500Mi, but limit is 600Mi"},
		),

		ginkgo.Entry("request below min CPU",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "10m", CPULim: "10m", MemReq: "400Mi", MemLim: "400Mi"},
				},
			},
			originalContainers,
			[]string{"forbidden: minimum cpu usage per Container is 50m, but request is 10m"},
		),

		ginkgo.Entry("request below min Memory",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "250m", CPULim: "250m", MemReq: "10Mi", MemLim: "10Mi"},
				},
			},
			originalContainers,
			[]string{"forbidden: minimum memory usage per Container is 50Mi, but request is 10Mi"},
		),

		ginkgo.Entry("request below min CPU and min Memory",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "10m", CPULim: "10m", MemReq: "10Mi", MemLim: "10Mi"},
				},
			},
			originalContainers,
			[]string{"minimum cpu usage per Container is 50m, but request is 10m", "minimum memory usage per Container is 50Mi, but request is 10Mi"},
		),

		ginkgo.Entry("valid increase of CPU",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "350m", CPULim: "350m", MemReq: "300Mi", MemLim: "300Mi"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "350m", CPULim: "350m", MemReq: "300Mi", MemLim: "300Mi"},
				},
			},
			nil,
		),

		ginkgo.Entry("valid increase of Memory",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "350m", CPULim: "350m", MemReq: "350Mi", MemLim: "350Mi"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "350m", CPULim: "350m", MemReq: "350Mi", MemLim: "350Mi"},
				},
			},
			nil,
		),

		ginkgo.Entry("valid increase of CPU and Memory",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "400m", CPULim: "400m", MemReq: "400Mi", MemLim: "400Mi"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "400m", CPULim: "400m", MemReq: "400Mi", MemLim: "400Mi"},
				},
			},
			nil,
		),

		ginkgo.Entry("valid decrease of Memory",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "400m", CPULim: "400m", MemReq: "350Mi", MemLim: "350Mi"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "400m", CPULim: "400m", MemReq: "350Mi", MemLim: "350Mi"},
				},
			},
			nil,
		),

		ginkgo.Entry("valid decrease of CPU",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "350m", CPULim: "350m", MemReq: "350Mi", MemLim: "350Mi"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "350m", CPULim: "350m", MemReq: "350Mi", MemLim: "350Mi"},
				},
			},
			nil,
		),

		ginkgo.Entry("valid decrease of CPU and Memory",
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "250m", CPULim: "250m", MemReq: "250Mi", MemLim: "250Mi"},
				},
			},
			[]podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &cgroups.ContainerResources{CPUReq: "250m", CPULim: "250m", MemReq: "250Mi", MemLim: "250Mi"},
				},
			},
			nil,
		),
	)
}

func doPodResizeSchedulerTests(f *framework.Framework) {
	ginkgo.It("pod-resize-scheduler-tests", func(ctx context.Context) {
		podClient := e2epod.NewPodClient(f)
		nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err, "failed to get running nodes")
		gomega.Expect(nodes.Items).ShouldNot(gomega.BeEmpty())
		framework.Logf("Found %d schedulable nodes", len(nodes.Items))

		ginkgo.By("Find node CPU resources available for allocation!")
		node := nodes.Items[0]
		nodeAllocatableMilliCPU, nodeAvailableMilliCPU := getNodeAllocatableAndAvailableValues(ctx, f, &node, v1.ResourceCPU)
		framework.Logf("Node '%s': NodeAllocatable MilliCPUs = %dm. MilliCPUs currently available to allocate = %dm.",
			node.Name, nodeAllocatableMilliCPU, nodeAvailableMilliCPU)

		//
		// Scheduler focused pod resize E2E test case #1:
		//     1. Create pod1 and pod2 on node such that pod1 has enough CPU to be scheduled, but pod2 does not.
		//     2. Resize pod2 down so that it fits on the node and can be scheduled.
		//     3. Verify that pod2 gets scheduled and comes up and running.
		//
		testPod1CPUQuantity := resource.NewMilliQuantity(nodeAvailableMilliCPU/2, resource.DecimalSI)
		testPod2CPUQuantity := resource.NewMilliQuantity(nodeAvailableMilliCPU, resource.DecimalSI)
		testPod2CPUQuantityResized := resource.NewMilliQuantity(testPod1CPUQuantity.MilliValue()/2, resource.DecimalSI)
		framework.Logf("TEST1: testPod1 initial CPU request is '%dm'", testPod1CPUQuantity.MilliValue())
		framework.Logf("TEST1: testPod2 initial CPU request is '%dm'", testPod2CPUQuantity.MilliValue())
		framework.Logf("TEST1: testPod2 resized CPU request is '%dm'", testPod2CPUQuantityResized.MilliValue())

		c1 := []podresize.ResizableContainerInfo{
			{
				Name:      "c1",
				Resources: &cgroups.ContainerResources{CPUReq: testPod1CPUQuantity.String(), CPULim: testPod1CPUQuantity.String()},
			},
		}
		c2 := []podresize.ResizableContainerInfo{
			{
				Name:      "c2",
				Resources: &cgroups.ContainerResources{CPUReq: testPod2CPUQuantity.String(), CPULim: testPod2CPUQuantity.String()},
			},
		}
		patchTestpod2ToFitNode := fmt.Sprintf(`{
				"spec": {
					"containers": [
						{
							"name":      "c2",
							"resources": {"requests": {"cpu": "%dm"}, "limits": {"cpu": "%dm"}}
						}
					]
				}
			}`, testPod2CPUQuantityResized.MilliValue(), testPod2CPUQuantityResized.MilliValue())

		tStamp := strconv.Itoa(time.Now().Nanosecond())
		testPod1 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod1", tStamp, c1, nil)
		testPod1 = e2epod.MustMixinRestrictedPodSecurity(testPod1)
		testPod2 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod2", tStamp, c2, nil)
		testPod2 = e2epod.MustMixinRestrictedPodSecurity(testPod2)
		e2epod.SetNodeAffinity(&testPod1.Spec, node.Name)
		e2epod.SetNodeAffinity(&testPod2.Spec, node.Name)

		ginkgo.By(fmt.Sprintf("TEST1: Create pod '%s' that fits the node '%s'", testPod1.Name, node.Name))
		testPod1 = podClient.CreateSync(ctx, testPod1)
		gomega.Expect(testPod1.Status.Phase).To(gomega.Equal(v1.PodRunning))
		gomega.Expect(testPod1.Generation).To(gomega.BeEquivalentTo(1))

		ginkgo.By(fmt.Sprintf("TEST1: Create pod '%s' that won't fit node '%s' with pod '%s' on it", testPod2.Name, node.Name, testPod1.Name))
		testPod2 = podClient.Create(ctx, testPod2)
		err = e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, testPod2.Name, testPod2.Namespace)
		framework.ExpectNoError(err)
		gomega.Expect(testPod2.Status.Phase).To(gomega.Equal(v1.PodPending))
		gomega.Expect(testPod2.Generation).To(gomega.BeEquivalentTo(1))

		ginkgo.By(fmt.Sprintf("TEST1: Resize pod '%s' to fit in node '%s'", testPod2.Name, node.Name))
		testPod2, pErr := f.ClientSet.CoreV1().Pods(testPod2.Namespace).Patch(ctx,
			testPod2.Name, types.StrategicMergePatchType, []byte(patchTestpod2ToFitNode), metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(pErr, "failed to patch pod for resize")
		gomega.Expect(testPod2.Generation).To(gomega.BeEquivalentTo(2))

		ginkgo.By(fmt.Sprintf("TEST1: Verify that pod '%s' is running after resize", testPod2.Name))
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, testPod2))

		// Scheduler focused pod resize E2E test case #2
		//     1. With pod1 + pod2 running on node above, create pod3 that requests more CPU than available, verify pending.
		//     2. Resize pod1 down so that pod3 gets room to be scheduled.
		//     3. Verify that pod3 is scheduled and running.
		//
		nodeAllocatableMilliCPU2, nodeAvailableMilliCPU2 := getNodeAllocatableAndAvailableValues(ctx, f, &node, v1.ResourceCPU)
		framework.Logf("TEST2: Node '%s': NodeAllocatable MilliCPUs = %dm. MilliCPUs currently available to allocate = %dm.",
			node.Name, nodeAllocatableMilliCPU2, nodeAvailableMilliCPU2)
		testPod3CPUQuantity := resource.NewMilliQuantity(nodeAvailableMilliCPU2+testPod1CPUQuantity.MilliValue()/4, resource.DecimalSI)
		testPod1CPUQuantityResized := resource.NewMilliQuantity(testPod1CPUQuantity.MilliValue()/3, resource.DecimalSI)
		framework.Logf("TEST2: testPod1 MilliCPUs after resize '%dm'", testPod1CPUQuantityResized.MilliValue())

		c3 := []podresize.ResizableContainerInfo{
			{
				Name:      "c3",
				Resources: &cgroups.ContainerResources{CPUReq: testPod3CPUQuantity.String(), CPULim: testPod3CPUQuantity.String()},
			},
		}
		patchTestpod1ToMakeSpaceForPod3 := fmt.Sprintf(`{
				"spec": {
					"containers": [
						{
							"name":      "c1",
							"resources": {"requests": {"cpu": "%dm"},"limits": {"cpu": "%dm"}}
						}
					]
				}
			}`, testPod1CPUQuantityResized.MilliValue(), testPod1CPUQuantityResized.MilliValue())

		tStamp = strconv.Itoa(time.Now().Nanosecond())
		testPod3 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod3", tStamp, c3, nil)
		testPod3 = e2epod.MustMixinRestrictedPodSecurity(testPod3)
		e2epod.SetNodeAffinity(&testPod3.Spec, node.Name)

		ginkgo.By(fmt.Sprintf("TEST2: Create testPod3 '%s' that cannot fit node '%s' due to insufficient CPU.", testPod3.Name, node.Name))
		testPod3 = podClient.Create(ctx, testPod3)
		p3Err := e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, testPod3.Name, testPod3.Namespace)
		framework.ExpectNoError(p3Err, "failed to create pod3 or pod3 did not become pending!")
		gomega.Expect(testPod3.Status.Phase).To(gomega.Equal(v1.PodPending))
		gomega.Expect(testPod3.Generation).To(gomega.BeEquivalentTo(1))

		ginkgo.By(fmt.Sprintf("TEST2: Resize pod '%s' to make enough space for pod '%s'", testPod1.Name, testPod3.Name))
		testPod1, p1Err := f.ClientSet.CoreV1().Pods(testPod1.Namespace).Patch(ctx,
			testPod1.Name, types.StrategicMergePatchType, []byte(patchTestpod1ToMakeSpaceForPod3), metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(p1Err, "failed to patch pod for resize")
		gomega.Expect(testPod1.Generation).To(gomega.BeEquivalentTo(2))

		ginkgo.By(fmt.Sprintf("TEST2: Verify pod '%s' is running after successfully resizing pod '%s'", testPod3.Name, testPod1.Name))
		framework.Logf("TEST2: Pod '%s' CPU requests '%dm'", testPod1.Name, testPod1.Spec.Containers[0].Resources.Requests.Cpu().MilliValue())
		framework.Logf("TEST2: Pod '%s' CPU requests '%dm'", testPod2.Name, testPod2.Spec.Containers[0].Resources.Requests.Cpu().MilliValue())
		framework.Logf("TEST2: Pod '%s' CPU requests '%dm'", testPod3.Name, testPod3.Spec.Containers[0].Resources.Requests.Cpu().MilliValue())
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, testPod3))

		// Scheduler focused pod resize E2E test case #3
		//     1. With pod1 + pod2 + pod3 running on node above, attempt to scale up pod1 to requests more CPU than available, verify deferred.
		//     2. Delete pod2 + pod3 to make room for pod3.
		//     3. Verify that pod1 resize has completed.
		//     4. Attempt to scale up pod1 to request more cpu than the node has, verify infeasible.
		patchTestpod1ExceedNodeAvailable := fmt.Sprintf(`{
			"spec": {
				"containers": [
					{
						"name":      "c1",
						"resources": {"requests": {"cpu": "%dm"},"limits": {"cpu": "%dm"}}
					}
				]
			}
		}`, testPod1CPUQuantity.MilliValue(), testPod1CPUQuantity.MilliValue())

		testPod1CPUExceedingAllocatable := resource.NewMilliQuantity(nodeAllocatableMilliCPU*2, resource.DecimalSI)
		patchTestpod1ExceedNodeAllocatable := fmt.Sprintf(`{
			"spec": {
				"containers": [
					{
						"name":      "c1",
						"resources": {"requests": {"cpu": "%dm"},"limits": {"cpu": "%dm"}}
					}
				]
			}
		}`, testPod1CPUExceedingAllocatable.MilliValue(), testPod1CPUExceedingAllocatable.MilliValue())

		ginkgo.By(fmt.Sprintf("TEST3: Resize pod '%s' exceed node available capacity", testPod1.Name))
		testPod1, p1Err = f.ClientSet.CoreV1().Pods(testPod1.Namespace).Patch(ctx,
			testPod1.Name, types.StrategicMergePatchType, []byte(patchTestpod1ExceedNodeAvailable), metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(p1Err, "failed to patch pod for resize")
		gomega.Expect(testPod1.Generation).To(gomega.BeEquivalentTo(3))
		waitForPodDeferred(ctx, f, testPod1)

		ginkgo.By("deleting pods 2 and 3")
		e2epod.DeletePodsWithWait(ctx, f.ClientSet, []*v1.Pod{testPod2, testPod3})

		ginkgo.By(fmt.Sprintf("TEST3: Verify pod '%s' is resized successfully after pod deletion '%s' and '%s", testPod1.Name, testPod2.Name, testPod3.Name))
		expected := []podresize.ResizableContainerInfo{
			{
				Name:         "c1",
				Resources:    &cgroups.ContainerResources{CPUReq: testPod1CPUQuantity.String(), CPULim: testPod1CPUQuantity.String()},
				RestartCount: testPod1.Status.ContainerStatuses[0].RestartCount,
			},
		}
		resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod1, expected)
		podresize.ExpectPodResized(ctx, f, resizedPod, expected)

		ginkgo.By(fmt.Sprintf("TEST3: Resize pod '%s' to exceed the node capacity", testPod1.Name))
		testPod1, p1Err = f.ClientSet.CoreV1().Pods(testPod1.Namespace).Patch(ctx,
			testPod1.Name, types.StrategicMergePatchType, []byte(patchTestpod1ExceedNodeAllocatable), metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(p1Err, "failed to patch pod for resize")
		gomega.Expect(testPod1.Generation).To(gomega.BeEquivalentTo(4))
		framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, testPod1.Namespace, testPod1.Name, "display pod resize status as infeasible", f.Timeouts.PodStart, func(pod *v1.Pod) (bool, error) {
			return helpers.IsPodResizeInfeasible(pod), nil
		}))

		ginkgo.By("deleting pod 1")
		delErr1 := e2epod.DeletePodWithWait(ctx, f.ClientSet, testPod1)
		framework.ExpectNoError(delErr1, "failed to delete pod %s", testPod1.Name)
	})
}

func doPodResizeRetryDeferredTests(f *framework.Framework) {
	ginkgo.It("pod-resize-retry-deferred-test-1", func(ctx context.Context) {
		// Deferred resize E2E test case #1:
		// 	   1. Create pod1 and pod2 and pod3 on node.
		// 	   2. Resize pod3 to request more cpu than available, verify the resize is deferred.
		//	   3. Resize pod1 down to make space for pod3, verify pod3's resize has completed.

		podClient := e2epod.NewPodClient(f)
		nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err, "failed to get running nodes")
		gomega.Expect(nodes.Items).ShouldNot(gomega.BeEmpty())
		framework.Logf("Found %d schedulable nodes", len(nodes.Items))

		ginkgo.By("Find node CPU resources available for allocation!")
		node := nodes.Items[0]
		nodeAllocatableMilliCPU, nodeAvailableMilliCPU := getNodeAllocatableAndAvailableValues(ctx, f, &node, v1.ResourceCPU)
		framework.Logf("Node '%s': NodeAllocatable MilliCPUs = %dm. MilliCPUs currently available to allocate = %dm.",
			node.Name, nodeAllocatableMilliCPU, nodeAvailableMilliCPU)

		testPod1CPUQuantity := resource.NewMilliQuantity(nodeAvailableMilliCPU/2, resource.DecimalSI)
		testPod2CPUQuantity := resource.NewMilliQuantity(testPod1CPUQuantity.MilliValue()/2, resource.DecimalSI)
		framework.Logf("testPod1 initial CPU request is '%dm'", testPod1CPUQuantity.MilliValue())
		framework.Logf("testPod2 initial CPU request is '%dm'", testPod2CPUQuantity.MilliValue())

		c1 := []podresize.ResizableContainerInfo{
			{
				Name:      "c1",
				Resources: &cgroups.ContainerResources{CPUReq: testPod1CPUQuantity.String(), CPULim: testPod1CPUQuantity.String()},
			},
		}
		c2 := []podresize.ResizableContainerInfo{
			{
				Name:      "c2",
				Resources: &cgroups.ContainerResources{CPUReq: testPod2CPUQuantity.String(), CPULim: testPod2CPUQuantity.String()},
			},
		}
		tStamp := strconv.Itoa(time.Now().Nanosecond())
		testPod1 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod1", tStamp, c1, nil)
		testPod1 = e2epod.MustMixinRestrictedPodSecurity(testPod1)
		testPod2 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod2", tStamp, c2, nil)
		testPod2 = e2epod.MustMixinRestrictedPodSecurity(testPod2)
		e2epod.SetNodeAffinity(&testPod1.Spec, node.Name)
		e2epod.SetNodeAffinity(&testPod2.Spec, node.Name)

		ginkgo.By(fmt.Sprintf("Create pod '%s' that fits the node '%s'", testPod1.Name, node.Name))
		testPod1 = podClient.CreateSync(ctx, testPod1)
		gomega.Expect(testPod1.Status.Phase).To(gomega.Equal(v1.PodRunning))
		gomega.Expect(testPod1.Generation).To(gomega.BeEquivalentTo(1))

		ginkgo.By(fmt.Sprintf("Create pod '%s' that fits the node '%s'", testPod2.Name, node.Name))
		testPod2 = podClient.CreateSync(ctx, testPod2)
		gomega.Expect(testPod2.Status.Phase).To(gomega.Equal(v1.PodRunning))
		gomega.Expect(testPod2.Generation).To(gomega.BeEquivalentTo(1))

		nodeAllocatableMilliCPU2, nodeAvailableMilliCPU2 := getNodeAllocatableAndAvailableValues(ctx, f, &node, v1.ResourceCPU)
		framework.Logf("Node '%s': NodeAllocatable MilliCPUs = %dm. MilliCPUs currently available to allocate = %dm.",
			node.Name, nodeAllocatableMilliCPU2, nodeAvailableMilliCPU2)

		testPod3CPUQuantity := resource.NewMilliQuantity(nodeAvailableMilliCPU2/4, resource.DecimalSI)
		testPod3CPUQuantityResized := resource.NewMilliQuantity(nodeAvailableMilliCPU2+testPod1CPUQuantity.MilliValue()/4, resource.DecimalSI)
		framework.Logf("testPod3 MilliCPUs after resize '%dm'", testPod3CPUQuantityResized.MilliValue())

		testPod1CPUQuantityResizedCPU := resource.NewMilliQuantity(testPod1CPUQuantity.MilliValue()/3, resource.DecimalSI)
		framework.Logf("testPod1 MilliCPUs after resize '%dm'", testPod1CPUQuantityResizedCPU.MilliValue())

		c3 := []podresize.ResizableContainerInfo{
			{
				Name:      "c3",
				Resources: &cgroups.ContainerResources{CPUReq: testPod3CPUQuantity.String(), CPULim: testPod3CPUQuantity.String()},
			},
		}
		patchTestpod3ToDeferred := fmt.Sprintf(`{
			"spec": {
				"containers": [
					{
						"name":      "c3",
						"resources": {"requests": {"cpu": "%dm"},"limits": {"cpu": "%dm"}}
					}
				]
			}
		}`, testPod3CPUQuantityResized.MilliValue(), testPod3CPUQuantityResized.MilliValue())
		patchTestpod1ToMakeSpaceForPod3 := fmt.Sprintf(`{
				"spec": {
					"containers": [
						{
							"name":      "c1",
							"resources": {"requests": {"cpu": "%dm"},"limits": {"cpu": "%dm"}}
						}
					]
				}
			}`, testPod1CPUQuantityResizedCPU.MilliValue(), testPod1CPUQuantityResizedCPU.MilliValue())

		tStamp = strconv.Itoa(time.Now().Nanosecond())
		testPod3 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod3", tStamp, c3, nil)
		testPod3 = e2epod.MustMixinRestrictedPodSecurity(testPod3)
		e2epod.SetNodeAffinity(&testPod3.Spec, node.Name)

		ginkgo.By(fmt.Sprintf("Create pod '%s' that fits the node '%s'", testPod3.Name, node.Name))
		testPod3 = podClient.CreateSync(ctx, testPod3)
		gomega.Expect(testPod3.Status.Phase).To(gomega.Equal(v1.PodRunning))
		gomega.Expect(testPod3.Generation).To(gomega.BeEquivalentTo(1))

		ginkgo.By(fmt.Sprintf("Resize pod '%s' that cannot fit node due to insufficient CPU", testPod3.Name))
		testPod3, p3Err := f.ClientSet.CoreV1().Pods(testPod3.Namespace).Patch(ctx,
			testPod3.Name, types.StrategicMergePatchType, []byte(patchTestpod3ToDeferred), metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(p3Err, "failed to patch pod for resize")
		waitForPodDeferred(ctx, f, testPod3)

		ginkgo.By(fmt.Sprintf("Resize pod '%s' to make enough space for pod '%s'", testPod1.Name, testPod3.Name))
		testPod1, p1Err := f.ClientSet.CoreV1().Pods(testPod1.Namespace).Patch(ctx,
			testPod1.Name, types.StrategicMergePatchType, []byte(patchTestpod1ToMakeSpaceForPod3), metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(p1Err, "failed to patch pod for resize")
		gomega.Expect(testPod1.Generation).To(gomega.BeEquivalentTo(2))

		ginkgo.By(fmt.Sprintf("Verify pod '%s' is resized successfully after pod resize '%s'", testPod3.Name, testPod1.Name))
		expected := []podresize.ResizableContainerInfo{
			{
				Name:      "c3",
				Resources: &cgroups.ContainerResources{CPUReq: testPod3CPUQuantityResized.String(), CPULim: testPod3CPUQuantityResized.String()},
			},
		}
		resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod3, expected)
		podresize.ExpectPodResized(ctx, f, resizedPod, expected)

		ginkgo.By("deleting pods")
		e2epod.DeletePodsWithWait(ctx, f.ClientSet, []*v1.Pod{testPod1, testPod2, testPod3})
	})

	ginkgo.It("pod-resize-retry-deferred-test-2", func(ctx context.Context) {
		// Deferred resize E2E test case #2:
		//	   1. Create 5 pods on the node, where the first one has 2/3 of the node allocatable CPU,
		//        and the remaining ones each have 1/16 of the node allocatable CPU.
		//     2. Resize all remaining pods to request 2/3 of the node allocatable CPU, verify deferred.
		//     3. Delete the first pod, verify the pod with the highest priority has its resize accepted.
		//     4. Repeat step 3 until all but the last pod has been deleted.

		podClient := e2epod.NewPodClient(f)
		nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err, "failed to get running nodes")
		gomega.Expect(nodes.Items).ShouldNot(gomega.BeEmpty())
		framework.Logf("Found %d schedulable nodes", len(nodes.Items))

		ginkgo.By("Find node CPU and memory resources available for allocation!")
		node := nodes.Items[0]

		nodeAllocatableMilliCPU, nodeAvailableMilliCPU := getNodeAllocatableAndAvailableValues(ctx, f, &node, v1.ResourceCPU)
		framework.Logf("Node '%s': NodeAllocatable MilliCPUs = %dm. MilliCPUs currently available to allocate = %dm.",
			node.Name, nodeAllocatableMilliCPU, nodeAvailableMilliCPU)
		framework.Logf("Node '%s': NodeAllocatable MilliCPUs = %dm. MilliCPUs currently available to allocate = %dm.",
			node.Name, nodeAllocatableMilliCPU, nodeAvailableMilliCPU)

		majorityCPUQuantity := resource.NewMilliQuantity(2*nodeAvailableMilliCPU/3, resource.DecimalSI)
		littleCPUQuantity := resource.NewMilliQuantity(nodeAvailableMilliCPU/16, resource.DecimalSI)
		containerWithMajorityCPU := []podresize.ResizableContainerInfo{
			{
				Name:      "c",
				Resources: &cgroups.ContainerResources{CPUReq: majorityCPUQuantity.String(), CPULim: majorityCPUQuantity.String()},
			},
		}
		containerWithLittleCPU := []podresize.ResizableContainerInfo{
			{
				Name:      "c",
				Resources: &cgroups.ContainerResources{CPUReq: littleCPUQuantity.String(), CPULim: littleCPUQuantity.String()},
			},
		}
		containerWithLittleCPUGuaranteedQoS := []podresize.ResizableContainerInfo{
			{
				Name: "c",
				Resources: &cgroups.ContainerResources{
					CPUReq: littleCPUQuantity.String(),
					CPULim: littleCPUQuantity.String(),
					MemReq: "100Mi",
					MemLim: "100Mi",
				},
			},
		}

		tStamp := strconv.Itoa(time.Now().Nanosecond())
		testPod1 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod1", tStamp, containerWithMajorityCPU, nil)
		testPod1 = e2epod.MustMixinRestrictedPodSecurity(testPod1)
		e2epod.SetNodeAffinity(&testPod1.Spec, node.Name)

		ginkgo.By(fmt.Sprintf("Create pod '%s' with 2/3 of the node cpu", testPod1.Name))
		testPod1 = podClient.CreateSync(ctx, testPod1)
		gomega.Expect(testPod1.Status.Phase).To(gomega.Equal(v1.PodRunning))

		// Create pod2 with 1/16 of the node allocatable CPU, with high priority based on priority class.
		testPod2 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod2", tStamp, containerWithLittleCPU, nil)
		testPod2 = e2epod.MustMixinRestrictedPodSecurity(testPod2)
		pc, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(ctx, &schedulingv1.PriorityClass{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("testpod2-priority-class-%s", testPod1.Namespace),
			},
			Value: 1000,
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		defer func() {
			framework.ExpectNoError(f.ClientSet.SchedulingV1().PriorityClasses().Delete(ctx, pc.Name, metav1.DeleteOptions{}))
		}()

		testPod2.Spec.PriorityClassName = pc.Name
		e2epod.SetNodeAffinity(&testPod2.Spec, node.Name)

		ginkgo.By(fmt.Sprintf("Create pod '%s' with 1/16 of the node cpu and high priority class", testPod2.Name))
		testPod2 = podClient.CreateSync(ctx, testPod2)
		gomega.Expect(testPod2.Status.Phase).To(gomega.Equal(v1.PodRunning))

		// Create pod3 with 1/16 of the node allocatable CPU, that is a "guaranteed" pod (all others should be "burstable").
		testPod3 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod3", tStamp, containerWithLittleCPUGuaranteedQoS, nil)
		testPod3 = e2epod.MustMixinRestrictedPodSecurity(testPod3)
		e2epod.SetNodeAffinity(&testPod3.Spec, node.Name)

		ginkgo.By(fmt.Sprintf("Create pod '%s' with 1/16 of the node cpu and guaranteed qos", testPod3.Name))
		testPod3 = podClient.CreateSync(ctx, testPod3)
		gomega.Expect(testPod3.Status.Phase).To(gomega.Equal(v1.PodRunning))

		// Create pod4 with 1/16 of the node allocatable CPU.
		testPod4 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod4", tStamp, containerWithLittleCPU, nil)
		testPod4 = e2epod.MustMixinRestrictedPodSecurity(testPod4)
		e2epod.SetNodeAffinity(&testPod4.Spec, node.Name)

		ginkgo.By(fmt.Sprintf("Create pod '%s' with 1/16 of the node cpu", testPod4.Name))
		testPod4 = podClient.CreateSync(ctx, testPod4)
		gomega.Expect(testPod4.Status.Phase).To(gomega.Equal(v1.PodRunning))

		// Create pod5 with 1/16 of the node allocatable CPU.
		testPod5 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod5", tStamp, containerWithLittleCPU, nil)
		testPod5 = e2epod.MustMixinRestrictedPodSecurity(testPod5)
		e2epod.SetNodeAffinity(&testPod5.Spec, node.Name)

		ginkgo.By(fmt.Sprintf("Create pod '%s' with 1/16 of the node cpu", testPod5.Name))
		testPod5 = podClient.CreateSync(ctx, testPod5)
		gomega.Expect(testPod5.Status.Phase).To(gomega.Equal(v1.PodRunning))

		patchTestPod := fmt.Sprintf(`{
			"spec": {
				"containers": [
					{
						"name":      "c",
						"resources": {"requests": {"cpu": "%dm"},"limits": {"cpu": "%dm"}}
					}
				]
			}
		}`, majorityCPUQuantity.MilliValue(), majorityCPUQuantity.MilliValue())

		// Resize requests are done in an arbitrary order, to verify that the priority based on priority class
		// or qos class takes precedent over the order of the requests.

		// Attempt pod4 resize request to 2/3 of the node allocatable CPU, verify deferred.
		ginkgo.By(fmt.Sprintf("Resize pod '%s'", testPod4.Name))
		testPod4, err = f.ClientSet.CoreV1().Pods(testPod4.Namespace).Patch(ctx,
			testPod4.Name, types.StrategicMergePatchType, []byte(patchTestPod), metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(err, "failed to patch pod for resize")
		waitForPodDeferred(ctx, f, testPod4)

		// Attempt pod3 resize request to 2/3 of the node allocatable CPU, verify deferred.
		ginkgo.By(fmt.Sprintf("Resize pod '%s'", testPod3.Name))
		testPod3, err = f.ClientSet.CoreV1().Pods(testPod3.Namespace).Patch(ctx,
			testPod3.Name, types.StrategicMergePatchType, []byte(patchTestPod), metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(err, "failed to patch pod for resize")
		waitForPodDeferred(ctx, f, testPod3)

		// Attempt pod2 resize request to 2/3 of the node allocatable CPU, verify deferred.
		ginkgo.By(fmt.Sprintf("Resize pod '%s'", testPod2.Name))
		testPod2, err = f.ClientSet.CoreV1().Pods(testPod2.Namespace).Patch(ctx,
			testPod2.Name, types.StrategicMergePatchType, []byte(patchTestPod), metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(err, "failed to patch pod for resize")
		waitForPodDeferred(ctx, f, testPod2)

		// Attempt pod5 resize request to 2/3 of the node allocatable CPU, verify deferred.
		ginkgo.By(fmt.Sprintf("Resize pod '%s'", testPod5.Name))
		testPod5, err = f.ClientSet.CoreV1().Pods(testPod5.Namespace).Patch(ctx,
			testPod5.Name, types.StrategicMergePatchType, []byte(patchTestPod), metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(err, "failed to patch pod for resize")
		waitForPodDeferred(ctx, f, testPod5)

		// Delete pod1. Verify pod2's resize has completed, while the others are still deferred.
		ginkgo.By("deleting pod1")
		delErr1 := e2epod.DeletePodWithWait(ctx, f.ClientSet, testPod1)
		framework.ExpectNoError(delErr1, "failed to delete pod %s", testPod1.Name)

		ginkgo.By(fmt.Sprintf("Verify pod '%s' is resized successfully after pod '%s' deleted", testPod2.Name, testPod1.Name))
		expected := []podresize.ResizableContainerInfo{
			{
				Name:      "c",
				Resources: &cgroups.ContainerResources{CPUReq: majorityCPUQuantity.String(), CPULim: majorityCPUQuantity.String()},
			},
		}
		resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod2, expected)
		podresize.ExpectPodResized(ctx, f, resizedPod, expected)
		waitForPodDeferred(ctx, f, testPod3)
		waitForPodDeferred(ctx, f, testPod4)
		waitForPodDeferred(ctx, f, testPod5)

		// Delete pod2. Verify pod3's resize has completed, while the others are still deferred.
		ginkgo.By("deleting pod2")
		delErr2 := e2epod.DeletePodWithWait(ctx, f.ClientSet, testPod2)
		framework.ExpectNoError(delErr2, "failed to delete pod %s", testPod2.Name)

		ginkgo.By(fmt.Sprintf("Verify pod '%s' is resized successfully after pod '%s' deleted", testPod3.Name, testPod2.Name))
		expected = []podresize.ResizableContainerInfo{
			{
				Name: "c",
				Resources: &cgroups.ContainerResources{
					CPUReq: majorityCPUQuantity.String(),
					CPULim: majorityCPUQuantity.String(),
					MemReq: "100Mi",
					MemLim: "100Mi",
				},
			},
		}
		resizedPod = podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod3, expected)
		podresize.ExpectPodResized(ctx, f, resizedPod, expected)
		waitForPodDeferred(ctx, f, testPod4)
		waitForPodDeferred(ctx, f, testPod5)

		// Delete pod3. Verify pod4's resize has completed, while the others are still deferred.
		ginkgo.By("deleting pod3")
		delErr3 := e2epod.DeletePodWithWait(ctx, f.ClientSet, testPod3)
		framework.ExpectNoError(delErr3, "failed to delete pod %s", testPod3.Name)

		ginkgo.By(fmt.Sprintf("Verify pod '%s' is resized successfully after pod '%s' deleted", testPod4.Name, testPod3.Name))
		expected = []podresize.ResizableContainerInfo{
			{
				Name:      "c",
				Resources: &cgroups.ContainerResources{CPUReq: majorityCPUQuantity.String(), CPULim: majorityCPUQuantity.String()},
			},
		}
		resizedPod = podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod4, expected)
		podresize.ExpectPodResized(ctx, f, resizedPod, expected)
		waitForPodDeferred(ctx, f, testPod5)

		// Delete pod4. Verify pod5's resize has completed.
		ginkgo.By("deleting pod4")
		delErr4 := e2epod.DeletePodWithWait(ctx, f.ClientSet, testPod4)
		framework.ExpectNoError(delErr4, "failed to delete pod %s", testPod4.Name)

		ginkgo.By(fmt.Sprintf("Verify pod '%s' is resized successfully after pod '%s' deleted", testPod5.Name, testPod4.Name))
		expected = []podresize.ResizableContainerInfo{
			{
				Name:      "c",
				Resources: &cgroups.ContainerResources{CPUReq: majorityCPUQuantity.String(), CPULim: majorityCPUQuantity.String()},
			},
		}
		resizedPod = podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod5, expected)
		podresize.ExpectPodResized(ctx, f, resizedPod, expected)

		ginkgo.By("deleting pod5")
		delErr5 := e2epod.DeletePodWithWait(ctx, f.ClientSet, testPod5)
		framework.ExpectNoError(delErr5, "failed to delete pod %s", testPod5.Name)
	})

	ginkgo.It("pod-resize-retry-deferred-test-3", func(ctx context.Context) {
		// Deferred resize E2E test case #3:
		// 	   1. Create pod1, pod2, pod3, and pod4 on node, each starting with 1/4 of the node's allocatable CPU and and 1/5 the memory.
		// 	   2. Resize pod2 to request more CPU than available (1/3), along with a decrease in memory (1/24), verify the resize is deferred.
		//	   3. Resize pod3 to request more memory than available (2/3), along with a decrease in CPU (1/24), verify the resize is deferred.
		// 	   4. Resize pod4 to request more CPU than available (5/8), verify the resize is deferred.
		//	   5. The deferred resizes above are chosen carefully such that:
		//	      - deletion of pod1 should make room for pod2's resize (but not pod3 or pod4).
		//	      - pod2's resize should make room for pod3's resize (but not pod4).
		//	      - pod3's resize should make room for pod4's resize.
		//     6. Delete pod1, verify the chain of deferred resizes is actuated.

		podClient := e2epod.NewPodClient(f)
		nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err, "failed to get running nodes")
		gomega.Expect(nodes.Items).ShouldNot(gomega.BeEmpty())
		framework.Logf("Found %d schedulable nodes", len(nodes.Items))

		ginkgo.By("Find node CPU and memory resources available for allocation!")
		node := nodes.Items[0]

		nodeAllocatableMilliCPU, initNodeAvailableMilliCPU := getNodeAllocatableAndAvailableValues(ctx, f, &node, v1.ResourceCPU)
		framework.Logf("Node '%s': NodeAllocatable MilliCPUs = %dm. MilliCPUs currently available to allocate = %dm.",
			node.Name, nodeAllocatableMilliCPU, initNodeAvailableMilliCPU)

		nodeAllocatableMem, initNodeAvailableMem := getNodeAllocatableAndAvailableValues(ctx, f, &node, v1.ResourceMemory)
		framework.Logf("Node '%s': NodeAllocatable Memory = %d. Memory currently available to allocate = %d.",
			node.Name, nodeAllocatableMem, initNodeAvailableMem)

		initialCPUQuantity := resource.NewMilliQuantity(initNodeAvailableMilliCPU/4, resource.DecimalSI)
		initialMemoryQuantity := resource.NewQuantity(initNodeAvailableMem/5, resource.DecimalSI)
		framework.Logf("initial CPU request is '%dm'", initialCPUQuantity.MilliValue())
		framework.Logf("initial Memory request is '%d'", initialMemoryQuantity.Value())

		c := []podresize.ResizableContainerInfo{
			{
				Name: "c",
				Resources: &cgroups.ContainerResources{
					CPUReq: initialCPUQuantity.String(),
					MemReq: initialMemoryQuantity.String(),
				},
			},
		}

		tStamp := strconv.Itoa(time.Now().Nanosecond())
		testPod1 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod1", tStamp, c, nil)
		testPod1 = e2epod.MustMixinRestrictedPodSecurity(testPod1)
		e2epod.SetNodeAffinity(&testPod1.Spec, node.Name)

		testPod2 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod2", tStamp, c, nil)
		testPod2 = e2epod.MustMixinRestrictedPodSecurity(testPod2)
		e2epod.SetNodeAffinity(&testPod2.Spec, node.Name)

		testPod3 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod3", tStamp, c, nil)
		testPod3 = e2epod.MustMixinRestrictedPodSecurity(testPod3)
		e2epod.SetNodeAffinity(&testPod3.Spec, node.Name)

		testPod4 := podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod4", tStamp, c, nil)
		testPod4 = e2epod.MustMixinRestrictedPodSecurity(testPod4)
		e2epod.SetNodeAffinity(&testPod4.Spec, node.Name)

		testPods := []*v1.Pod{testPod1, testPod2, testPod3, testPod4}
		ginkgo.By(fmt.Sprintf("Create pods that fits the node '%s'", node.Name))
		podClient.CreateBatch(ctx, testPods)

		testPod2CPUQuantityResized := resource.NewMilliQuantity(initNodeAvailableMilliCPU/3, resource.DecimalSI)
		testPod2MemoryQuantityResized := resource.NewQuantity(initNodeAvailableMem/24, resource.DecimalSI)

		testPod3CPUQuantityResized := resource.NewMilliQuantity(initNodeAvailableMilliCPU/24, resource.DecimalSI)
		testPod3MemoryQuantityResized := resource.NewQuantity(2*initNodeAvailableMem/3, resource.DecimalSI)

		testPod4CPUQuantityResized := resource.NewMilliQuantity(initNodeAvailableMilliCPU/2+initNodeAvailableMilliCPU/8, resource.DecimalSI)
		testPod4MemoryQuantityResized := initialMemoryQuantity

		expectedTestPod2Resized := []podresize.ResizableContainerInfo{
			{
				Name: "c",
				Resources: &cgroups.ContainerResources{
					CPUReq: testPod2CPUQuantityResized.String(),
					MemReq: testPod2MemoryQuantityResized.String(),
				},
			},
		}
		expectedTestPod3Resized := []podresize.ResizableContainerInfo{
			{
				Name: "c",
				Resources: &cgroups.ContainerResources{
					CPUReq: testPod3CPUQuantityResized.String(),
					MemReq: testPod3MemoryQuantityResized.String(),
				},
			},
		}
		expectedTestPod4Resized := []podresize.ResizableContainerInfo{
			{
				Name: "c",
				Resources: &cgroups.ContainerResources{
					CPUReq: testPod4CPUQuantityResized.String(),
					MemReq: testPod4MemoryQuantityResized.String(),
				},
			},
		}

		patchTestPod2ToDeferred := podresize.MakeResizePatch(c, expectedTestPod2Resized, nil, nil)
		patchTestPod3ToDeferred := podresize.MakeResizePatch(c, expectedTestPod3Resized, nil, nil)
		patchTestPod4ToDeferred := podresize.MakeResizePatch(c, expectedTestPod4Resized, nil, nil)

		patches := []string{string(patchTestPod2ToDeferred), string(patchTestPod3ToDeferred), string(patchTestPod4ToDeferred)}

		for i := range patches {
			testPod := testPods[i+1]
			patch := patches[i]
			ginkgo.By(fmt.Sprintf("Resize pod '%s' that cannot fit node due to insufficient CPU or memory", testPod.Name))
			testPod, err = f.ClientSet.CoreV1().Pods(testPod.Namespace).Patch(ctx,
				testPod.Name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{}, "resize")
			framework.ExpectNoError(err, "failed to patch pod for resize")
			waitForPodDeferred(ctx, f, testPod)
		}

		ginkgo.By("deleting pod 1")
		delErr1 := e2epod.DeletePodWithWait(ctx, f.ClientSet, testPod1)
		framework.ExpectNoError(delErr1, "failed to delete pod %s", testPod1.Name)

		ginkgo.By(fmt.Sprintf("Verify pod '%s' is resized successfully after pod deletion '%s'", testPod2.Name, testPod1.Name))
		resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod2, expectedTestPod2Resized)
		podresize.ExpectPodResized(ctx, f, resizedPod, expectedTestPod2Resized)

		ginkgo.By(fmt.Sprintf("Verify pod '%s' is resized successfully after pod resize '%s'", testPod3.Name, testPod2.Name))
		resizedPod = podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod3, expectedTestPod3Resized)
		podresize.ExpectPodResized(ctx, f, resizedPod, expectedTestPod3Resized)

		ginkgo.By(fmt.Sprintf("Verify pod '%s' is resized successfully after pod resize '%s'", testPod4.Name, testPod3.Name))
		resizedPod = podresize.WaitForPodResizeActuation(ctx, f, podClient, testPod4, expectedTestPod4Resized)
		podresize.ExpectPodResized(ctx, f, resizedPod, expectedTestPod4Resized)

		ginkgo.By("deleting pods")
		e2epod.DeletePodsWithWait(ctx, f.ClientSet, testPods)
	})
}

var _ = SIGDescribe(framework.WithSerial(), "Pod InPlace Resize Container (scheduler-focused)", framework.WithFeatureGate(features.InPlacePodVerticalScaling), func() {
	f := framework.NewDefaultFramework("pod-resize-scheduler-tests")
	ginkgo.BeforeEach(func(ctx context.Context) {
		node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		if framework.NodeOSDistroIs("windows") || e2enode.IsARM64(node) {
			e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
		}
	})
	doPodResizeSchedulerTests(f)
})

var _ = SIGDescribe(framework.WithSerial(), "Pod InPlace Resize Container (deferred-resizes)", framework.WithFeatureGate(features.InPlacePodVerticalScaling), func() {
	f := framework.NewDefaultFramework("pod-resize-deferred-resize-tests")
	ginkgo.BeforeEach(func(ctx context.Context) {
		node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		if framework.NodeOSDistroIs("windows") || e2enode.IsARM64(node) {
			e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
		}
	})
	doPodResizeRetryDeferredTests(f)
})

var _ = SIGDescribe("Pod InPlace Resize Container (resource-quota)", framework.WithFeatureGate(features.InPlacePodVerticalScaling), func() {
	f := framework.NewDefaultFramework("pod-resize-resource-quota-tests")

	ginkgo.BeforeEach(func(ctx context.Context) {
		node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		if framework.NodeOSDistroIs("windows") || e2enode.IsARM64(node) {
			e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
		}
	})
	doPodResizeResourceQuotaTests(f)
})

var _ = SIGDescribe("Pod InPlace Resize Container (limit-ranger)", framework.WithFeatureGate(features.InPlacePodVerticalScaling), func() {
	f := framework.NewDefaultFramework("pod-resize-limit-ranger-tests")

	ginkgo.BeforeEach(func(ctx context.Context) {
		node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		if framework.NodeOSDistroIs("windows") || e2enode.IsARM64(node) {
			e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
		}
	})
	doPodResizeLimitRangerTests(f)
})

func waitForResourceQuota(ctx context.Context, c clientset.Interface, ns, quotaName string) error {
	return framework.Gomega().Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (v1.ResourceList, error) {
		quota, err := c.CoreV1().ResourceQuotas(ns).Get(ctx, quotaName, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		return quota.Status.Used, nil
	})).WithTimeout(framework.PollShortTimeout).ShouldNot(gomega.BeEmpty())
}

// Calculate available resource. nodeAvailable = nodeAllocatable - sum(podAllocated). If resourceName is "CPU", the values
// returned are in MilliValues.
func getNodeAllocatableAndAvailableValues(ctx context.Context, f *framework.Framework, n *v1.Node, resourceName v1.ResourceName) (int64, int64) {
	var nodeAllocatable int64
	switch resourceName {
	case v1.ResourceCPU:
		nodeAllocatable = n.Status.Allocatable.Cpu().MilliValue()
	case v1.ResourceMemory:
		nodeAllocatable = n.Status.Allocatable.Memory().Value()
	default:
		framework.Failf("unexpected resource type %q; expected either 'CPU' or 'Memory'", resourceName)
	}

	gomega.Expect(n.Status.Allocatable).ShouldNot(gomega.BeEmpty(), "allocatable")
	podAllocated := int64(0)

	// Exclude pods that are in the Succeeded or Failed states
	selector := fmt.Sprintf("spec.nodeName=%s,status.phase!=%v,status.phase!=%v", n.Name, v1.PodSucceeded, v1.PodFailed)
	listOptions := metav1.ListOptions{FieldSelector: selector}
	podList, err := f.ClientSet.CoreV1().Pods(metav1.NamespaceAll).List(ctx, listOptions)

	framework.ExpectNoError(err, "failed to get running pods")
	framework.Logf("Found %d pods on node '%s'", len(podList.Items), n.Name)
	for _, pod := range podList.Items {
		podRequest := resourceapi.GetResourceRequest(&pod, resourceName)
		podAllocated += podRequest
	}
	nodeAvailable := nodeAllocatable - podAllocated
	if nodeAvailable < 0 {
		framework.Failf("unexpected negative value of nodeAvailable %d", nodeAvailable)
	}

	return nodeAllocatable, nodeAvailable
}

func waitForPodDeferred(ctx context.Context, f *framework.Framework, testPod *v1.Pod) {
	framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, testPod.Namespace, testPod.Name, "display pod resize status as deferred", f.Timeouts.PodStart, func(pod *v1.Pod) (bool, error) {
		return helpers.IsPodResizeDeferred(pod), nil
	}))
}
