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
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletdevicepluginv1beta1 "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e_node/testdeviceplugin"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Device Plugin Allocation", framework.WithSerial(), feature.DevicePlugin, func() {
	f := framework.NewDefaultFramework("device-plugin-allocation")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("does not expose in-flight device allocations through pod-resources", func(ctx context.Context) {
		resourceName := fmt.Sprintf("test.device/%s", f.UniqueName)
		const deviceID = "device-1"

		allocateStarted := make(chan struct{})
		continueAllocate := make(chan struct{})
		var signalStarted sync.Once
		var releaseAllocate sync.Once

		plugin := testdeviceplugin.NewDevicePlugin(nil)
		plugin.SetAllocateFunc(func(ctx context.Context, _ *kubeletdevicepluginv1beta1.AllocateRequest) error {
			signalStarted.Do(func() { close(allocateStarted) })
			select {
			case <-continueAllocate:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		defer plugin.Stop()
		defer releaseAllocate.Do(func() { close(continueAllocate) })

		err := plugin.RegisterDevicePlugin(ctx, f.UniqueName, resourceName, []*kubeletdevicepluginv1beta1.Device{{
			ID:     deviceID,
			Health: kubeletdevicepluginv1beta1.Healthy,
		}})
		framework.ExpectNoError(err)

		ginkgo.By("waiting for the device resource to become allocatable")
		gomega.Eventually(ctx, func(ctx context.Context) int64 {
			node := getLocalNode(ctx, f)
			quantity, found := node.Status.Allocatable[v1.ResourceName(resourceName)]
			if !found {
				return 0
			}
			return quantity.Value()
		}, time.Minute, framework.Poll).Should(gomega.Equal(int64(1)))

		podClient := e2epod.NewPodClient(f)
		podCommand := fmt.Sprintf("sleep %s", sleepIntervalForever)

		ginkgo.By("starting an allocation and pausing its device plugin RPC")
		pod := podClient.Create(ctx, makeBusyboxPod(resourceName, podCommand))
		gomega.Eventually(ctx, allocateStarted, f.Timeouts.PodStart).Should(gomega.BeClosed())

		ginkgo.By("confirming the reservation is hidden from pod-resources")
		podResources, err := getV1NodeDevices(ctx)
		framework.ExpectNoError(err)
		err, found := checkPodResourcesAssignment(
			podResources,
			pod.Namespace,
			pod.Name,
			pod.Spec.Containers[0].Name,
			resourceName,
			[]string{},
		)
		framework.ExpectNoError(err)
		gomega.Expect(found).To(gomega.BeTrueBecause(
			"expected pod %s/%s to be present in pod-resources",
			pod.Namespace,
			pod.Name,
		))

		ginkgo.By("allowing the allocation to commit")
		releaseAllocate.Do(func() { close(continueAllocate) })
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))

		ginkgo.By("confirming the committed allocation is visible")
		podResources, err = getV1NodeDevices(ctx)
		framework.ExpectNoError(err)
		err, found = checkPodResourcesAssignment(
			podResources,
			pod.Namespace,
			pod.Name,
			pod.Spec.Containers[0].Name,
			resourceName,
			[]string{deviceID},
		)
		framework.ExpectNoError(err)
		gomega.Expect(found).To(gomega.BeTrueBecause(
			"expected pod %s/%s to have a committed device allocation",
			pod.Namespace,
			pod.Name,
		))
	})

	f.Context("with pod-level resource managers", feature.PodLevelResources, feature.PodLevelResourceManagers, framework.WithFeatureGate(features.PodLevelResources), framework.WithFeatureGate(features.PodLevelResourceManagers), func() {
		tempSetCurrentKubeletConfig(f, func(_ context.Context, cfg *kubeletconfig.KubeletConfiguration) {
			if cfg.FeatureGates == nil {
				cfg.FeatureGates = map[string]bool{}
			}
			cfg.FeatureGates[string(features.PodLevelResources)] = true
			cfg.FeatureGates[string(features.PodLevelResourceManagers)] = true
			cfg.TopologyManagerScope = topologymanager.PodTopologyScope
			cfg.TopologyManagerPolicy = topologymanager.PolicyBestEffort
		})

		ginkgo.It("allocates container devices during admission for a pod with pod-level resources", func(ctx context.Context) {
			resourceName := fmt.Sprintf("test.device/%s", f.UniqueName)
			const deviceID = "device-1"
			plugin := testdeviceplugin.NewDevicePlugin(nil)
			// PreStartContainer checks the allocation cache before start-time re-allocation,
			// so this pod cannot start unless admission already allocated its device.
			plugin.SetDevicePluginOptions(&kubeletdevicepluginv1beta1.DevicePluginOptions{PreStartRequired: true})
			defer plugin.Stop()
			framework.ExpectNoError(plugin.RegisterDevicePlugin(ctx, f.UniqueName, resourceName, []*kubeletdevicepluginv1beta1.Device{{
				ID:     deviceID,
				Health: kubeletdevicepluginv1beta1.Healthy,
			}}))

			ginkgo.By("waiting for the device resource to become allocatable")
			gomega.Eventually(ctx, func(ctx context.Context) int64 {
				node := getLocalNode(ctx, f)
				quantity := node.Status.Allocatable[v1.ResourceName(resourceName)]
				return quantity.Value()
			}, time.Minute, framework.Poll).Should(gomega.Equal(int64(1)))

			pod := makeBusyboxPod(resourceName, fmt.Sprintf("sleep %s", sleepIntervalForever))
			podResources := v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("10m"),
				v1.ResourceMemory: resource.MustParse("10Mi"),
			}
			pod.Spec.Resources = &v1.ResourceRequirements{Requests: podResources, Limits: podResources}
			podClient := e2epod.NewPodClient(f)
			pod = podClient.Create(ctx, pod)
			defer podClient.DeleteSync(ctx, pod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))
			gomega.Expect(plugin.WasCalled("PreStartContainer")).To(gomega.BeTrueBecause("the container must pass the pre-start allocation check"))

			ginkgo.By("verifying the committed device assignment using pod-resources")
			podResourcesResponse, err := getV1NodeDevices(ctx)
			framework.ExpectNoError(err)
			err, found := checkPodResourcesAssignment(podResourcesResponse, pod.Namespace, pod.Name, pod.Spec.Containers[0].Name, resourceName, []string{deviceID})
			framework.ExpectNoError(err)
			gomega.Expect(found).To(gomega.BeTrueBecause("expected pod %s/%s to have a committed device allocation", pod.Namespace, pod.Name))
		})
	})
})
