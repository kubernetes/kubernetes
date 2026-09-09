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
	"path/filepath"
	"time"

	"k8s.io/kubernetes/test/e2e_node/testdeviceplugin"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletdevicepluginv1beta1 "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	kubeletpodresourcesv1alpha1 "k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

// Serial because the test restarts Kubelet
var _ = SIGDescribe("Device Plugin Multiple", framework.WithSerial(), feature.DevicePlugin, func() {
	f := framework.NewDefaultFramework("device-plugin-errors")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	testDevicePluginMultiple(f, kubeletdevicepluginv1beta1.DevicePluginPath)
})

func testDevicePluginMultiple(f *framework.Framework, pluginSockDir string) {
	pluginSockDir = filepath.Clean(pluginSockDir) + "/"

	f.Context("DevicePlugin", f.WithSerial(), f.WithDisruptive(), func() {
		var devicePluginPod, devicePluginPod2 *v1.Pod
		var v1alphaPodResources *kubeletpodresourcesv1alpha1.ListPodResourcesResponse
		var v1PodResources *kubeletpodresourcesv1.ListPodResourcesResponse
		var err error

		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("Wait for node to be ready")
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				nodes, err := e2enode.TotalReady(ctx, f.ClientSet)
				framework.ExpectNoError(err)
				return nodes == 1
			}, time.Minute, time.Second).Should(gomega.BeTrueBecause("expected node to be ready"))

			// Before we run the device plugin test, we need to ensure
			// that the cluster is in a clean state and there are no
			// pods running on this node.
			// This is done in a gomega.Eventually with retries since a prior test in a different test suite could've run and the deletion of it's resources may still be in progress.
			// xref: https://issue.k8s.io/115381
			gomega.Eventually(ctx, func(ctx context.Context) error {
				v1alphaPodResources, err = getV1alpha1NodeDevices(ctx)
				if err != nil {
					return fmt.Errorf("failed to get node local podresources by accessing the (v1alpha) podresources API endpoint: %w", err)
				}

				v1PodResources, err = getV1NodeDevices(ctx)
				if err != nil {
					return fmt.Errorf("failed to get node local podresources by accessing the (v1) podresources API endpoint: %w", err)
				}

				if len(v1alphaPodResources.PodResources) > 0 {
					return fmt.Errorf("expected v1alpha pod resources to be empty, but got non-empty resources: %+v", v1alphaPodResources.PodResources)
				}

				if len(v1PodResources.PodResources) > 0 {
					return fmt.Errorf("expected v1 pod resources to be empty, but got non-empty resources: %+v", v1PodResources.PodResources)
				}
				return nil
			}, f.Timeouts.PodDelete, f.Timeouts.Poll).Should(gomega.Succeed())

			ginkgo.By("Scheduling a sample device plugin pod")
			dp := getSampleDevicePluginPod(pluginSockDir, "dp1")
			devicePluginPod = e2epod.NewPodClient(f).CreateSync(ctx, dp)

			ginkgo.By("Waiting for devices to become available on the local node")
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready && e2enode.CountSampleDeviceCapacity(node) > 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrueBecause("expected devices to be available on local node"))
			framework.Logf("Successfully created device plugin pod")

			ginkgo.By(fmt.Sprintf("Waiting for the resource exported by the sample device plugin to become available on the local node (instances: %d)", e2enode.SampleDevsAmount))
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready &&
					e2enode.CountSampleDeviceCapacity(node) == e2enode.SampleDevsAmount &&
					e2enode.CountSampleDeviceAllocatable(node) == e2enode.SampleDevsAmount
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrueBecause("expected resource to be available on local node"))
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("Deleting the device plugin pods")
			if devicePluginPod != nil {
				e2epod.NewPodClient(f).DeleteSync(ctx, devicePluginPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			}
			if devicePluginPod2 != nil {
				e2epod.NewPodClient(f).DeleteSync(ctx, devicePluginPod2.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			}

			ginkgo.By("Deleting any Pods created by the test")
			l, err := e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, p := range l.Items {
				if p.Namespace != f.Namespace.Name {
					continue
				}
				framework.Logf("Deleting pod: %s", p.Name)
				e2epod.NewPodClient(f).DeleteSync(ctx, p.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			}

			restartKubelet(ctx, true)

			ginkgo.By("Waiting for devices to become unavailable on the local node")
			gomega.Eventually(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready && e2enode.CountSampleDeviceCapacity(node) <= 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrueBecause("expected devices to be unavailable on local node"))

			ginkgo.By("devices now unavailable on the local node")
		})

		// Pod1 scheduled with DP1, DP2 appears, Pod2 scheduled successfully with both DPs running, DP1 is deleted,
		// Pod3 is scheduled successfully after DP1 is removed.
		ginkgo.It("Device Plugin Multiple: Basic rollout of device plugins with zero downtime", func(ctx context.Context) {
			ginkgo.By("Scheduling Pod1 with DP1 successfully")
			podRECMD := fmt.Sprintf("devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep %s", sleepIntervalForever)
			pod1 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(e2enode.SampleDeviceResourceName, podRECMD))
			pod1, err := e2epod.NewPodClient(f).Get(ctx, pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod1.Status.Phase).To(gomega.Equal(v1.PodRunning))

			ginkgo.By("Creating DP2")
			dp2 := getSampleDevicePluginPodMultiple(pluginSockDir, "dp2")
			devicePluginPod2 = e2epod.NewPodClient(f).CreateSync(ctx, dp2)

			ginkgo.By("Waiting for DP2 to register for 30s, and number of devices to remain unchanged")
			gomega.Consistently(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready && e2enode.CountSampleDeviceCapacity(node) == e2enode.SampleDevsAmount && e2enode.CountSampleDeviceAllocatable(node) == e2enode.SampleDevsAmount
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrueBecause("expected devices after DP2 appears"))

			ginkgo.By("Verifying DP2 is running")
			dp2Pod, err := e2epod.NewPodClient(f).Get(ctx, devicePluginPod2.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(dp2Pod.Status.Phase).To(gomega.Equal(v1.PodRunning))

			ginkgo.By("Scheduling Pod2 while both are running")
			pod2 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(e2enode.SampleDeviceResourceName, podRECMD))

			ginkgo.By("Verifying Pod2 is running")
			pod2, err = e2epod.NewPodClient(f).Get(ctx, pod2.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod2.Status.Phase).To(gomega.Equal(v1.PodRunning))

			ginkgo.By("Deleting DP1")
			e2epod.NewPodClient(f).DeleteSync(ctx, devicePluginPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			waitForContainerRemoval(ctx, devicePluginPod.Spec.Containers[0].Name, devicePluginPod.Name, devicePluginPod.Namespace)

			ginkgo.By("Waiting for DP1 to unregister, and number of devices to remain unchanged")
			gomega.Consistently(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready && e2enode.CountSampleDeviceCapacity(node) == e2enode.SampleDevsAmount && e2enode.CountSampleDeviceAllocatable(node) == e2enode.SampleDevsAmount
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrueBecause("expected devices to decrease after DP1 disappears"))

			ginkgo.By("Deleting Pod1 to free up resources")
			e2epod.NewPodClient(f).DeleteSync(ctx, pod1.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

			ginkgo.By("Scheduling Pod3 successfully after DP1 disappeared")
			pod3 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(e2enode.SampleDeviceResourceName, podRECMD))

			ginkgo.By("Verifying Pod3 is running")
			pod3, err = e2epod.NewPodClient(f).Get(ctx, pod3.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod3.Status.Phase).To(gomega.Equal(v1.PodRunning))
		})

		// Pod1 scheduled with DP1, DP2 appears after 30 seconds, Pod2 scheduled successfully before DP2 connected, Pod3 scheduled successfully after DP2 connected.
		ginkgo.It("Device Plugin Multiple: DP2 takes long time to start working", func(ctx context.Context) {
			var err error
			ginkgo.By("Scheduling Pod1 with DP1 successfully")
			podRECMD := fmt.Sprintf("devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep %s", sleepIntervalForever)
			pod1 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(e2enode.SampleDeviceResourceName, podRECMD))
			pod1, err = e2epod.NewPodClient(f).Get(ctx, pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod1.Status.Phase).To(gomega.Equal(v1.PodRunning))

			ginkgo.By("Creating DP2 but struggle to register for 30s")
			expectedErr := fmt.Errorf("GetDevicePluginOptions failed")

			plugin2 := testdeviceplugin.NewDevicePlugin(func(name string) error {
				if name == "GetDevicePluginOptions" {
					return expectedErr
				}
				return nil
			})
			err = plugin2.RegisterDevicePlugin(ctx, f.UniqueName, e2enode.SampleDeviceResourceName, []*kubeletdevicepluginv1beta1.Device{{ID: "testdevice", Health: kubeletdevicepluginv1beta1.Healthy}})
			defer plugin2.Stop()
			gomega.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("failed to get device plugin options")))
			gomega.Expect(err).To(gomega.MatchError(gomega.ContainSubstring(expectedErr.Error())))

			ginkgo.By("Scheduling Pod2 successfully")
			pod2 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(e2enode.SampleDeviceResourceName, podRECMD))

			ginkgo.By("Verifying Pod2 is running")
			pod2, err = e2epod.NewPodClient(f).Get(ctx, pod2.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod2.Status.Phase).To(gomega.Equal(v1.PodRunning))

			// Device Plugin's 'take over' implementation will create a new endpoint on every registration attempt.
			// DP3 represents re-registration of the second DP that failed to register first time.
			ginkgo.By("Scheduling DP3")
			dp3 := getSampleDevicePluginPodMultiple(pluginSockDir, "dp3")
			devicePluginPod2 = e2epod.NewPodClient(f).CreateSync(ctx, dp3)

			ginkgo.By("Waiting for DP3 to register for 30s, and number of devices to remain unchanged")
			gomega.Consistently(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready && e2enode.CountSampleDeviceCapacity(node) == e2enode.SampleDevsAmount && e2enode.CountSampleDeviceAllocatable(node) == e2enode.SampleDevsAmount
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrueBecause("expected devices after DP2 appears"))

			ginkgo.By("Verifying DP3 is running")
			dp2Pod, err := e2epod.NewPodClient(f).Get(ctx, devicePluginPod2.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(dp2Pod.Status.Phase).To(gomega.Equal(v1.PodRunning))

			ginkgo.By("Deleting DP1")
			e2epod.NewPodClient(f).DeleteSync(ctx, devicePluginPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			waitForContainerRemoval(ctx, devicePluginPod.Spec.Containers[0].Name, devicePluginPod.Name, devicePluginPod.Namespace)

			ginkgo.By("Waiting for DP1 to unregister, and number of devices to remain unchanged")
			gomega.Consistently(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready && e2enode.CountSampleDeviceCapacity(node) == e2enode.SampleDevsAmount && e2enode.CountSampleDeviceAllocatable(node) == e2enode.SampleDevsAmount
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrueBecause("expected devices after DP1 disappears"))

			ginkgo.By("Deleting Pod1 to free up resources")
			e2epod.NewPodClient(f).DeleteSync(ctx, pod1.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

			ginkgo.By("Scheduling Pod3 after DP3 registration succeeds")
			pod3 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(e2enode.SampleDeviceResourceName, podRECMD))

			ginkgo.By("Verifying Pod3 is running")
			pod3, err = e2epod.NewPodClient(f).Get(ctx, pod3.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod3.Status.Phase).To(gomega.Equal(v1.PodRunning))
		})

		// Pod1 scheduled with DP1, DP2 appears, Pod2 scheduled successfully with both DPs running,
		// DP2 is deleted, Pod3 is scheduled successfully after DP2 is removed.
		ginkgo.It("Device Plugin Multiple: DP2 changed its mind", func(ctx context.Context) {
			var err error
			ginkgo.By("Scheduling Pod1 with DP1 successfully")
			podRECMD := fmt.Sprintf("devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep %s", sleepIntervalForever)
			pod1 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(e2enode.SampleDeviceResourceName, podRECMD))
			pod1, err = e2epod.NewPodClient(f).Get(ctx, pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod1.Status.Phase).To(gomega.Equal(v1.PodRunning))

			ginkgo.By("Scheduling DP2")
			dp2 := getSampleDevicePluginPodMultiple(pluginSockDir, "dp2")
			devicePluginPod2 = e2epod.NewPodClient(f).CreateSync(ctx, dp2)

			ginkgo.By("Waiting for DP2 to register for 30s, and number of devices to remain unchanged")
			gomega.Consistently(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready && e2enode.CountSampleDeviceCapacity(node) == e2enode.SampleDevsAmount && e2enode.CountSampleDeviceAllocatable(node) == e2enode.SampleDevsAmount
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrueBecause("expected devices after DP2 appears"))

			ginkgo.By("Verifying DP2 is running")
			dp2Pod, err := e2epod.NewPodClient(f).Get(ctx, devicePluginPod2.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(dp2Pod.Status.Phase).To(gomega.Equal(v1.PodRunning))

			ginkgo.By("Scheduling Pod2 while both are running")
			pod2 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(e2enode.SampleDeviceResourceName, podRECMD))

			ginkgo.By("Verifying Pod2 is running")
			pod2, err = e2epod.NewPodClient(f).Get(ctx, pod2.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod2.Status.Phase).To(gomega.Equal(v1.PodRunning))

			ginkgo.By("Deleting DP2")
			e2epod.NewPodClient(f).DeleteSync(ctx, devicePluginPod2.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			waitForContainerRemoval(ctx, devicePluginPod2.Spec.Containers[0].Name, devicePluginPod2.Name, devicePluginPod2.Namespace)

			ginkgo.By("Waiting for DP2 to unregister, and number of devices to remain unchanged")
			gomega.Consistently(ctx, func(ctx context.Context) bool {
				node, ready := getLocalTestNode(ctx, f)
				return ready && e2enode.CountSampleDeviceCapacity(node) == e2enode.SampleDevsAmount && e2enode.CountSampleDeviceAllocatable(node) == e2enode.SampleDevsAmount
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrueBecause("expected devices after DP2 disappears"))

			ginkgo.By("Deleting Pod1 to free up resources")
			e2epod.NewPodClient(f).DeleteSync(ctx, pod1.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

			ginkgo.By("Scheduling Pod3 successfully after DP2 disappeared")
			pod3 := e2epod.NewPodClient(f).CreateSync(ctx, makeBusyboxPod(e2enode.SampleDeviceResourceName, podRECMD))

			ginkgo.By("Verifying Pod3 is running")
			pod3, err = e2epod.NewPodClient(f).Get(ctx, pod3.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod3.Status.Phase).To(gomega.Equal(v1.PodRunning))
		})
	})
}

// This is a quick fix to allow duplicated device plugin otherwise, because CDI is hardcoded,
// we need another device plugin yaml.
// getSampleDevicePluginPodMultiple returns the Sample Device Plugin pod with CDI disabled.
func getSampleDevicePluginPodMultiple(pluginSockDir string, version string) *v1.Pod {
	dp := getSampleDevicePluginPod(pluginSockDir, version)
	for i := range dp.Spec.Containers[0].Env {
		if dp.Spec.Containers[0].Env[i].Name == "CDI_ENABLED" {
			dp.Spec.Containers[0].Env[i].Value = ""
		}
	}
	return dp
}
