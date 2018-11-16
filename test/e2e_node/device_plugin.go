/*
Copyright 2017 The Kubernetes Authors.

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

package e2e_node

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"regexp"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1beta1"
	dm "k8s.io/kubernetes/pkg/kubelet/cm/devicemanager"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// fake resource name
	resourceName                 = "fake.com/resource"
	resourceNameWithProbeSupport = "fake.com/resource2"
)

// Serial because the test restarts Kubelet
var _ = framework.KubeDescribe("Device Plugin [Feature:DevicePlugin][NodeFeature:DevicePlugin][Serial]", func() {
	f := framework.NewDefaultFramework("device-plugin-errors")
	testDevicePlugin(f, false, pluginapi.DevicePluginPath)
})

var _ = framework.KubeDescribe("Device Plugin [Feature:DevicePluginProbe][NodeFeature:DevicePluginProbe][Serial]", func() {
	f := framework.NewDefaultFramework("device-plugin-errors")
	testDevicePlugin(f, true, "/var/lib/kubelet/plugins_registry")
})

func testDevicePlugin(f *framework.Framework, enablePluginWatcher bool, pluginSockDir string) {
	Context("DevicePlugin", func() {
		By("Enabling support for Kubelet Plugins Watcher")
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = map[string]bool{}
			}
			initialConfig.FeatureGates[string(features.KubeletPluginsWatcher)] = enablePluginWatcher
			initialConfig.FeatureGates[string(features.KubeletPodResources)] = true
		})
		It("Verifies the Kubelet device plugin functionality.", func() {
			By("Start stub device plugin")
			// fake devices for e2e test
			devs := []*pluginapi.Device{
				{ID: "Dev-1", Health: pluginapi.Healthy},
				{ID: "Dev-2", Health: pluginapi.Healthy},
			}

			socketPath := pluginSockDir + "dp." + fmt.Sprintf("%d", time.Now().Unix())
			framework.Logf("socketPath %v", socketPath)

			dp1 := dm.NewDevicePluginStub(devs, socketPath, resourceName, false)
			dp1.SetAllocFunc(stubAllocFunc)
			err := dp1.Start()
			framework.ExpectNoError(err)

			By("Register resources")
			err = dp1.Register(pluginapi.KubeletSocket, resourceName, pluginSockDir)
			framework.ExpectNoError(err)

			By("Waiting for the resource exported by the stub device plugin to become available on the local node")
			devsLen := int64(len(devs))
			Eventually(func() bool {
				node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfDevicesCapacity(node, resourceName) == devsLen &&
					numberOfDevicesAllocatable(node, resourceName) == devsLen
			}, 30*time.Second, framework.Poll).Should(BeTrue())

			By("Creating one pod on node with at least one fake-device")
			podRECMD := "devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs"
			pod1 := f.PodClient().CreateSync(makeBusyboxPod(resourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devId1 := parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			Expect(devId1).To(Not(Equal("")))

			podResources, err := getNodeDevices()
			Expect(err).To(BeNil())
			Expect(len(podResources.PodResources)).To(Equal(1))
			Expect(podResources.PodResources[0].Name).To(Equal(pod1.Name))
			Expect(podResources.PodResources[0].Namespace).To(Equal(pod1.Namespace))
			Expect(len(podResources.PodResources[0].Containers)).To(Equal(1))
			Expect(podResources.PodResources[0].Containers[0].Name).To(Equal(pod1.Spec.Containers[0].Name))
			Expect(len(podResources.PodResources[0].Containers[0].Devices)).To(Equal(1))
			Expect(podResources.PodResources[0].Containers[0].Devices[0].ResourceName).To(Equal(resourceName))
			Expect(len(podResources.PodResources[0].Containers[0].Devices[0].DeviceIds)).To(Equal(1))

			pod1, err = f.PodClient().Get(pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ensurePodContainerRestart(f, pod1.Name, pod1.Name)

			By("Confirming that device assignment persists even after container restart")
			devIdAfterRestart := parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			Expect(devIdAfterRestart).To(Equal(devId1))

			restartTime := time.Now()
			By("Restarting Kubelet")
			restartKubelet()

			// We need to wait for node to be ready before re-registering stub device plugin.
			// Otherwise, Kubelet DeviceManager may remove the re-registered sockets after it starts.
			By("Wait for node is ready")
			Eventually(func() bool {
				node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				for _, cond := range node.Status.Conditions {
					if cond.Type == v1.NodeReady && cond.Status == v1.ConditionTrue && cond.LastHeartbeatTime.After(restartTime) {
						return true
					}
				}
				return false
			}, 5*time.Minute, framework.Poll).Should(BeTrue())

			By("Re-Register resources")
			dp1 = dm.NewDevicePluginStub(devs, socketPath, resourceName, false)
			dp1.SetAllocFunc(stubAllocFunc)
			err = dp1.Start()
			framework.ExpectNoError(err)

			err = dp1.Register(pluginapi.KubeletSocket, resourceName, pluginSockDir)
			framework.ExpectNoError(err)

			ensurePodContainerRestart(f, pod1.Name, pod1.Name)
			By("Confirming that after a kubelet restart, fake-device assignement is kept")
			devIdRestart1 := parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			Expect(devIdRestart1).To(Equal(devId1))

			By("Waiting for resource to become available on the local node after re-registration")
			Eventually(func() bool {
				node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfDevicesCapacity(node, resourceName) == devsLen &&
					numberOfDevicesAllocatable(node, resourceName) == devsLen
			}, 30*time.Second, framework.Poll).Should(BeTrue())

			By("Creating another pod")
			pod2 := f.PodClient().CreateSync(makeBusyboxPod(resourceName, podRECMD))

			By("Checking that pod got a different fake device")
			devId2 := parseLog(f, pod2.Name, pod2.Name, deviceIDRE)

			Expect(devId1).To(Not(Equal(devId2)))

			By("Deleting device plugin.")
			err = dp1.Stop()
			framework.ExpectNoError(err)

			By("Waiting for stub device plugin to become unhealthy on the local node")
			Eventually(func() int64 {
				node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfDevicesAllocatable(node, resourceName)
			}, 30*time.Second, framework.Poll).Should(Equal(int64(0)))

			By("Checking that scheduled pods can continue to run even after we delete device plugin.")
			ensurePodContainerRestart(f, pod1.Name, pod1.Name)
			devIdRestart1 = parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			Expect(devIdRestart1).To(Equal(devId1))

			ensurePodContainerRestart(f, pod2.Name, pod2.Name)
			devIdRestart2 := parseLog(f, pod2.Name, pod2.Name, deviceIDRE)
			Expect(devIdRestart2).To(Equal(devId2))

			By("Re-register resources")
			dp1 = dm.NewDevicePluginStub(devs, socketPath, resourceName, false)
			dp1.SetAllocFunc(stubAllocFunc)
			err = dp1.Start()
			framework.ExpectNoError(err)

			err = dp1.Register(pluginapi.KubeletSocket, resourceName, pluginSockDir)
			framework.ExpectNoError(err)

			By("Waiting for the resource exported by the stub device plugin to become healthy on the local node")
			Eventually(func() int64 {
				node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfDevicesAllocatable(node, resourceName)
			}, 30*time.Second, framework.Poll).Should(Equal(devsLen))

			By("Deleting device plugin again.")
			err = dp1.Stop()
			framework.ExpectNoError(err)

			By("Waiting for stub device plugin to become unavailable on the local node")
			Eventually(func() bool {
				node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfDevicesCapacity(node, resourceName) <= 0
			}, 10*time.Minute, framework.Poll).Should(BeTrue())

			// Cleanup
			f.PodClient().DeleteSync(pod1.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
			f.PodClient().DeleteSync(pod2.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
		})
	})
}

// makeBusyboxPod returns a simple Pod spec with a busybox container
// that requests resourceName and runs the specified command.
func makeBusyboxPod(resourceName, cmd string) *v1.Pod {
	podName := "device-plugin-test-" + string(uuid.NewUUID())
	rl := v1.ResourceList{v1.ResourceName(resourceName): *resource.NewQuantity(1, resource.DecimalSI)}

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyAlways,
			Containers: []v1.Container{{
				Image: busyboxImage,
				Name:  podName,
				// Runs the specified command in the test pod.
				Command: []string{"sh", "-c", cmd},
				Resources: v1.ResourceRequirements{
					Limits:   rl,
					Requests: rl,
				},
			}},
		},
	}
}

// ensurePodContainerRestart confirms that pod container has restarted at least once
func ensurePodContainerRestart(f *framework.Framework, podName string, contName string) {
	var initialCount int32
	var currentCount int32
	p, err := f.PodClient().Get(podName, metav1.GetOptions{})
	if err != nil || len(p.Status.ContainerStatuses) < 1 {
		framework.Failf("ensurePodContainerRestart failed for pod %q: %v", podName, err)
	}
	initialCount = p.Status.ContainerStatuses[0].RestartCount
	Eventually(func() bool {
		p, err = f.PodClient().Get(podName, metav1.GetOptions{})
		if err != nil || len(p.Status.ContainerStatuses) < 1 {
			return false
		}
		currentCount = p.Status.ContainerStatuses[0].RestartCount
		framework.Logf("initial %v, current %v", initialCount, currentCount)
		return currentCount > initialCount
	}, 5*time.Minute, framework.Poll).Should(BeTrue())
}

// parseLog returns the matching string for the specified regular expression parsed from the container logs.
func parseLog(f *framework.Framework, podName string, contName string, re string) string {
	logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, podName, contName)
	if err != nil {
		framework.Failf("GetPodLogs for pod %q failed: %v", podName, err)
	}

	framework.Logf("got pod logs: %v", logs)
	regex := regexp.MustCompile(re)
	matches := regex.FindStringSubmatch(logs)
	if len(matches) < 2 {
		return ""
	}

	return matches[1]
}

// numberOfDevicesCapacity returns the number of devices of resourceName advertised by a node capacity
func numberOfDevicesCapacity(node *v1.Node, resourceName string) int64 {
	val, ok := node.Status.Capacity[v1.ResourceName(resourceName)]
	if !ok {
		return 0
	}

	return val.Value()
}

// numberOfDevicesAllocatable returns the number of devices of resourceName advertised by a node allocatable
func numberOfDevicesAllocatable(node *v1.Node, resourceName string) int64 {
	val, ok := node.Status.Allocatable[v1.ResourceName(resourceName)]
	if !ok {
		return 0
	}

	return val.Value()
}

// stubAllocFunc will pass to stub device plugin
func stubAllocFunc(r *pluginapi.AllocateRequest, devs map[string]pluginapi.Device) (*pluginapi.AllocateResponse, error) {
	var responses pluginapi.AllocateResponse
	for _, req := range r.ContainerRequests {
		response := &pluginapi.ContainerAllocateResponse{}
		for _, requestID := range req.DevicesIDs {
			dev, ok := devs[requestID]
			if !ok {
				return nil, fmt.Errorf("invalid allocation request with non-existing device %s", requestID)
			}

			if dev.Health != pluginapi.Healthy {
				return nil, fmt.Errorf("invalid allocation request with unhealthy device: %s", requestID)
			}

			// create fake device file
			fpath := filepath.Join("/tmp", dev.ID)

			// clean first
			os.RemoveAll(fpath)
			f, err := os.Create(fpath)
			if err != nil && !os.IsExist(err) {
				return nil, fmt.Errorf("failed to create fake device file: %s", err)
			}

			f.Close()

			response.Mounts = append(response.Mounts, &pluginapi.Mount{
				ContainerPath: fpath,
				HostPath:      fpath,
			})
		}
		responses.ContainerResponses = append(responses.ContainerResponses, response)
	}

	return &responses, nil
}
