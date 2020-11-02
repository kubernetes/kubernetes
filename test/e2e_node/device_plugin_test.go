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

package e2enode

import (
	"context"
	"path/filepath"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"

	"regexp"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubeletpodresourcesv1alpha1 "k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	// sampleResourceName is the name of the example resource which is used in the e2e test
	sampleResourceName = "example.com/resource"
	// sampleDevicePluginDSYAML is the path of the daemonset template of the sample device plugin. // TODO: Parametrize it by making it a feature in TestFramework.
	sampleDevicePluginDSYAML = "test/e2e/testing-manifests/sample-device-plugin.yaml"
	// sampleDevicePluginName is the name of the device plugin pod
	sampleDevicePluginName = "sample-device-plugin"

	// fake resource name
	resourceName            = "example.com/resource"
	envVarNamePluginSockDir = "PLUGIN_SOCK_DIR"
)

var (
	appsScheme = runtime.NewScheme()
	appsCodecs = serializer.NewCodecFactory(appsScheme)
)

// Serial because the test restarts Kubelet
var _ = framework.KubeDescribe("Device Plugin [Feature:DevicePluginProbe][NodeFeature:DevicePluginProbe][Serial]", func() {
	f := framework.NewDefaultFramework("device-plugin-errors")
	testDevicePlugin(f, "/var/lib/kubelet/plugins_registry")
})

// numberOfSampleResources returns the number of resources advertised by a node.
func numberOfSampleResources(node *v1.Node) int64 {
	val, ok := node.Status.Capacity[sampleResourceName]

	if !ok {
		return 0
	}

	return val.Value()
}

// getSampleDevicePluginPod returns the Device Plugin pod for sample resources in e2e tests.
func getSampleDevicePluginPod() *v1.Pod {
	data, err := e2etestfiles.Read(sampleDevicePluginDSYAML)
	if err != nil {
		framework.Fail(err.Error())
	}

	ds := readDaemonSetV1OrDie(data)
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sampleDevicePluginName,
			Namespace: metav1.NamespaceSystem,
		},

		Spec: ds.Spec.Template.Spec,
	}

	return p
}

// readDaemonSetV1OrDie reads daemonset object from bytes. Panics on error.
func readDaemonSetV1OrDie(objBytes []byte) *appsv1.DaemonSet {
	appsv1.AddToScheme(appsScheme)
	requiredObj, err := runtime.Decode(appsCodecs.UniversalDecoder(appsv1.SchemeGroupVersion), objBytes)
	if err != nil {
		panic(err)
	}
	return requiredObj.(*appsv1.DaemonSet)
}

func testDevicePlugin(f *framework.Framework, pluginSockDir string) {
	pluginSockDir = filepath.Join(pluginSockDir) + "/"
	ginkgo.Context("DevicePlugin", func() {
		ginkgo.By("Enabling support for Kubelet Plugins Watcher")
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = map[string]bool{}
			}
			initialConfig.FeatureGates[string(features.KubeletPodResources)] = true
		})
		ginkgo.It("Verifies the Kubelet device plugin functionality.", func() {
			ginkgo.By("Wait for node is ready to start with")
			e2enode.WaitForNodeToBeReady(f.ClientSet, framework.TestContext.NodeName, 5*time.Minute)
			dp := getSampleDevicePluginPod()
			for i := range dp.Spec.Containers[0].Env {
				if dp.Spec.Containers[0].Env[i].Name == envVarNamePluginSockDir {
					dp.Spec.Containers[0].Env[i].Value = pluginSockDir
				}
			}
			framework.Logf("env %v", dp.Spec.Containers[0].Env)
			dp.Spec.NodeName = framework.TestContext.NodeName
			ginkgo.By("Create sample device plugin pod")
			devicePluginPod, err := f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Create(context.TODO(), dp, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Waiting for devices to become available on the local node")
			gomega.Eventually(func() bool {
				return numberOfSampleResources(getLocalNode(f)) > 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())
			framework.Logf("Successfully created device plugin pod")

			ginkgo.By("Waiting for the resource exported by the sample device plugin to become available on the local node")
			// TODO(vikasc): Instead of hard-coding number of devices, provide number of devices in the sample-device-plugin using configmap
			// and then use the same here
			devsLen := int64(2)
			gomega.Eventually(func() bool {
				node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfDevicesCapacity(node, resourceName) == devsLen &&
					numberOfDevicesAllocatable(node, resourceName) == devsLen
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrue())

			ginkgo.By("Creating one pod on node with at least one fake-device")
			podRECMD := "devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs"
			pod1 := f.PodClient().CreateSync(makeBusyboxPod(resourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devID1 := parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			gomega.Expect(devID1).To(gomega.Not(gomega.Equal("")))

			podResources, err := getNodeDevices()
			var resourcesForOurPod *kubeletpodresourcesv1alpha1.PodResources
			framework.Logf("pod resources %v", podResources)
			framework.ExpectNoError(err)
			framework.ExpectEqual(len(podResources.PodResources), 2)
			for _, res := range podResources.GetPodResources() {
				if res.Name == pod1.Name {
					resourcesForOurPod = res
				}
			}
			framework.Logf("resourcesForOurPod %v", resourcesForOurPod)
			gomega.Expect(resourcesForOurPod).NotTo(gomega.BeNil())
			framework.ExpectEqual(resourcesForOurPod.Name, pod1.Name)
			framework.ExpectEqual(resourcesForOurPod.Namespace, pod1.Namespace)
			framework.ExpectEqual(len(resourcesForOurPod.Containers), 1)
			framework.ExpectEqual(resourcesForOurPod.Containers[0].Name, pod1.Spec.Containers[0].Name)
			framework.ExpectEqual(len(resourcesForOurPod.Containers[0].Devices), 1)
			framework.ExpectEqual(resourcesForOurPod.Containers[0].Devices[0].ResourceName, resourceName)
			framework.ExpectEqual(len(resourcesForOurPod.Containers[0].Devices[0].DeviceIds), 1)

			pod1, err = f.PodClient().Get(context.TODO(), pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ensurePodContainerRestart(f, pod1.Name, pod1.Name)

			ginkgo.By("Confirming that device assignment persists even after container restart")
			devIDAfterRestart := parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectEqual(devIDAfterRestart, devID1)

			restartTime := time.Now()
			ginkgo.By("Restarting Kubelet")
			restartKubelet()

			// We need to wait for node to be ready before re-registering stub device plugin.
			// Otherwise, Kubelet DeviceManager may remove the re-registered sockets after it starts.
			ginkgo.By("Wait for node is ready")
			gomega.Eventually(func() bool {
				node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				for _, cond := range node.Status.Conditions {
					if cond.Type == v1.NodeReady && cond.Status == v1.ConditionTrue && cond.LastHeartbeatTime.After(restartTime) {
						return true
					}
				}
				return false
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())

			ginkgo.By("Re-Register resources and deleting the pods and waiting for container removal")
			getOptions := metav1.GetOptions{}
			gp := int64(0)
			deleteOptions := metav1.DeleteOptions{
				GracePeriodSeconds: &gp,
			}
			err = f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Delete(context.TODO(), dp.Name, deleteOptions)
			framework.ExpectNoError(err)
			waitForContainerRemoval(devicePluginPod.Spec.Containers[0].Name, devicePluginPod.Name, devicePluginPod.Namespace)
			_, err = f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Get(context.TODO(), dp.Name, getOptions)
			framework.Logf("Trying to get dp pod after deletion. err must be non-nil. err: %v", err)
			framework.ExpectError(err)

			devicePluginPod, err = f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Create(context.TODO(), dp, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ensurePodContainerRestart(f, pod1.Name, pod1.Name)
			ginkgo.By("Confirming that after a kubelet restart, fake-device assignement is kept")
			devIDRestart1 := parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart1, devID1)

			ginkgo.By("Waiting for resource to become available on the local node after re-registration")
			gomega.Eventually(func() bool {
				node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfDevicesCapacity(node, resourceName) == devsLen &&
					numberOfDevicesAllocatable(node, resourceName) == devsLen
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrue())

			ginkgo.By("Creating another pod")
			pod2 := f.PodClient().CreateSync(makeBusyboxPod(resourceName, podRECMD))

			ginkgo.By("Checking that pod got a different fake device")
			devID2 := parseLog(f, pod2.Name, pod2.Name, deviceIDRE)

			gomega.Expect(devID1).To(gomega.Not(gomega.Equal(devID2)))

			ginkgo.By("By deleting the pods and waiting for container removal")
			err = f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Delete(context.TODO(), dp.Name, deleteOptions)
			framework.ExpectNoError(err)
			waitForContainerRemoval(devicePluginPod.Spec.Containers[0].Name, devicePluginPod.Name, devicePluginPod.Namespace)

			ginkgo.By("Waiting for stub device plugin to become unhealthy on the local node")
			gomega.Eventually(func() int64 {
				node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfDevicesAllocatable(node, resourceName)
			}, 30*time.Second, framework.Poll).Should(gomega.Equal(int64(0)))

			ginkgo.By("Checking that scheduled pods can continue to run even after we delete device plugin.")
			ensurePodContainerRestart(f, pod1.Name, pod1.Name)
			devIDRestart1 = parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart1, devID1)

			ensurePodContainerRestart(f, pod2.Name, pod2.Name)
			devIDRestart2 := parseLog(f, pod2.Name, pod2.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart2, devID2)

			ginkgo.By("Re-register resources")
			devicePluginPod, err = f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Create(context.TODO(), dp, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Waiting for the resource exported by the stub device plugin to become healthy on the local node")
			gomega.Eventually(func() int64 {
				node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfDevicesAllocatable(node, resourceName)
			}, 30*time.Second, framework.Poll).Should(gomega.Equal(devsLen))

			ginkgo.By("by deleting the pods and waiting for container removal")
			err = f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Delete(context.TODO(), dp.Name, deleteOptions)
			framework.ExpectNoError(err)
			waitForContainerRemoval(devicePluginPod.Spec.Containers[0].Name, devicePluginPod.Name, devicePluginPod.Namespace)

			ginkgo.By("Waiting for stub device plugin to become unavailable on the local node")
			gomega.Eventually(func() bool {
				node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfDevicesCapacity(node, resourceName) <= 0
			}, 10*time.Minute, framework.Poll).Should(gomega.BeTrue())

			// Cleanup
			f.PodClient().DeleteSync(pod1.Name, metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
			f.PodClient().DeleteSync(pod2.Name, metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
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
	p, err := f.PodClient().Get(context.TODO(), podName, metav1.GetOptions{})
	if err != nil || len(p.Status.ContainerStatuses) < 1 {
		framework.Failf("ensurePodContainerRestart failed for pod %q: %v", podName, err)
	}
	initialCount = p.Status.ContainerStatuses[0].RestartCount
	gomega.Eventually(func() bool {
		p, err = f.PodClient().Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil || len(p.Status.ContainerStatuses) < 1 {
			return false
		}
		currentCount = p.Status.ContainerStatuses[0].RestartCount
		framework.Logf("initial %v, current %v", initialCount, currentCount)
		return currentCount > initialCount
	}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())
}

// parseLog returns the matching string for the specified regular expression parsed from the container logs.
func parseLog(f *framework.Framework, podName string, contName string, re string) string {
	logs, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, podName, contName)
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
