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
	"regexp"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	kubeletdevicepluginv1beta1 "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	kubeletpodresourcesv1alpha1 "k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

const (
	// sampleResourceName is the name of the example resource which is used in the e2e test
	sampleResourceName = "example.com/resource"
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
var _ = SIGDescribe("Device Plugin [Feature:DevicePluginProbe][NodeFeature:DevicePluginProbe][Serial]", func() {
	f := framework.NewDefaultFramework("device-plugin-errors")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	testDevicePlugin(f, kubeletdevicepluginv1beta1.DevicePluginPath)
})

// numberOfSampleResources returns the number of resources advertised by a node.
func numberOfSampleResources(node *v1.Node) int64 {
	val, ok := node.Status.Capacity[sampleResourceName]

	if !ok {
		return 0
	}

	return val.Value()
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
	ginkgo.Context("DevicePlugin [Serial] [Disruptive]", func() {
		// TODO(vikasc): Instead of hard-coding number of devices, provide number of devices in the sample-device-plugin using configmap
		// and then use the same here
		devsLen := int64(2)
		var devicePluginPod, dptemplate *v1.Pod

		ginkgo.BeforeEach(func() {
			ginkgo.By("Wait for node to be ready")
			gomega.Eventually(func() bool {
				nodes, err := e2enode.TotalReady(f.ClientSet)
				framework.ExpectNoError(err)
				return nodes == 1
			}, time.Minute, time.Second).Should(gomega.BeTrue())

			ginkgo.By("Scheduling a sample device plugin pod")
			data, err := e2etestfiles.Read(SampleDevicePluginDSYAML)
			if err != nil {
				framework.Fail(err.Error())
			}
			ds := readDaemonSetV1OrDie(data)

			dp := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: sampleDevicePluginName,
				},
				Spec: ds.Spec.Template.Spec,
			}

			for i := range dp.Spec.Containers[0].Env {
				if dp.Spec.Containers[0].Env[i].Name == envVarNamePluginSockDir {
					dp.Spec.Containers[0].Env[i].Value = pluginSockDir
				}
			}
			dptemplate = dp.DeepCopy()
			devicePluginPod = f.PodClient().CreateSync(dp)

			ginkgo.By("Waiting for devices to become available on the local node")
			gomega.Eventually(func() bool {
				node, ready := getLocalTestNode(f)
				return ready && numberOfSampleResources(node) > 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())
			framework.Logf("Successfully created device plugin pod")

			ginkgo.By("Waiting for the resource exported by the sample device plugin to become available on the local node")
			gomega.Eventually(func() bool {
				node, ready := getLocalTestNode(f)
				return ready &&
					numberOfDevicesCapacity(node, resourceName) == devsLen &&
					numberOfDevicesAllocatable(node, resourceName) == devsLen
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrue())
		})

		ginkgo.AfterEach(func() {
			ginkgo.By("Deleting the device plugin pod")
			f.PodClient().DeleteSync(devicePluginPod.Name, metav1.DeleteOptions{}, time.Minute)

			ginkgo.By("Deleting any Pods created by the test")
			l, err := f.PodClient().List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, p := range l.Items {
				if p.Namespace != f.Namespace.Name {
					continue
				}

				framework.Logf("Deleting pod: %s", p.Name)
				f.PodClient().DeleteSync(p.Name, metav1.DeleteOptions{}, 2*time.Minute)
			}

			restartKubelet(true)

			ginkgo.By("Waiting for devices to become unavailable on the local node")
			gomega.Eventually(func() bool {
				node, ready := getLocalTestNode(f)
				return ready && numberOfSampleResources(node) <= 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())

			ginkgo.By("devices now unavailable on the local node")
		})

		ginkgo.It("Can schedule a pod that requires a device", func() {
			podRECMD := "devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep 60"
			pod1 := f.PodClient().CreateSync(makeBusyboxPod(resourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devID1 := parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			gomega.Expect(devID1).To(gomega.Not(gomega.Equal("")))

			v1alphaPodResources, err := getV1alpha1NodeDevices()
			framework.ExpectNoError(err)

			v1PodResources, err := getV1NodeDevices()
			framework.ExpectNoError(err)

			framework.ExpectEqual(len(v1alphaPodResources.PodResources), 2)
			framework.ExpectEqual(len(v1PodResources.PodResources), 2)

			var v1alphaResourcesForOurPod *kubeletpodresourcesv1alpha1.PodResources
			for _, res := range v1alphaPodResources.GetPodResources() {
				if res.Name == pod1.Name {
					v1alphaResourcesForOurPod = res
				}
			}

			var v1ResourcesForOurPod *kubeletpodresourcesv1.PodResources
			for _, res := range v1PodResources.GetPodResources() {
				if res.Name == pod1.Name {
					v1ResourcesForOurPod = res
				}
			}

			gomega.Expect(v1alphaResourcesForOurPod).NotTo(gomega.BeNil())
			gomega.Expect(v1ResourcesForOurPod).NotTo(gomega.BeNil())

			framework.ExpectEqual(v1alphaResourcesForOurPod.Name, pod1.Name)
			framework.ExpectEqual(v1ResourcesForOurPod.Name, pod1.Name)

			framework.ExpectEqual(v1alphaResourcesForOurPod.Namespace, pod1.Namespace)
			framework.ExpectEqual(v1ResourcesForOurPod.Namespace, pod1.Namespace)

			framework.ExpectEqual(len(v1alphaResourcesForOurPod.Containers), 1)
			framework.ExpectEqual(len(v1ResourcesForOurPod.Containers), 1)

			framework.ExpectEqual(v1alphaResourcesForOurPod.Containers[0].Name, pod1.Spec.Containers[0].Name)
			framework.ExpectEqual(v1ResourcesForOurPod.Containers[0].Name, pod1.Spec.Containers[0].Name)

			framework.ExpectEqual(len(v1alphaResourcesForOurPod.Containers[0].Devices), 1)
			framework.ExpectEqual(len(v1ResourcesForOurPod.Containers[0].Devices), 1)

			framework.ExpectEqual(v1alphaResourcesForOurPod.Containers[0].Devices[0].ResourceName, resourceName)
			framework.ExpectEqual(v1ResourcesForOurPod.Containers[0].Devices[0].ResourceName, resourceName)

			framework.ExpectEqual(len(v1alphaResourcesForOurPod.Containers[0].Devices[0].DeviceIds), 1)
			framework.ExpectEqual(len(v1ResourcesForOurPod.Containers[0].Devices[0].DeviceIds), 1)
		})

		ginkgo.It("Keeps device plugin assignments across pod and kubelet restarts", func() {
			podRECMD := "devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep 60"
			pod1 := f.PodClient().CreateSync(makeBusyboxPod(resourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devID1 := parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			gomega.Expect(devID1).To(gomega.Not(gomega.Equal("")))

			pod1, err := f.PodClient().Get(context.TODO(), pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ensurePodContainerRestart(f, pod1.Name, pod1.Name)

			ginkgo.By("Confirming that device assignment persists even after container restart")
			devIDAfterRestart := parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectEqual(devIDAfterRestart, devID1)

			ginkgo.By("Restarting Kubelet")
			restartKubelet(true)

			ginkgo.By("Wait for node to be ready again")
			framework.WaitForAllNodesSchedulable(f.ClientSet, 5*time.Minute)

			ginkgo.By("Validating that assignment is kept")
			ensurePodContainerRestart(f, pod1.Name, pod1.Name)
			ginkgo.By("Confirming that after a kubelet restart, fake-device assignment is kept")
			devIDRestart1 := parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart1, devID1)
		})

		ginkgo.It("Keeps device plugin assignments after the device plugin has been re-registered", func() {
			podRECMD := "devs=$(ls /tmp/ | egrep '^Dev-[0-9]+$') && echo stub devices: $devs && sleep 60"
			pod1 := f.PodClient().CreateSync(makeBusyboxPod(resourceName, podRECMD))
			deviceIDRE := "stub devices: (Dev-[0-9]+)"
			devID1 := parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			gomega.Expect(devID1).To(gomega.Not(gomega.Equal("")))

			pod1, err := f.PodClient().Get(context.TODO(), pod1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Restarting Kubelet")
			restartKubelet(true)

			ginkgo.By("Wait for node to be ready again")
			framework.WaitForAllNodesSchedulable(f.ClientSet, 5*time.Minute)

			ginkgo.By("Re-Register resources and delete the plugin pod")
			gp := int64(0)
			deleteOptions := metav1.DeleteOptions{
				GracePeriodSeconds: &gp,
			}
			f.PodClient().DeleteSync(devicePluginPod.Name, deleteOptions, time.Minute)
			waitForContainerRemoval(devicePluginPod.Spec.Containers[0].Name, devicePluginPod.Name, devicePluginPod.Namespace)

			ginkgo.By("Recreating the plugin pod")
			devicePluginPod = f.PodClient().CreateSync(dptemplate)

			ginkgo.By("Confirming that after a kubelet and pod restart, fake-device assignment is kept")
			ensurePodContainerRestart(f, pod1.Name, pod1.Name)
			devIDRestart1 := parseLog(f, pod1.Name, pod1.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart1, devID1)

			ginkgo.By("Waiting for resource to become available on the local node after re-registration")
			gomega.Eventually(func() bool {
				node, ready := getLocalTestNode(f)
				return ready &&
					numberOfDevicesCapacity(node, resourceName) == devsLen &&
					numberOfDevicesAllocatable(node, resourceName) == devsLen
			}, 30*time.Second, framework.Poll).Should(gomega.BeTrue())

			ginkgo.By("Creating another pod")
			pod2 := f.PodClient().CreateSync(makeBusyboxPod(resourceName, podRECMD))

			ginkgo.By("Checking that pod got a different fake device")
			devID2 := parseLog(f, pod2.Name, pod2.Name, deviceIDRE)

			gomega.Expect(devID1).To(gomega.Not(gomega.Equal(devID2)))
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
