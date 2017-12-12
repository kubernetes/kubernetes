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
	"os/exec"
	"regexp"
	"strconv"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/metrics"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/prometheus/common/model"
)

const (
	devicePluginFeatureGate = "DevicePlugins=true"
	testPodNamePrefix       = "nvidia-gpu-"
)

// Serial because the test restarts Kubelet
var _ = framework.KubeDescribe("NVIDIA GPU Device Plugin [Feature:GPUDevicePlugin] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("device-plugin-gpus-errors")

	Context("DevicePlugin", func() {
		By("Enabling support for Device Plugin")
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates[string(features.DevicePlugins)] = true
		})

		var devicePluginPod *v1.Pod
		BeforeEach(func() {
			By("Ensuring that Nvidia GPUs exists on the node")
			if !checkIfNvidiaGPUsExistOnNode() {
				Skip("Nvidia GPUs do not exist on the node. Skipping test.")
			}

			framework.WaitForAllNodesSchedulable(f.ClientSet, framework.TestContext.NodeSchedulableTimeout)

			By("Creating the Google Device Plugin pod for NVIDIA GPU in GKE")
			devicePluginPod = f.PodClient().CreateSync(framework.NVIDIADevicePlugin(f.Namespace.Name))

			By("Waiting for GPUs to become available on the local node")
			Eventually(func() bool {
				return framework.NumberOfNVIDIAGPUs(getLocalNode(f)) > 0
			}, 10*time.Second, framework.Poll).Should(BeTrue())

			if framework.NumberOfNVIDIAGPUs(getLocalNode(f)) < 2 {
				Skip("Not enough GPUs to execute this test (at least two needed)")
			}
		})

		AfterEach(func() {
			l, err := f.PodClient().List(metav1.ListOptions{})
			framework.ExpectNoError(err)

			for _, p := range l.Items {
				if p.Namespace != f.Namespace.Name {
					continue
				}

				f.PodClient().Delete(p.Name, &metav1.DeleteOptions{})
			}
		})

		It("checks that when Kubelet restarts exclusive GPU assignation to pods is kept.", func() {
			By("Creating one GPU pod on a node with at least two GPUs")
			p1 := f.PodClient().CreateSync(makeCudaPauseImage())
			count1, devId1 := getDeviceId(f, p1.Name, p1.Name, 1)
			p1, err := f.PodClient().Get(p1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			By("Restarting Kubelet and waiting for the current running pod to restart")
			restartKubelet(f)

			By("Confirming that after a kubelet and pod restart, GPU assignement is kept")
			count1, devIdRestart1 := getDeviceId(f, p1.Name, p1.Name, count1+1)
			Expect(devIdRestart1).To(Equal(devId1))

			By("Restarting Kubelet and creating another pod")
			restartKubelet(f)
			p2 := f.PodClient().CreateSync(makeCudaPauseImage())

			By("Checking that pods got a different GPU")
			count2, devId2 := getDeviceId(f, p2.Name, p2.Name, 1)
			Expect(devId1).To(Not(Equal(devId2)))

			By("Deleting device plugin.")
			f.PodClient().Delete(devicePluginPod.Name, &metav1.DeleteOptions{})
			By("Waiting for GPUs to become unavailable on the local node")
			Eventually(func() bool {
				node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return framework.NumberOfNVIDIAGPUs(node) <= 0
			}, 10*time.Minute, framework.Poll).Should(BeTrue())
			By("Checking that scheduled pods can continue to run even after we delete device plugin.")
			count1, devIdRestart1 = getDeviceId(f, p1.Name, p1.Name, count1+1)
			Expect(devIdRestart1).To(Equal(devId1))
			count2, devIdRestart2 := getDeviceId(f, p2.Name, p2.Name, count2+1)
			Expect(devIdRestart2).To(Equal(devId2))
			By("Restarting Kubelet.")
			restartKubelet(f)
			By("Checking that scheduled pods can continue to run even after we delete device plugin and restart Kubelet.")
			count1, devIdRestart1 = getDeviceId(f, p1.Name, p1.Name, count1+2)
			Expect(devIdRestart1).To(Equal(devId1))
			count2, devIdRestart2 = getDeviceId(f, p2.Name, p2.Name, count2+2)
			Expect(devIdRestart2).To(Equal(devId2))
			logDevicePluginMetrics()

			// Cleanup
			f.PodClient().DeleteSync(p1.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
			f.PodClient().DeleteSync(p2.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
		})
	})
})

func logDevicePluginMetrics() {
	ms, err := metrics.GrabKubeletMetricsWithoutProxy(framework.TestContext.NodeName + ":10255")
	framework.ExpectNoError(err)
	for msKey, samples := range ms {
		switch msKey {
		case kubeletmetrics.KubeletSubsystem + "_" + kubeletmetrics.DevicePluginAllocationLatencyKey:
			for _, sample := range samples {
				latency := sample.Value
				resource := string(sample.Metric["resource_name"])
				var quantile float64
				if val, ok := sample.Metric[model.QuantileLabel]; ok {
					var err error
					if quantile, err = strconv.ParseFloat(string(val), 64); err != nil {
						continue
					}
					framework.Logf("Metric: %v ResourceName: %v Quantile: %v Latency: %v", msKey, resource, quantile, latency)
				}
			}
		case kubeletmetrics.KubeletSubsystem + "_" + kubeletmetrics.DevicePluginRegistrationCountKey:
			for _, sample := range samples {
				resource := string(sample.Metric["resource_name"])
				count := sample.Value
				framework.Logf("Metric: %v ResourceName: %v Count: %v", msKey, resource, count)
			}
		}
	}
}

func makeCudaPauseImage() *v1.Pod {
	podName := testPodNamePrefix + string(uuid.NewUUID())

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyAlways,
			Containers: []v1.Container{{
				Image: busyboxImage,
				Name:  podName,
				// Retrieves the gpu devices created in the user pod.
				// Note the nvidia device plugin implementation doesn't do device id remapping currently.
				// Will probably need to use nvidia-smi if that changes.
				Command: []string{"sh", "-c", "devs=$(ls /dev/ | egrep '^nvidia[0-9]+$') && echo gpu devices: $devs"},

				Resources: v1.ResourceRequirements{
					Limits:   newDecimalResourceList(framework.NVIDIAGPUResourceName, 1),
					Requests: newDecimalResourceList(framework.NVIDIAGPUResourceName, 1),
				},
			}},
		},
	}
}

func newDecimalResourceList(name v1.ResourceName, quantity int64) v1.ResourceList {
	return v1.ResourceList{name: *resource.NewQuantity(quantity, resource.DecimalSI)}
}

// TODO: Find a uniform way to deal with systemctl/initctl/service operations. #34494
func restartKubelet(f *framework.Framework) {
	stdout, err := exec.Command("sudo", "systemctl", "list-units", "kubelet*", "--state=running").CombinedOutput()
	framework.ExpectNoError(err)
	regex := regexp.MustCompile("(kubelet-[0-9]+)")
	matches := regex.FindStringSubmatch(string(stdout))
	Expect(len(matches)).NotTo(BeZero())
	kube := matches[0]
	framework.Logf("Get running kubelet with systemctl: %v, %v", string(stdout), kube)
	stdout, err = exec.Command("sudo", "systemctl", "restart", kube).CombinedOutput()
	framework.ExpectNoError(err, "Failed to restart kubelet with systemctl: %v, %v", err, stdout)
}

func getDeviceId(f *framework.Framework, podName string, contName string, restartCount int32) (int32, string) {
	var count int32
	// Wait till pod has been restarted at least restartCount times.
	Eventually(func() bool {
		p, err := f.PodClient().Get(podName, metav1.GetOptions{})
		if err != nil || len(p.Status.ContainerStatuses) < 1 {
			return false
		}
		count = p.Status.ContainerStatuses[0].RestartCount
		return count >= restartCount
	}, 5*time.Minute, framework.Poll).Should(BeTrue())
	logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, podName, contName)
	if err != nil {
		framework.Failf("GetPodLogs for pod %q failed: %v", podName, err)
	}
	framework.Logf("got pod logs: %v", logs)
	regex := regexp.MustCompile("gpu devices: (nvidia[0-9]+)")
	matches := regex.FindStringSubmatch(logs)
	if len(matches) < 2 {
		return count, ""
	}
	return count, matches[1]
}
