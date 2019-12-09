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
	"os/exec"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/gpu"
	"k8s.io/kubernetes/test/e2e/framework/metrics"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"github.com/prometheus/common/model"
)

// numberOfNVIDIAGPUs returns the number of GPUs advertised by a node
// This is based on the Device Plugin system and expected to run on a COS based node
// After the NVIDIA drivers were installed
// TODO make this generic and not linked to COS only
func numberOfNVIDIAGPUs(node *v1.Node) int64 {
	val, ok := node.Status.Capacity[gpu.NVIDIAGPUResourceName]
	if !ok {
		return 0
	}
	return val.Value()
}

// NVIDIADevicePlugin returns the official Google Device Plugin pod for NVIDIA GPU in GKE
func NVIDIADevicePlugin() *v1.Pod {
	ds, err := framework.DsFromManifest(gpu.GPUDevicePluginDSYAML)
	framework.ExpectNoError(err)
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "device-plugin-nvidia-gpu-" + string(uuid.NewUUID()),
			Namespace: metav1.NamespaceSystem,
		},
		Spec: ds.Spec.Template.Spec,
	}
	// Remove node affinity
	p.Spec.Affinity = nil
	return p
}

// Serial because the test restarts Kubelet
var _ = framework.KubeDescribe("NVIDIA GPU Device Plugin [Feature:GPUDevicePlugin][NodeFeature:GPUDevicePlugin][Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("device-plugin-gpus-errors")

	ginkgo.Context("DevicePlugin", func() {
		var devicePluginPod *v1.Pod
		var err error
		ginkgo.BeforeEach(func() {
			ginkgo.By("Ensuring that Nvidia GPUs exists on the node")
			if !checkIfNvidiaGPUsExistOnNode() {
				ginkgo.Skip("Nvidia GPUs do not exist on the node. Skipping test.")
			}

			ginkgo.By("Creating the Google Device Plugin pod for NVIDIA GPU in GKE")
			devicePluginPod, err = f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Create(NVIDIADevicePlugin())
			framework.ExpectNoError(err)

			ginkgo.By("Waiting for GPUs to become available on the local node")
			gomega.Eventually(func() bool {
				return numberOfNVIDIAGPUs(getLocalNode(f)) > 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())

			if numberOfNVIDIAGPUs(getLocalNode(f)) < 2 {
				ginkgo.Skip("Not enough GPUs to execute this test (at least two needed)")
			}
		})

		ginkgo.AfterEach(func() {
			l, err := f.PodClient().List(metav1.ListOptions{})
			framework.ExpectNoError(err)

			for _, p := range l.Items {
				if p.Namespace != f.Namespace.Name {
					continue
				}

				f.PodClient().Delete(p.Name, &metav1.DeleteOptions{})
			}
		})

		ginkgo.It("checks that when Kubelet restarts exclusive GPU assignation to pods is kept.", func() {
			ginkgo.By("Creating one GPU pod on a node with at least two GPUs")
			podRECMD := "devs=$(ls /dev/ | egrep '^nvidia[0-9]+$') && echo gpu devices: $devs"
			p1 := f.PodClient().CreateSync(makeBusyboxPod(gpu.NVIDIAGPUResourceName, podRECMD))

			deviceIDRE := "gpu devices: (nvidia[0-9]+)"
			devID1 := parseLog(f, p1.Name, p1.Name, deviceIDRE)
			p1, err := f.PodClient().Get(p1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Restarting Kubelet and waiting for the current running pod to restart")
			restartKubelet()

			ginkgo.By("Confirming that after a kubelet and pod restart, GPU assignment is kept")
			ensurePodContainerRestart(f, p1.Name, p1.Name)
			devIDRestart1 := parseLog(f, p1.Name, p1.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart1, devID1)

			ginkgo.By("Restarting Kubelet and creating another pod")
			restartKubelet()
			framework.WaitForAllNodesSchedulable(f.ClientSet, framework.TestContext.NodeSchedulableTimeout)
			gomega.Eventually(func() bool {
				return numberOfNVIDIAGPUs(getLocalNode(f)) > 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())
			p2 := f.PodClient().CreateSync(makeBusyboxPod(gpu.NVIDIAGPUResourceName, podRECMD))

			ginkgo.By("Checking that pods got a different GPU")
			devID2 := parseLog(f, p2.Name, p2.Name, deviceIDRE)

			framework.ExpectEqual(devID1, devID2)

			ginkgo.By("Deleting device plugin.")
			f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Delete(devicePluginPod.Name, &metav1.DeleteOptions{})
			ginkgo.By("Waiting for GPUs to become unavailable on the local node")
			gomega.Eventually(func() bool {
				node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfNVIDIAGPUs(node) <= 0
			}, 10*time.Minute, framework.Poll).Should(gomega.BeTrue())
			ginkgo.By("Checking that scheduled pods can continue to run even after we delete device plugin.")
			ensurePodContainerRestart(f, p1.Name, p1.Name)
			devIDRestart1 = parseLog(f, p1.Name, p1.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart1, devID1)

			ensurePodContainerRestart(f, p2.Name, p2.Name)
			devIDRestart2 := parseLog(f, p2.Name, p2.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart2, devID2)
			ginkgo.By("Restarting Kubelet.")
			restartKubelet()
			ginkgo.By("Checking that scheduled pods can continue to run even after we delete device plugin and restart Kubelet.")
			ensurePodContainerRestart(f, p1.Name, p1.Name)
			devIDRestart1 = parseLog(f, p1.Name, p1.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart1, devID1)
			ensurePodContainerRestart(f, p2.Name, p2.Name)
			devIDRestart2 = parseLog(f, p2.Name, p2.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart2, devID2)
			logDevicePluginMetrics()

			// Cleanup
			f.PodClient().DeleteSync(p1.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
			f.PodClient().DeleteSync(p2.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
		})
	})
})

func checkIfNvidiaGPUsExistOnNode() bool {
	// Cannot use `lspci` because it is not installed on all distros by default.
	err := exec.Command("/bin/sh", "-c", "find /sys/devices/pci* -type f | grep vendor | xargs cat | grep 0x10de").Run()
	if err != nil {
		framework.Logf("check for nvidia GPUs failed. Got Error: %v", err)
		return false
	}
	return true
}

func logDevicePluginMetrics() {
	ms, err := metrics.GrabKubeletMetricsWithoutProxy(framework.TestContext.NodeName+":10255", "/metrics")
	framework.ExpectNoError(err)
	for msKey, samples := range ms {
		switch msKey {
		case kubeletmetrics.KubeletSubsystem + "_" + kubeletmetrics.DevicePluginAllocationDurationKey:
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
