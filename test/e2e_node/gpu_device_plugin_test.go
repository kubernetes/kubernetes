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
	"os/exec"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2egpu "k8s.io/kubernetes/test/e2e/framework/gpu"
	e2emanifest "k8s.io/kubernetes/test/e2e/framework/manifest"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// numberOfNVIDIAGPUs returns the number of GPUs advertised by a node
// This is based on the Device Plugin system and expected to run on a COS based node
// After the NVIDIA drivers were installed
// TODO make this generic and not linked to COS only
func numberOfNVIDIAGPUs(node *v1.Node) int64 {
	val, ok := node.Status.Capacity[e2egpu.NVIDIAGPUResourceName]
	if !ok {
		return 0
	}
	return val.Value()
}

// NVIDIADevicePlugin returns the official Google Device Plugin pod for NVIDIA GPU in GKE
func NVIDIADevicePlugin() *v1.Pod {
	ds, err := e2emanifest.DaemonSetFromURL(e2egpu.GPUDevicePluginDSYAML)
	framework.ExpectNoError(err)
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "device-plugin-nvidia-gpu-" + string(uuid.NewUUID()),
		},
		Spec: ds.Spec.Template.Spec,
	}
	// Remove node affinity
	p.Spec.Affinity = nil
	return p
}

// Serial because the test restarts Kubelet
var _ = SIGDescribe("NVIDIA GPU Device Plugin [Feature:GPUDevicePlugin][NodeFeature:GPUDevicePlugin][Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("device-plugin-gpus-errors")

	ginkgo.Context("DevicePlugin", func() {
		var devicePluginPod *v1.Pod
		ginkgo.BeforeEach(func() {
			ginkgo.By("Ensuring that Nvidia GPUs exists on the node")
			if !checkIfNvidiaGPUsExistOnNode() {
				ginkgo.Skip("Nvidia GPUs do not exist on the node. Skipping test.")
			}

			if framework.TestContext.ContainerRuntime != "docker" {
				ginkgo.Skip("Test works only with in-tree dockershim. Skipping test.")
			}

			ginkgo.By("Creating the Google Device Plugin pod for NVIDIA GPU")
			devicePluginPod = f.PodClient().Create(NVIDIADevicePlugin())

			ginkgo.By("Waiting for GPUs to become available on the local node")
			gomega.Eventually(func() bool {
				return numberOfNVIDIAGPUs(getLocalNode(f)) > 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue(), "GPUs never became available on the local node")

			if numberOfNVIDIAGPUs(getLocalNode(f)) < 2 {
				ginkgo.Skip("Not enough GPUs to execute this test (at least two needed)")
			}
		})

		ginkgo.AfterEach(func() {
			l, err := f.PodClient().List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)

			f.PodClient().DeleteSync(devicePluginPod.Name, metav1.DeleteOptions{}, 2*time.Minute)

			for _, p := range l.Items {
				if p.Namespace != f.Namespace.Name {
					continue
				}

				framework.Logf("Deleting pod: %s", p.Name)
				f.PodClient().DeleteSync(p.Name, metav1.DeleteOptions{}, 2*time.Minute)
			}

			restartKubelet()

			ginkgo.By("Waiting for GPUs to become unavailable on the local node")
			gomega.Eventually(func() bool {
				node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfNVIDIAGPUs(node) <= 0
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())
		})

		// This test is disabled as this behaviour has not existed since at least
		// kubernetes 0.19. If this is a bug, then this test should pass when the
		// issue is resolved. If the behaviour is intentional then it can be removed.
		ginkgo.XIt("keeps GPU assignation to pods after the device plugin has been removed.", func() {
			ginkgo.By("Creating one GPU pod")
			podRECMD := "devs=$(ls /dev/ | egrep '^nvidia[0-9]+$') && echo gpu devices: $devs && sleep 180"
			p1 := f.PodClient().CreateSync(makeBusyboxPod(e2egpu.NVIDIAGPUResourceName, podRECMD))

			deviceIDRE := "gpu devices: (nvidia[0-9]+)"
			devID1 := parseLog(f, p1.Name, p1.Name, deviceIDRE)
			p1, err := f.PodClient().Get(context.TODO(), p1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Deleting the device plugin")
			f.PodClient().DeleteSync(devicePluginPod.Name, metav1.DeleteOptions{}, 2*time.Minute)

			ginkgo.By("Waiting for GPUs to become unavailable on the local node")
			gomega.Eventually(func() int64 {
				node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				return numberOfNVIDIAGPUs(node)
			}, 10*time.Minute, framework.Poll).Should(gomega.BeZero(), "Expected GPUs to eventually be unavailable")

			ginkgo.By("Checking that scheduled pods can continue to run even after we delete the device plugin")
			ensurePodContainerRestart(f, p1.Name, p1.Name)
			devIDRestart1 := parseLog(f, p1.Name, p1.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart1, devID1)

			ginkgo.By("Restarting Kubelet")
			restartKubelet()
			framework.WaitForAllNodesSchedulable(f.ClientSet, 30*time.Minute)

			ginkgo.By("Checking that scheduled pods can continue to run even after we delete device plugin and restart Kubelet.")
			ensurePodContainerRestart(f, p1.Name, p1.Name)
			devIDRestart1 = parseLog(f, p1.Name, p1.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart1, devID1)

			// Cleanup
			f.PodClient().DeleteSync(p1.Name, metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
		})

		ginkgo.It("keeps GPU assignment to pods across pod and kubelet restarts.", func() {
			ginkgo.By("Creating one GPU pod on a node with at least two GPUs")
			podRECMD := "devs=$(ls /dev/ | egrep '^nvidia[0-9]+$') && echo gpu devices: $devs && sleep 40"
			p1 := f.PodClient().CreateSync(makeBusyboxPod(e2egpu.NVIDIAGPUResourceName, podRECMD))

			deviceIDRE := "gpu devices: (nvidia[0-9]+)"
			devID1 := parseLog(f, p1.Name, p1.Name, deviceIDRE)
			p1, err := f.PodClient().Get(context.TODO(), p1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Confirming that after many pod restarts, GPU assignment is kept")
			for i := 0; i < 3; i++ {
				ensurePodContainerRestart(f, p1.Name, p1.Name)
				devIDRestart1 := parseLog(f, p1.Name, p1.Name, deviceIDRE)
				framework.ExpectEqual(devIDRestart1, devID1)
			}

			ginkgo.By("Restarting Kubelet")
			restartKubelet()

			ginkgo.By("Confirming that after a kubelet and pod restart, GPU assignment is kept")
			ensurePodContainerRestart(f, p1.Name, p1.Name)
			devIDRestart1 := parseLog(f, p1.Name, p1.Name, deviceIDRE)
			framework.ExpectEqual(devIDRestart1, devID1)

			ginkgo.By("Restarting Kubelet and creating another pod")

			restartKubelet()
			framework.WaitForAllNodesSchedulable(f.ClientSet, 30*time.Minute)

			ensurePodContainerRestart(f, p1.Name, p1.Name)

			gomega.Eventually(func() bool {
				return numberOfNVIDIAGPUs(getLocalNode(f)) >= 2
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())

			p2 := f.PodClient().CreateSync(makeBusyboxPod(e2egpu.NVIDIAGPUResourceName, podRECMD))

			ginkgo.By("Checking that pods got a different GPU")
			devID2 := parseLog(f, p2.Name, p2.Name, deviceIDRE)

			framework.ExpectNotEqual(devID1, devID2)
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
