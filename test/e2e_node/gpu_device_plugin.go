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
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
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
			initialConfig.FeatureGates += "," + devicePluginFeatureGate
		})

		BeforeEach(func() {
			By("Ensuring that Nvidia GPUs exists on the node")
			if !checkIfNvidiaGPUsExistOnNode() {
				Skip("Nvidia GPUs do not exist on the node. Skipping test.")
			}

			By("Creating the Google Device Plugin pod for NVIDIA GPU in GKE")
			f.PodClient().CreateSync(framework.NVIDIADevicePlugin(f.Namespace.Name))

			By("Waiting for GPUs to become available on the local node")
			Eventually(func() bool {
				return framework.NumberOfNVIDIAGPUs(getLocalNode(f)) > 0
			}, 10*time.Second, time.Second).Should(BeTrue())

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
			devId1 := getDeviceId(f, p1.Name, p1.Name, 1)
			p1, err := f.PodClient().Get(p1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			By("Restarting Kubelet and waiting for the current running pod to restart")
			restartKubelet(f)

			By("Confirming that after a kubelet and pod restart, GPU assignement is kept")
			devIdRestart := getDeviceId(f, p1.Name, p1.Name, 2)
			Expect(devIdRestart).To(Equal(devId1))

			By("Restarting Kubelet and creating another pod")
			restartKubelet(f)
			p2 := f.PodClient().CreateSync(makeCudaPauseImage())

			By("Checking that pods got a different GPU")
			devId2 := getDeviceId(f, p2.Name, p2.Name, 1)
			Expect(devId1).To(Not(Equal(devId2)))

			// Cleanup
			f.PodClient().DeleteSync(p1.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
			f.PodClient().DeleteSync(p2.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
		})
	})
})

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
	if err == nil {
		return
	}
	framework.Failf("Failed to restart kubelet with systemctl: %v, %v", err, stdout)
}

func getDeviceId(f *framework.Framework, podName string, contName string, restartCount int32) string {
	// Wait till pod has been restarted at least restartCount times.
	Eventually(func() bool {
		p, err := f.PodClient().Get(podName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		return p.Status.ContainerStatuses[0].RestartCount >= restartCount
	}, time.Minute, time.Second).Should(BeTrue())
	logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, podName, contName)
	if err != nil {
		framework.Failf("GetPodLogs for pod %q failed: %v", podName, err)
	}
	framework.Logf("got pod logs: %v", logs)
	regex := regexp.MustCompile("gpu devices: (nvidia[0-9]+)")
	matches := regex.FindStringSubmatch(logs)
	if len(matches) < 2 {
		return ""
	}
	return matches[1]
}
