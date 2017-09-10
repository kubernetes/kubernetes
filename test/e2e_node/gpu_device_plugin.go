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
	"os/exec"
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
	sleepTimeout            = 30
)

// Serial because the test restarts Kubelet
var _ = framework.KubeDescribe("NVIDIA GPU Device Plugin [Feature:GPUDevicePlugin] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("device-plugin-gpus-errors")

	Context("", func() {
		BeforeEach(func() {
			By("Ensuring that Nvidia GPUs exists on the node")
			if !checkIfNvidiaGPUsExistOnNode() {
				Skip("Nvidia GPUs do not exist on the node. Skipping test.")
			}

			By("Enabling support for Device Plugin")
			tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
				initialConfig.FeatureGates += "," + devicePluginFeatureGate
			})

			By("Creating the Google Device Plugin pod for NVIDIA GPU in GKE")
			f.PodClient().CreateSync(framework.NVIDIADevicePlugin(f.Namespace.Name))

			By("Waiting for GPUs to become available on the local node")
			Eventually(framework.NumberOfNVIDIAGPUs(getLocalNode(f)) != 0, time.Minute, time.Second).Should(BeTrue())

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
			n := getLocalNode(f)

			By("Creating one GPU pod on a node with at least two GPUs")
			p1 := f.PodClient().CreateSync(makeCudaPauseImage())
			cmd := fmt.Sprintf("exec %s %s nvidia-smi -L", n.Name, p1.Spec.Containers[0].Name)
			uuid1, _ := framework.RunKubectl(cmd)

			By("Restarting Kubelet and waiting for the current running pod to restart")
			restartKubelet(f)
			Eventually(func() bool {
				p, err := f.PodClient().Get(p1.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)

				return p.Status.ContainerStatuses[0].RestartCount != p1.Status.ContainerStatuses[0].RestartCount
			}, 2*sleepTimeout)

			By("Confirming that after a kubelet and pod restart, GPU assignement is kept")
			uuid1Restart, _ := framework.RunKubectl(cmd)
			Expect(uuid1Restart).To(Equal(uuid1))

			By("Restarting Kubelet and creating another pod")
			restartKubelet(f)
			p2 := f.PodClient().CreateSync(makeCudaPauseImage())

			By("Checking that pods got a different GPU")
			cmd = fmt.Sprintf("exec %s %s nvidia-smi -L", n.Name, p2.Spec.Containers[0].Name)
			uuid2, _ := framework.RunKubectl(cmd)
			Expect(uuid1).To(Not(Equal(uuid2)))

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
				Name:    "cuda-pause",
				Image:   "nvidia/cuda",
				Command: []string{"sleep", string(sleepTimeout)},

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
	stdout1, err1 := exec.Command("sudo", "systemctl", "restart", "kubelet").CombinedOutput()
	if err1 == nil {
		return
	}

	stdout2, err2 := exec.Command("sudo", "/etc/init.d/kubelet", "restart").CombinedOutput()
	if err2 == nil {
		return
	}

	stdout3, err3 := exec.Command("sudo", "service", "kubelet", "restart").CombinedOutput()
	if err3 == nil {
		return
	}

	framework.Failf("Failed to trigger kubelet restart with systemctl/initctl/service operations:"+
		"\nsystemclt: %v, %v"+
		"\ninitctl:   %v, %v"+
		"\nservice:   %v, %v", err1, stdout1, err2, stdout2, err3, stdout3)
}
