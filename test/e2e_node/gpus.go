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
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const acceleratorsFeatureGate = "Accelerators=true"

func getGPUsAvailable(f *framework.Framework) int64 {
	nodeList, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	var gpusAvailable int64
	for _, node := range nodeList.Items {
		gpusAvailable += node.Status.Capacity.NvidiaGPU().Value()
	}
	return gpusAvailable
}

func gpusExistOnAllNodes(f *framework.Framework) bool {
	nodeList, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "getting node list")
	for _, node := range nodeList.Items {
		if node.Name == "kubernetes-master" {
			continue
		}
		if node.Status.Capacity.NvidiaGPU().Value() == 0 {
			return false
		}
	}
	return true
}

func checkIfNvidiaGPUsExistOnNode() bool {
	// Cannot use `lspci` because it is not installed on all distros by default.
	err := exec.Command("/bin/sh", "-c", "find /sys/devices/pci* -type f | grep vendor | xargs cat | grep 0x10de").Run()
	if err != nil {
		framework.Logf("check for nvidia GPUs failed. Got Error: %v", err)
		return false
	}
	return true
}

// Serial because the test updates kubelet configuration.
var _ = framework.KubeDescribe("GPU [Serial]", func() {
	f := framework.NewDefaultFramework("gpu-test")
	Context("attempt to use GPUs if available", func() {
		It("setup the node and create pods to test gpus", func() {
			By("ensuring that Nvidia GPUs exist on the node")
			if !checkIfNvidiaGPUsExistOnNode() {
				Skip("Nvidia GPUs do not exist on the node. Skipping test.")
			}
			By("ensuring that dynamic kubelet configuration is enabled")
			enabled, err := isKubeletConfigEnabled(f)
			framework.ExpectNoError(err)
			if !enabled {
				Skip("Dynamic Kubelet configuration is not enabled. Skipping test.")
			}

			By("enabling support for GPUs")
			var oldCfg *kubeletconfig.KubeletConfiguration
			defer func() {
				if oldCfg != nil {
					framework.ExpectNoError(setKubeletConfiguration(f, oldCfg))
				}
			}()

			oldCfg, err = getCurrentKubeletConfig()
			framework.ExpectNoError(err)
			clone, err := scheme.Scheme.DeepCopy(oldCfg)
			framework.ExpectNoError(err)
			newCfg := clone.(*kubeletconfig.KubeletConfiguration)
			if newCfg.FeatureGates != "" {
				newCfg.FeatureGates = fmt.Sprintf("%s,%s", acceleratorsFeatureGate, newCfg.FeatureGates)
			} else {
				newCfg.FeatureGates = acceleratorsFeatureGate
			}
			framework.ExpectNoError(setKubeletConfiguration(f, newCfg))

			By("Waiting for GPUs to become available on the local node")
			Eventually(gpusExistOnAllNodes(f), 10*time.Minute, time.Second).Should(BeTrue())

			By("Creating a pod that will consume all GPUs")
			podSuccess := makePod(getGPUsAvailable(f), "gpus-success")
			podSuccess = f.PodClient().CreateSync(podSuccess)

			By("Checking the containers in the pod had restarted at-least twice successfully thereby ensuring GPUs are reused")
			const minContainerRestartCount = 2
			Eventually(func() bool {
				p, err := f.ClientSet.Core().Pods(f.Namespace.Name).Get(podSuccess.Name, metav1.GetOptions{})
				if err != nil {
					framework.Logf("failed to get pod status: %v", err)
					return false
				}
				if p.Status.ContainerStatuses[0].RestartCount < minContainerRestartCount {
					return false
				}
				return true
			}, time.Minute, time.Second).Should(BeTrue())

			By("Checking if the pod outputted Success to its logs")
			framework.ExpectNoError(f.PodClient().MatchContainerOutput(podSuccess.Name, podSuccess.Name, "Success"))

			By("Creating a new pod requesting a GPU and noticing that it is rejected by the Kubelet")
			podFailure := makePod(1, "gpu-failure")
			framework.WaitForPodCondition(f.ClientSet, f.Namespace.Name, podFailure.Name, "pod rejected", framework.PodStartTimeout, func(pod *v1.Pod) (bool, error) {
				if pod.Status.Phase == v1.PodFailed {
					return true, nil

				}
				return false, nil
			})

			By("stopping the original Pod with GPUs")
			gp := int64(0)
			deleteOptions := metav1.DeleteOptions{
				GracePeriodSeconds: &gp,
			}
			f.PodClient().DeleteSync(podSuccess.Name, &deleteOptions, framework.DefaultPodDeletionTimeout)

			By("attempting to start the failed pod again")
			f.PodClient().DeleteSync(podFailure.Name, &deleteOptions, framework.DefaultPodDeletionTimeout)
			podFailure = f.PodClient().CreateSync(podFailure)

			By("Checking if the pod outputted Success to its logs")
			framework.ExpectNoError(f.PodClient().MatchContainerOutput(podFailure.Name, podFailure.Name, "Success"))
		})
	})
})

func makePod(gpus int64, name string) *v1.Pod {
	resources := v1.ResourceRequirements{
		Limits: v1.ResourceList{
			v1.ResourceNvidiaGPU: *resource.NewQuantity(gpus, resource.DecimalSI),
		},
	}
	gpuverificationCmd := fmt.Sprintf("if [[ %d -ne $(ls /dev/ | egrep '^nvidia[0-9]+$' | wc -l) ]]; then exit 1; else echo Success; fi", gpus)
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyAlways,
			Containers: []v1.Container{
				{
					Image:     "gcr.io/google_containers/busybox:1.24",
					Name:      name,
					Command:   []string{"sh", "-c", gpuverificationCmd},
					Resources: resources,
				},
			},
		},
	}
}
