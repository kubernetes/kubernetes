/*
Copyright 2018 The Kubernetes Authors.

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
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubelogs "k8s.io/kubernetes/pkg/kubelet/logs"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	testContainerLogMaxFiles    = 3
	testContainerLogMaxSize     = "40Ki"
	rotationPollInterval        = 5 * time.Second
	rotationEventuallyTimeout   = 3 * time.Minute
	rotationConsistentlyTimeout = 2 * time.Minute
)

var _ = framework.KubeDescribe("ContainerLogRotation [Slow] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("container-log-rotation-test")
	Context("when a container generates a lot of log", func() {
		BeforeEach(func() {
			if framework.TestContext.ContainerRuntime != kubetypes.RemoteContainerRuntime {
				framework.Skipf("Skipping ContainerLogRotation test since the container runtime is not remote")
			}
		})

		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates[string(features.CRIContainerLogRotation)] = true
			initialConfig.ContainerLogMaxFiles = testContainerLogMaxFiles
			initialConfig.ContainerLogMaxSize = testContainerLogMaxSize
		})

		It("should be rotated and limited to a fixed amount of files", func() {
			By("create log container")
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-container-log-rotation",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:  "log-container",
							Image: busyboxImage,
							Command: []string{
								"sh",
								"-c",
								// ~12Kb/s. Exceeding 40Kb in 4 seconds. Log rotation period is 10 seconds.
								"while true; do echo hello world; sleep 0.001; done;",
							},
						},
					},
				},
			}
			pod = f.PodClient().CreateSync(pod)
			By("get container log path")
			Expect(len(pod.Status.ContainerStatuses)).To(Equal(1))
			id := kubecontainer.ParseContainerID(pod.Status.ContainerStatuses[0].ContainerID).ID
			r, _, err := getCRIClient()
			Expect(err).NotTo(HaveOccurred())
			status, err := r.ContainerStatus(id)
			Expect(err).NotTo(HaveOccurred())
			logPath := status.GetLogPath()
			By("wait for container log being rotated to max file limit")
			Eventually(func() (int, error) {
				logs, err := kubelogs.GetAllLogs(logPath)
				if err != nil {
					return 0, err
				}
				return len(logs), nil
			}, rotationEventuallyTimeout, rotationPollInterval).Should(Equal(testContainerLogMaxFiles), "should eventually rotate to max file limit")
			By("make sure container log number won't exceed max file limit")
			Consistently(func() (int, error) {
				logs, err := kubelogs.GetAllLogs(logPath)
				if err != nil {
					return 0, err
				}
				return len(logs), nil
			}, rotationConsistentlyTimeout, rotationPollInterval).Should(BeNumerically("<=", testContainerLogMaxFiles), "should never exceed max file limit")
		})
	})
})
