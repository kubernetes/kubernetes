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
	"strings"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Docker features [Feature:Docker]", func() {
	f := framework.NewDefaultFramework("docker-feature-test")

	BeforeEach(func() {
		framework.RunIfContainerRuntimeIs("docker")
	})

	Context("when shared PID namespace is enabled", func() {
		It("processes in different containers of the same pod should be able to see each other", func() {
			// TODO(yguo0905): Change this test to run unless the runtime is
			// Docker and its version is <1.13.
			By("Check whether shared PID namespace is supported.")
			isEnabled, err := isSharedPIDNamespaceSupported()
			framework.ExpectNoError(err)
			if !isEnabled {
				framework.Skipf("Skipped because shared PID namespace is not supported by this docker version.")
			}

			By("Create a pod with two containers.")
			f.PodClient().CreateSync(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "shared-pid-ns-test-pod"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "test-container-1",
							Image:   "busybox",
							Command: []string{"/bin/top"},
						},
						{
							Name:    "test-container-2",
							Image:   "busybox",
							Command: []string{"/bin/sleep"},
							Args:    []string{"10000"},
						},
					},
				},
			})

			By("Check if the process in one container is visible to the process in the other.")
			pid1 := f.ExecCommandInContainer("shared-pid-ns-test-pod", "test-container-1", "/bin/pidof", "top")
			pid2 := f.ExecCommandInContainer("shared-pid-ns-test-pod", "test-container-2", "/bin/pidof", "top")
			if pid1 != pid2 {
				framework.Failf("PIDs are not the same in different containers: test-container-1=%v, test-container-2=%v", pid1, pid2)
			}
		})
	})

	Context("when live-restore is enabled [Serial] [Slow] [Disruptive]", func() {
		It("containers should not be disrupted when the daemon shuts down and restarts", func() {
			const (
				podName       = "live-restore-test-pod"
				containerName = "live-restore-test-container"
			)

			isSupported, err := isDockerLiveRestoreSupported()
			framework.ExpectNoError(err)
			if !isSupported {
				framework.Skipf("Docker live-restore is not supported.")
			}
			isEnabled, err := isDockerLiveRestoreEnabled()
			framework.ExpectNoError(err)
			if !isEnabled {
				framework.Skipf("Docker live-restore is not enabled.")
			}

			By("Create the test pod.")
			pod := f.PodClient().CreateSync(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: podName},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:  containerName,
						Image: "gcr.io/google_containers/nginx-slim:0.7",
					}},
				},
			})

			By("Ensure that the container is running before Docker is down.")
			Eventually(func() bool {
				return isContainerRunning(pod.Status.PodIP)
			}).Should(BeTrue())

			startTime1, err := getContainerStartTime(f, podName, containerName)
			framework.ExpectNoError(err)

			By("Stop Docker daemon.")
			framework.ExpectNoError(stopDockerDaemon())
			isDockerDown := true
			defer func() {
				if isDockerDown {
					By("Start Docker daemon.")
					framework.ExpectNoError(startDockerDaemon())
				}
			}()

			By("Ensure that the container is running after Docker is down.")
			Consistently(func() bool {
				return isContainerRunning(pod.Status.PodIP)
			}).Should(BeTrue())

			By("Start Docker daemon.")
			framework.ExpectNoError(startDockerDaemon())
			isDockerDown = false

			By("Ensure that the container is running after Docker has restarted.")
			Consistently(func() bool {
				return isContainerRunning(pod.Status.PodIP)
			}).Should(BeTrue())

			By("Ensure that the container has not been restarted after Docker is restarted.")
			Consistently(func() bool {
				startTime2, err := getContainerStartTime(f, podName, containerName)
				framework.ExpectNoError(err)
				return startTime1 == startTime2
			}, 3*time.Second, time.Second).Should(BeTrue())
		})
	})
})

// isContainerRunning returns true if the container is running by checking
// whether the server is responding, and false otherwise.
func isContainerRunning(podIP string) bool {
	output, err := runCommand("curl", podIP)
	if err != nil {
		return false
	}
	return strings.Contains(output, "Welcome to nginx!")
}

// getContainerStartTime returns the start time of the container with the
// containerName of the pod having the podName.
func getContainerStartTime(f *framework.Framework, podName, containerName string) (time.Time, error) {
	pod, err := f.PodClient().Get(podName, metav1.GetOptions{})
	if err != nil {
		return time.Time{}, fmt.Errorf("failed to get pod %q: %v", podName, err)
	}
	for _, status := range pod.Status.ContainerStatuses {
		if status.Name != containerName {
			continue
		}
		if status.State.Running == nil {
			return time.Time{}, fmt.Errorf("%v/%v is not running", podName, containerName)
		}
		return status.State.Running.StartedAt.Time, nil
	}
	return time.Time{}, fmt.Errorf("failed to find %v/%v", podName, containerName)
}
