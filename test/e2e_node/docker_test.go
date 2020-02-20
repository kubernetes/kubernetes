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
	"fmt"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Docker features [Feature:Docker][Legacy:Docker]", func() {
	f := framework.NewDefaultFramework("docker-feature-test")

	ginkgo.BeforeEach(func() {
		e2eskipper.RunIfContainerRuntimeIs("docker")
	})

	ginkgo.Context("when live-restore is enabled [Serial] [Slow] [Disruptive]", func() {
		ginkgo.It("containers should not be disrupted when the daemon shuts down and restarts", func() {
			const (
				podName       = "live-restore-test-pod"
				containerName = "live-restore-test-container"
			)

			isSupported, err := isDockerLiveRestoreSupported()
			framework.ExpectNoError(err)
			if !isSupported {
				e2eskipper.Skipf("Docker live-restore is not supported.")
			}
			isEnabled, err := isDockerLiveRestoreEnabled()
			framework.ExpectNoError(err)
			if !isEnabled {
				e2eskipper.Skipf("Docker live-restore is not enabled.")
			}

			ginkgo.By("Create the test pod.")
			pod := f.PodClient().CreateSync(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: podName},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:  containerName,
						Image: imageutils.GetE2EImage(imageutils.Nginx),
					}},
				},
			})

			ginkgo.By("Ensure that the container is running before Docker is down.")
			gomega.Eventually(func() bool {
				return isContainerRunning(pod.Status.PodIP)
			}).Should(gomega.BeTrue())

			startTime1, err := getContainerStartTime(f, podName, containerName)
			framework.ExpectNoError(err)

			ginkgo.By("Stop Docker daemon.")
			framework.ExpectNoError(stopDockerDaemon())
			isDockerDown := true
			defer func() {
				if isDockerDown {
					ginkgo.By("Start Docker daemon.")
					framework.ExpectNoError(startDockerDaemon())
				}
			}()

			ginkgo.By("Ensure that the container is running after Docker is down.")
			gomega.Consistently(func() bool {
				return isContainerRunning(pod.Status.PodIP)
			}).Should(gomega.BeTrue())

			ginkgo.By("Start Docker daemon.")
			framework.ExpectNoError(startDockerDaemon())
			isDockerDown = false

			ginkgo.By("Ensure that the container is running after Docker has restarted.")
			gomega.Consistently(func() bool {
				return isContainerRunning(pod.Status.PodIP)
			}).Should(gomega.BeTrue())

			ginkgo.By("Ensure that the container has not been restarted after Docker is restarted.")
			gomega.Consistently(func() bool {
				startTime2, err := getContainerStartTime(f, podName, containerName)
				framework.ExpectNoError(err)
				return startTime1 == startTime2
			}, 3*time.Second, time.Second).Should(gomega.BeTrue())
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
	pod, err := f.PodClient().Get(context.TODO(), podName, metav1.GetOptions{})
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
