/*
Copyright 2016 The Kubernetes Authors.

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

package e2e_cri

import (
	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var (
	imageName = "busybox"
)

func testCreateContainer(c internalApi.RuntimeService) (string, string) {
	By("create a podSandbox with name")
	podName := "sandbox-for-create-container-" + string(uuid.NewUUID())
	podConfig := &runtimeApi.PodSandboxConfig{
		Metadata: &runtimeApi.PodSandboxMetadata{Name: &podName},
	}
	podID, err := c.RunPodSandbox(podConfig)
	framework.ExpectNoError(err, "Failed to create podsandbox: %v", err)
	framework.Logf("Created Podsanbox %s\n", podID)

	By("create container")
	containerName := "container-for-create-test-" + string(uuid.NewUUID())
	containerConfig := &runtimeApi.ContainerConfig{
		Metadata: &runtimeApi.ContainerMetadata{Name: &containerName},
		Image:    &runtimeApi.ImageSpec{Image: &imageName},
		Command:  []string{"sh", "-c", "top"},
	}
	containerID, err := c.CreateContainer(podID, containerConfig, podConfig)
	framework.ExpectNoError(err, "Failed to create container: %v", err)
	framework.Logf("Created container %s\n", containerID)

	By("get container status")
	status, err := c.ContainerStatus(containerID)
	framework.ExpectNoError(err, "Failed to get container %s status: %v", containerID, err)
	Expect(*status.State).To(Equal(runtimeApi.ContainerState_CREATED), "Container state should be created")

	return podID, containerID
}

func testStartContainer(c internalApi.RuntimeService, containerID string) {
	By("start container")
	err := c.StartContainer(containerID)
	framework.ExpectNoError(err, "Failed to start container: %v", err)
	framework.Logf("Start container %s\n", containerID)

	By("get container status")
	status, err := c.ContainerStatus(containerID)
	framework.ExpectNoError(err, "Failed to get container %s status: %v", containerID, err)
	Expect(*status.State).To(Equal(runtimeApi.ContainerState_RUNNING), "Container state should be started")
}

var _ = framework.KubeDescribe("Test CRI container", func() {
	f := framework.NewDefaultCRIFramework("CRI-container-test")

	var c internalApi.RuntimeService

	BeforeEach(func() {
		c = f.CRIClient.CRIRuntimeClient
	})

	It("test create container", func() {
		podID, _ := testCreateContainer(c)
		defer func() {
			By("delete pod sandbox")
			c.StopPodSandbox(podID)
			c.RemovePodSandbox(podID)
		}()
	})

	It("test start container", func() {
		podID, containerID := testCreateContainer(c)
		defer func() {
			By("delete pod sandbox")
			c.StopPodSandbox(podID)
			c.RemovePodSandbox(podID)
		}()

		testStartContainer(c, containerID)
	})

	It("test stop container", func() {
		podID, containerID := testCreateContainer(c)
		defer func() {
			By("delete pod sandbox")
			c.StopPodSandbox(podID)
			c.RemovePodSandbox(podID)
		}()

		testStartContainer(c, containerID)

		By("stop container")
		err := c.StopContainer(containerID, 60)
		framework.ExpectNoError(err, "Failed to stop container: %v", err)
		framework.Logf("Stop container %s\n", containerID)

		By("get container status")
		status, err := c.ContainerStatus(containerID)
		framework.ExpectNoError(err, "Failed to get container %s status: %v", containerID, err)
		Expect(*status.State).To(Equal(runtimeApi.ContainerState_EXITED), "Container state should be stopped")
	})

	It("test remove container", func() {
		podID, containerID := testCreateContainer(c)
		defer func() {
			By("delete pod sandbox")
			c.StopPodSandbox(podID)
			c.RemovePodSandbox(podID)
		}()

		By("remove container")
		err := c.RemoveContainer(containerID)
		framework.ExpectNoError(err, "Failed to remove container: %v", err)
		framework.Logf("Remove container %s\n", containerID)

		By("list containers with containerID")
		filter := &runtimeApi.ContainerFilter{
			Id: &containerID,
		}
		containers, err := c.ListContainers(filter)
		framework.ExpectNoError(err, "Failed to list containers %s status: %v", containerID, err)
		Expect(framework.ContainerFound(containers, containerID)).To(BeFalse(), "container should be removed")
	})
})
