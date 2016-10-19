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
	internalapi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var (
	imageName = "gcr.io/google_containers/busybox:1.24"
)

func createPodSandbox(c internalapi.RuntimeService) (string, *runtimeapi.PodSandboxConfig) {
	By("create a PodSandbox with name")
	podName := "sandbox-for-create-container-" + string(uuid.NewUUID())
	podConfig := &runtimeapi.PodSandboxConfig{
		Metadata: buildPodSandboxMetadata(&podName),
	}
	podID, err := c.RunPodSandbox(podConfig)
	framework.ExpectNoError(err, "Failed to create PodSandbox: %v", err)
	framework.Logf("Created PodSandbox %s\n", podID)
	return podID, podConfig
}

func testCreateContainer(c internalapi.RuntimeService, podID string, podConfig *runtimeapi.PodSandboxConfig) string {
	By("create container")
	containerName := "container-for-create-test-" + string(uuid.NewUUID())
	containerConfig := &runtimeapi.ContainerConfig{
		Metadata: buildContainerMetadata(&containerName),
		Image:    &runtimeapi.ImageSpec{Image: &imageName},
		Command:  []string{"sh", "-c", "top"},
	}
	containerID, err := c.CreateContainer(podID, containerConfig, podConfig)
	framework.ExpectNoError(err, "Failed to create container: %v", err)
	framework.Logf("Created container %s\n", containerID)

	By("get container status")
	status, err := c.ContainerStatus(containerID)
	framework.ExpectNoError(err, "Failed to get container %s status: %v", containerID, err)
	Expect(*status.State).To(Equal(runtimeapi.ContainerState_CONTAINER_CREATED), "Container state should be created")

	return containerID
}

func testStartContainer(c internalapi.RuntimeService, containerID string) {
	By("start container")
	err := c.StartContainer(containerID)
	framework.ExpectNoError(err, "Failed to start container: %v", err)
	framework.Logf("Start container %s\n", containerID)

	By("get container status")
	status, err := c.ContainerStatus(containerID)
	framework.ExpectNoError(err, "Failed to get container %s status: %v", containerID, err)
	Expect(*status.State).To(Equal(runtimeapi.ContainerState_CONTAINER_RUNNING), "Container state should be started")
}

var _ = framework.KubeDescribe("Test CRI container", func() {
	f := framework.NewDefaultCRIFramework("CRI-container-test")

	var c internalapi.RuntimeService

	BeforeEach(func() {
		c = f.CRIClient.CRIRuntimeClient
	})
	Context("basic container operatings", func() {
		var podID string
		var podConfig *runtimeapi.PodSandboxConfig

		BeforeEach(func() {
			podID, podConfig = createPodSandbox(c)
		})

		AfterEach(func() {
			By("stop PodSandbox")
			c.StopPodSandbox(podID)
			By("delete PodSandbox")
			c.RemovePodSandbox(podID)
		})

		It("test create container", func() {
			testCreateContainer(c, podID, podConfig)
		})

		It("test start container", func() {
			containerID := testCreateContainer(c, podID, podConfig)
			testStartContainer(c, containerID)
		})

		It("test stop container", func() {
			containerID := testCreateContainer(c, podID, podConfig)
			testStartContainer(c, containerID)

			By("stop container")
			err := c.StopContainer(containerID, 60)
			framework.ExpectNoError(err, "Failed to stop container: %v", err)
			framework.Logf("Stop container %s\n", containerID)

			By("get container status")
			status, err := c.ContainerStatus(containerID)
			framework.ExpectNoError(err, "Failed to get container %s status: %v", containerID, err)
			Expect(*status.State).To(Equal(runtimeapi.ContainerState_CONTAINER_EXITED), "Container state should be stopped")
		})

		It("test remove container", func() {
			containerID := testCreateContainer(c, podID, podConfig)
			testStartContainer(c, containerID)

			By("stop container")
			err := c.StopContainer(containerID, 60)
			framework.ExpectNoError(err, "Failed to stop container: %v", err)
			framework.Logf("Stop container %s\n", containerID)

			By("remove container")
			err = c.RemoveContainer(containerID)
			framework.ExpectNoError(err, "Failed to remove container: %v", err)
			framework.Logf("Remove container %s\n", containerID)

			By("list containers with containerID")
			filter := &runtimeapi.ContainerFilter{
				Id: &containerID,
			}
			containers, err := c.ListContainers(filter)
			framework.ExpectNoError(err, "Failed to list containers %s status: %v", containerID, err)
			Expect(containerFound(containers, containerID)).To(BeFalse(), "container should be removed")
		})
	})
})
