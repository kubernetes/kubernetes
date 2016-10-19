/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance for the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
forOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package e2e_cri

import (
	"fmt"
	"strings"

	internalapi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/util/uuid"
	e2eframework "k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var (
	defaultUid                  string = "e2e-cri-uid"
	defaultNamespace            string = "e2e-cri-namespace"
	defaultAttempt              uint32 = 2
	defaultContainerImage       string = "gcr.io/google_containers/busybox:1.24"
	defaultStopContainerTimeout int64  = 60
)

// sandboxFound returns whether PodSandbox is found.
func podSandboxFound(podSandboxs []*runtimeapi.PodSandbox, podId string) bool {
	if len(podSandboxs) == 1 && podSandboxs[0].GetId() == podId {
		return true
	}
	return false
}

// containerFound returns whether containers is found.
func containerFound(containers []*runtimeapi.Container, containerID string) bool {
	if len(containers) == 1 && containers[0].GetId() == containerID {
		return true
	}
	return false
}

// buildPodSandboxMetadata builds default PodSandboxMetadata with podSandboxName.
func buildPodSandboxMetadata(podSandboxName *string) *runtimeapi.PodSandboxMetadata {
	return &runtimeapi.PodSandboxMetadata{
		Name:      podSandboxName,
		Uid:       &defaultUid,
		Namespace: &defaultNamespace,
		Attempt:   &defaultAttempt,
	}
}

// buildContainerMetadata builds default PodSandboxMetadata with containerName.
func buildContainerMetadata(containerName *string) *runtimeapi.ContainerMetadata {
	return &runtimeapi.ContainerMetadata{
		Name:    containerName,
		Attempt: &defaultAttempt,
	}
}

// verifyPodSandboxStatus verifies whether PodSandbox status for given podID matches.
func verifyPodSandboxStatus(c internalapi.RuntimeService, podID string, expectedStatus runtimeapi.PodSandboxState, statusName string) {
	status := getPodSandboxStatusOrFail(c, podID)
	Expect(*status.State).To(Equal(expectedStatus), "PodSandbox state should be "+statusName)
}

// runPodSandbox runs a PodSandbox with prefixed random name.
func runPodSandbox(c internalapi.RuntimeService, prefix string) (string, error) {
	By("create a PodSandbox with name")
	podSandboxName := prefix + string(uuid.NewUUID())
	config := &runtimeapi.PodSandboxConfig{
		Metadata: buildPodSandboxMetadata(&podSandboxName),
	}
	return c.RunPodSandbox(config)
}

// runPodSandboxOrFail runs a PodSandbox with the prefix of podSandboxName and fails if it gets error.
func runPodSandboxOrFail(c internalapi.RuntimeService, prefix string) string {
	podID, err := runPodSandbox(c, prefix)
	e2eframework.ExpectNoError(err, "Failed to create PodSandbox: %v", err)
	e2eframework.Logf("Created PodSandbox %s\n", podID)
	return podID
}

// testRunPodSandbox runs a PodSandbox and make sure it is ready.
func testRunPodSandbox(c internalapi.RuntimeService) string {
	podID := runPodSandboxOrFail(c, "PodSandbox-for-create-test-")
	verifyPodSandboxStatus(c, podID, runtimeapi.PodSandboxState_SANDBOX_READY, "ready")
	return podID
}

// getPodSandboxStatus gets gets PodSandboxStatus, error for podID.
func getPodSandboxStatus(c internalapi.RuntimeService, podID string) (*runtimeapi.PodSandboxStatus, error) {
	By("get PodSandbox status")
	return c.PodSandboxStatus(podID)
}

// getPodSandboxStatusOrFail gets PodSandboxStatus for podID and fails if it gets error.
func getPodSandboxStatusOrFail(c internalapi.RuntimeService, podID string) *runtimeapi.PodSandboxStatus {
	status, err := getPodSandboxStatus(c, podID)
	e2eframework.ExpectNoError(err, "Failed to get PodSandbox %s status: %v", podID, err)
	return status
}

// stopPodSandbox stops the PodSandbox for podID.
func stopPodSandbox(c internalapi.RuntimeService, podID string) error {
	By("stop PodSandbox for podID")
	return c.StopPodSandbox(podID)
}

// stopPodSandboxOrFail stops the PodSandbox for podID and fails if it gets error.
func stopPodSandboxOrFail(c internalapi.RuntimeService, podID string) {
	err := stopPodSandbox(c, podID)
	e2eframework.ExpectNoError(err, "Failed to stop PodSandbox: %v", err)
	e2eframework.Logf("Stopped PodSandbox %s\n", podID)
}

// testStopPodSandbox stops the PodSandbox for podID and make sure it is not ready.
func testStopPodSandbox(c internalapi.RuntimeService, podID string) {
	stopPodSandboxOrFail(c, podID)
	verifyPodSandboxStatus(c, podID, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, "not ready")
}

// removePodSandbox removes the PodSandbox for podID.
func removePodSandbox(c internalapi.RuntimeService, podID string) error {
	By("remove PodSandbox for podID")
	return c.RemovePodSandbox(podID)
}

// removePodSandboxOrFail removes the PodSandbox for podID and fails if it gets error.
func removePodSandboxOrFail(c internalapi.RuntimeService, podID string) {
	err := removePodSandbox(c, podID)
	e2eframework.ExpectNoError(err, "Failed to remove PodSandbox: %v", err)
	e2eframework.Logf("Removed PodSandbox %s\n", podID)
}

// listPodSanboxforID lists PodSandbox for podID.
func listPodSanboxForID(c internalapi.RuntimeService, podID string) ([]*runtimeapi.PodSandbox, error) {
	By("list PodSandbox for podID")
	filter := &runtimeapi.PodSandboxFilter{
		Id: &podID,
	}
	return c.ListPodSandbox(filter)
}

// listPodSanboxforIDOrFail lists PodSandbox for podID and fails if it gets error.
func listPodSanboxForIDOrFail(c internalapi.RuntimeService, podID string) []*runtimeapi.PodSandbox {
	pods, err := listPodSanboxForID(c, podID)
	e2eframework.ExpectNoError(err, "Failed to list PodSandbox %s status: %v", podID, err)
	return pods
}

// getContainerStatus gets ContainerState for containerID.
func getContainerStatus(c internalapi.RuntimeService, containerID string) (*runtimeapi.ContainerStatus, error) {
	By("get container status")
	return c.ContainerStatus(containerID)
}

// getPodSandboxStatusOrFail gets ContainerState for containerID and fails if it gets error.
func getContainerStatusOrFail(c internalapi.RuntimeService, containerID string) *runtimeapi.ContainerStatus {
	status, err := getContainerStatus(c, containerID)
	e2eframework.ExpectNoError(err, "Failed to get container %s status: %v", containerID, err)
	return status
}

// verifyContainerStatus verifies whether status for given containerID matches.
func verifyContainerStatus(c internalapi.RuntimeService, containerID string, expectedStatus runtimeapi.ContainerState, stateName string) {
	status := getContainerStatusOrFail(c, containerID)
	Expect(*status.State).To(Equal(expectedStatus), "Container state should be %s", stateName)
}

// runPodSandbox creates a container with the prefix of containerName.
func createContainer(c internalapi.RuntimeService, prefix string, podID string, podConfig *runtimeapi.PodSandboxConfig) (string, error) {
	By("create a container with name")
	containerName := prefix + string(uuid.NewUUID())
	containerConfig := &runtimeapi.ContainerConfig{
		Metadata: buildContainerMetadata(&containerName),
		Image:    &runtimeapi.ImageSpec{Image: &defaultContainerImage},
		Command:  []string{"sh", "-c", "top"},
	}
	return c.CreateContainer(podID, containerConfig, podConfig)
}

// createContainerOrFail creates a container with the prefix of containerName and fails if it gets error.
func createContainerOrFail(c internalapi.RuntimeService, prefix string, podID string, podConfig *runtimeapi.PodSandboxConfig) string {
	containerID, err := createContainer(c, prefix, podID, podConfig)
	e2eframework.ExpectNoError(err, "Failed to create container: %v", err)
	e2eframework.Logf("Created container %s\n", containerID)
	return containerID
}

// testCreateContainer creates a container in the pod which ID is podID and make sure it be ready.
func testCreateContainer(c internalapi.RuntimeService, podID string, podConfig *runtimeapi.PodSandboxConfig) string {
	containerID := createContainerOrFail(c, "container-for-create-test-", podID, podConfig)
	verifyContainerStatus(c, containerID, runtimeapi.ContainerState_CONTAINER_CREATED, "created")
	return containerID
}

// startContainer start the container for containerID.
func startContainer(c internalapi.RuntimeService, containerID string) error {
	By("start container")
	return c.StartContainer(containerID)
}

// startcontainerOrFail starts the container for containerID and fails if it gets error.
func startContainerOrFail(c internalapi.RuntimeService, containerID string) {
	err := startContainer(c, containerID)
	e2eframework.ExpectNoError(err, "Failed to start container: %v", err)
	e2eframework.Logf("Start container %s\n", containerID)
}

// testStartContainer starts the container for containerID and make sure it be running.
func testStartContainer(c internalapi.RuntimeService, containerID string) {
	startContainerOrFail(c, containerID)
	verifyContainerStatus(c, containerID, runtimeapi.ContainerState_CONTAINER_RUNNING, "running")
}

// stopContainer stops the container for containerID.
func stopContainer(c internalapi.RuntimeService, containerID string, timeout int64) error {
	By("stop container")
	return c.StopContainer(containerID, timeout)
}

// stopContainerOrFail stops the container for containerID and fails if it gets error.
func stopContainerOrFail(c internalapi.RuntimeService, containerID string, timeout int64) {
	err := stopContainer(c, containerID, timeout)
	e2eframework.ExpectNoError(err, "Failed to stop container: %v", err)
	e2eframework.Logf("Stop container %s\n", containerID)
}

// testStopContainer stops the container for containerID and make sure it be exited.
func testStopContainer(c internalapi.RuntimeService, containerID string) {
	stopContainerOrFail(c, containerID, defaultStopContainerTimeout)
	verifyContainerStatus(c, containerID, runtimeapi.ContainerState_CONTAINER_EXITED, "exited")
}

// removePodSandbox removes the container for containerID.
func removeContainer(c internalapi.RuntimeService, containerID string) error {
	By("remove container for containerID")
	return c.RemoveContainer(containerID)
}

// removeContainerOrFail removes the container for containerID and fails if it gets error.
func removeContainerOrFail(c internalapi.RuntimeService, containerID string) {
	err := removeContainer(c, containerID)
	e2eframework.ExpectNoError(err, "Failed to remove container: %v", err)
	e2eframework.Logf("Removed container %s\n", containerID)
}

// listContainerforID lists container for podID.
func listContainerForID(c internalapi.RuntimeService, containerID string) ([]*runtimeapi.Container, error) {
	By("list containers for containerID")
	filter := &runtimeapi.ContainerFilter{
		Id: &containerID,
	}
	return c.ListContainers(filter)
}

// listContainerforID lists container for podID and fails if it gets error.
func listContainerForIDOrFail(c internalapi.RuntimeService, containerID string) []*runtimeapi.Container {
	containers, err := listContainerForID(c, containerID)
	e2eframework.ExpectNoError(err, "Failed to list containers %s status: %v", containerID, err)
	return containers
}

// createPodSandboxForContainer creates a PodSandbox for creating containers.
func createPodSandboxForContainer(c internalapi.RuntimeService) (string, *runtimeapi.PodSandboxConfig) {
	By("create a PodSandbox for creating containers")
	podName := "PodSandbox-for-create-container-" + string(uuid.NewUUID())
	podConfig := &runtimeapi.PodSandboxConfig{
		Metadata: buildPodSandboxMetadata(&podName),
	}
	podID, err := c.RunPodSandbox(podConfig)
	e2eframework.ExpectNoError(err, "Failed to create PodSandbox: %v", err)
	e2eframework.Logf("Created PodSandbox %s\n", podID)
	return podID, podConfig
}

// pullPublicImage pulls an image named imageName.
func pullPublicImage(c internalapi.ImageManagerService, imageName string) error {
	By(fmt.Sprintf("pull image %s", imageName))
	imageSpec := &runtimeapi.ImageSpec{
		Image: &imageName,
	}
	return c.PullImage(imageSpec, nil)
}

// pullPublicImageOrFail pulls an image named imageName and fails if it gets error.
func pullPublicImageOrFail(c internalapi.ImageManagerService, imageName string) {
	err := pullPublicImage(c, imageName)
	e2eframework.ExpectNoError(err, "Failed to pull image: %v", err)
}

// listImages lists the images named imageName.
func listImages(c internalapi.ImageManagerService, imageName string) ([]*runtimeapi.Image, error) {
	By("get image list")
	imageSpec := &runtimeapi.ImageSpec{
		Image: &imageName,
	}
	filter := &runtimeapi.ImageFilter{
		Image: imageSpec,
	}
	return c.ListImages(filter)
}

// listImagesOrFail lists the images named imageName and fails if it gets error.
func listImagesOrFail(c internalapi.ImageManagerService, imageName string) []*runtimeapi.Image {
	images, err := listImages(c, imageName)
	e2eframework.ExpectNoError(err, "Failed to get image list: %v", err)
	return images
}

// removeImage removes the image named imagesName.
func removeImage(c internalapi.ImageManagerService, imageName string) error {
	By(fmt.Sprintf("remove image %s", imageName))
	imageSpec := &runtimeapi.ImageSpec{
		Image: &imageName,
	}
	return c.RemoveImage(imageSpec)
}

// removeImageOrFail removes the image named imagesName and fails if it gets error.
func removeImageOrFail(c internalapi.ImageManagerService, imageName string) {
	err := removeImage(c, imageName)
	e2eframework.ExpectNoError(err, "Failed to remove image: %v", err)
}

// testRemoveImage removes the image name imageName and check if it successes.
func testRemoveImage(c internalapi.ImageManagerService, imageName string) {
	By(fmt.Sprintf("remove image %s", imageName))
	removeImageOrFail(c, imageName)

	By("check image list empty")
	iamges := listImagesOrFail(c, imageName)
	Expect(len(iamges)).To(Equal(0), "should have none image in list")
}

// testPullPublicImage pulls the image named imageName, make sure it success and remove the image.
func testPullPublicImage(c internalapi.ImageManagerService, imageName string) {
	if !strings.Contains(imageName, ":") {
		imageName = imageName + ":latest"
	}
	pullPublicImageOrFail(c, imageName)

	By("check image list to make sure pulling image success")
	images := listImagesOrFail(c, imageName)
	Expect(len(images)).To(Equal(1), "should have one image in list")

	testRemoveImage(c, imageName)
}

// imageStatus gets the status of the image named imageName.
func imageStatus(c internalapi.ImageManagerService, imageName string) (*runtimeapi.Image, error) {
	By("get image status")
	imageSpec := &runtimeapi.ImageSpec{
		Image: &imageName,
	}
	return c.ImageStatus(imageSpec)
}

// imageStatusOrFail gets the status of the image named imageName and fails if it gets error.
func imageStatusOrFail(c internalapi.ImageManagerService, imageName string) *runtimeapi.Image {
	status, err := imageStatus(c, imageName)
	e2eframework.ExpectNoError(err, "Failed to get image status: %v", err)
	return status
}

// pullImageList pulls the images listed in the imageList.
func pullImageList(c internalapi.ImageManagerService, imageList []string) {
	for _, imageName := range imageList {
		pullPublicImageOrFail(c, imageName)
	}
}

// pullImageList removes the images listed in the imageList.
func removeImageList(c internalapi.ImageManagerService, imageList []string) {
	for _, imageName := range imageList {
		removeImageOrFail(c, imageName)
	}
}

// getVersion gets version info from server.
func getVersion(c internalapi.RuntimeService, apiVersion string) (*runtimeapi.VersionResponse, error) {
	return c.Version("test")
}

// getVersion gets version info from server and fails if it gets error.
func getVersionOrFail(c internalapi.RuntimeService, apiVersion string) *runtimeapi.VersionResponse {
	version, err := getVersion(c, apiVersion)
	e2eframework.ExpectNoError(err, "Failed to get version: %v", err)
	return version
}

// testGetVersion test if we can get runtime name.
func testGetVersion(c internalapi.RuntimeService) {
	version := getVersionOrFail(c, "test")
	e2eframework.Logf("get version runtime name " + *version.RuntimeName)
}
