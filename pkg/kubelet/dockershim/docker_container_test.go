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

package dockershim

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	dockertypes "github.com/docker/docker/api/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
)

// A helper to create a basic config.
func makeContainerConfig(sConfig *runtimeapi.PodSandboxConfig, name, image string, attempt uint32, labels, annotations map[string]string) *runtimeapi.ContainerConfig {
	return &runtimeapi.ContainerConfig{
		Metadata: &runtimeapi.ContainerMetadata{
			Name:    name,
			Attempt: attempt,
		},
		Image:       &runtimeapi.ImageSpec{Image: image},
		Labels:      labels,
		Annotations: annotations,
	}
}

func getTestCTX() context.Context {
	return context.Background()
}

// TestListContainers creates several containers and then list them to check
// whether the correct metadatas, states, and labels are returned.
func TestListContainers(t *testing.T) {
	ds, _, fakeClock := newTestDockerService()
	podName, namespace := "foo", "bar"
	containerName, image := "sidecar", "logger"

	configs := []*runtimeapi.ContainerConfig{}
	sConfigs := []*runtimeapi.PodSandboxConfig{}
	for i := 0; i < 3; i++ {
		s := makeSandboxConfig(fmt.Sprintf("%s%d", podName, i),
			fmt.Sprintf("%s%d", namespace, i), fmt.Sprintf("%d", i), 0)
		labels := map[string]string{"abc.xyz": fmt.Sprintf("label%d", i)}
		annotations := map[string]string{"foo.bar.baz": fmt.Sprintf("annotation%d", i)}
		c := makeContainerConfig(s, fmt.Sprintf("%s%d", containerName, i),
			fmt.Sprintf("%s:v%d", image, i), uint32(i), labels, annotations)
		sConfigs = append(sConfigs, s)
		configs = append(configs, c)
	}

	expected := []*runtimeapi.Container{}
	state := runtimeapi.ContainerState_CONTAINER_RUNNING
	var createdAt int64 = fakeClock.Now().UnixNano()
	for i := range configs {
		// We don't care about the sandbox id; pass a bogus one.
		sandboxID := fmt.Sprintf("sandboxid%d", i)
		req := &runtimeapi.CreateContainerRequest{PodSandboxId: sandboxID, Config: configs[i], SandboxConfig: sConfigs[i]}
		createResp, err := ds.CreateContainer(getTestCTX(), req)
		require.NoError(t, err)
		id := createResp.ContainerId
		_, err = ds.StartContainer(getTestCTX(), &runtimeapi.StartContainerRequest{ContainerId: id})
		require.NoError(t, err)

		imageRef := "" // FakeDockerClient doesn't populate ImageRef yet.
		// Prepend to the expected list because ListContainers returns
		// the most recent containers first.
		expected = append([]*runtimeapi.Container{{
			Metadata:     configs[i].Metadata,
			Id:           id,
			PodSandboxId: sandboxID,
			State:        state,
			CreatedAt:    createdAt,
			Image:        configs[i].Image,
			ImageRef:     imageRef,
			Labels:       configs[i].Labels,
			Annotations:  configs[i].Annotations,
		}}, expected...)
	}
	listResp, err := ds.ListContainers(getTestCTX(), &runtimeapi.ListContainersRequest{})
	require.NoError(t, err)
	assert.Len(t, listResp.Containers, len(expected))
	assert.Equal(t, expected, listResp.Containers)
}

// TestContainerStatus tests the basic lifecycle operations and verify that
// the status returned reflects the operations performed.
func TestContainerStatus(t *testing.T) {
	ds, fDocker, fClock := newTestDockerService()
	sConfig := makeSandboxConfig("foo", "bar", "1", 0)
	labels := map[string]string{"abc.xyz": "foo"}
	annotations := map[string]string{"foo.bar.baz": "abc"}
	imageName := "iamimage"
	config := makeContainerConfig(sConfig, "pause", imageName, 0, labels, annotations)

	var defaultTime time.Time
	dt := defaultTime.UnixNano()
	ct, st, ft := dt, dt, dt
	state := runtimeapi.ContainerState_CONTAINER_CREATED
	imageRef := DockerImageIDPrefix + imageName
	// The following variables are not set in FakeDockerClient.
	exitCode := int32(0)
	var reason, message string

	expected := &runtimeapi.ContainerStatus{
		State:       state,
		CreatedAt:   ct,
		StartedAt:   st,
		FinishedAt:  ft,
		Metadata:    config.Metadata,
		Image:       config.Image,
		ImageRef:    imageRef,
		ExitCode:    exitCode,
		Reason:      reason,
		Message:     message,
		Mounts:      []*runtimeapi.Mount{},
		Labels:      config.Labels,
		Annotations: config.Annotations,
	}

	fDocker.InjectImages([]dockertypes.ImageSummary{{ID: imageName}})

	// Create the container.
	fClock.SetTime(time.Now().Add(-1 * time.Hour))
	expected.CreatedAt = fClock.Now().UnixNano()
	const sandboxId = "sandboxid"

	req := &runtimeapi.CreateContainerRequest{PodSandboxId: sandboxId, Config: config, SandboxConfig: sConfig}
	createResp, err := ds.CreateContainer(getTestCTX(), req)
	require.NoError(t, err)
	id := createResp.ContainerId

	// Check internal labels
	c, err := fDocker.InspectContainer(id)
	require.NoError(t, err)
	assert.Equal(t, c.Config.Labels[containerTypeLabelKey], containerTypeLabelContainer)
	assert.Equal(t, c.Config.Labels[sandboxIDLabelKey], sandboxId)

	// Set the id manually since we don't know the id until it's created.
	expected.Id = id
	assert.NoError(t, err)
	resp, err := ds.ContainerStatus(getTestCTX(), &runtimeapi.ContainerStatusRequest{ContainerId: id})
	require.NoError(t, err)
	assert.Equal(t, expected, resp.Status)

	// Advance the clock and start the container.
	fClock.SetTime(time.Now())
	expected.StartedAt = fClock.Now().UnixNano()
	expected.State = runtimeapi.ContainerState_CONTAINER_RUNNING

	_, err = ds.StartContainer(getTestCTX(), &runtimeapi.StartContainerRequest{ContainerId: id})
	require.NoError(t, err)

	resp, err = ds.ContainerStatus(getTestCTX(), &runtimeapi.ContainerStatusRequest{ContainerId: id})
	require.NoError(t, err)
	assert.Equal(t, expected, resp.Status)

	// Advance the clock and stop the container.
	fClock.SetTime(time.Now().Add(1 * time.Hour))
	expected.FinishedAt = fClock.Now().UnixNano()
	expected.State = runtimeapi.ContainerState_CONTAINER_EXITED
	expected.Reason = "Completed"

	_, err = ds.StopContainer(getTestCTX(), &runtimeapi.StopContainerRequest{ContainerId: id, Timeout: int64(0)})
	assert.NoError(t, err)
	resp, err = ds.ContainerStatus(getTestCTX(), &runtimeapi.ContainerStatusRequest{ContainerId: id})
	require.NoError(t, err)
	assert.Equal(t, expected, resp.Status)

	// Remove the container.
	_, err = ds.RemoveContainer(getTestCTX(), &runtimeapi.RemoveContainerRequest{ContainerId: id})
	require.NoError(t, err)
	resp, err = ds.ContainerStatus(getTestCTX(), &runtimeapi.ContainerStatusRequest{ContainerId: id})
	assert.Error(t, err, fmt.Sprintf("status of container: %+v", resp))
}

// TestContainerLogPath tests the container log creation logic.
func TestContainerLogPath(t *testing.T) {
	ds, fDocker, _ := newTestDockerService()
	podLogPath := "/pod/1"
	containerLogPath := "0"
	kubeletContainerLogPath := filepath.Join(podLogPath, containerLogPath)
	sConfig := makeSandboxConfig("foo", "bar", "1", 0)
	sConfig.LogDirectory = podLogPath
	config := makeContainerConfig(sConfig, "pause", "iamimage", 0, nil, nil)
	config.LogPath = containerLogPath

	const sandboxId = "sandboxid"
	req := &runtimeapi.CreateContainerRequest{PodSandboxId: sandboxId, Config: config, SandboxConfig: sConfig}
	createResp, err := ds.CreateContainer(getTestCTX(), req)
	require.NoError(t, err)
	id := createResp.ContainerId

	// Check internal container log label
	c, err := fDocker.InspectContainer(id)
	assert.NoError(t, err)
	assert.Equal(t, c.Config.Labels[containerLogPathLabelKey], kubeletContainerLogPath)

	// Set docker container log path
	dockerContainerLogPath := "/docker/container/log"
	c.LogPath = dockerContainerLogPath

	// Verify container log symlink creation
	fakeOS := ds.os.(*containertest.FakeOS)
	fakeOS.SymlinkFn = func(oldname, newname string) error {
		assert.Equal(t, dockerContainerLogPath, oldname)
		assert.Equal(t, kubeletContainerLogPath, newname)
		return nil
	}
	_, err = ds.StartContainer(getTestCTX(), &runtimeapi.StartContainerRequest{ContainerId: id})
	require.NoError(t, err)

	_, err = ds.StopContainer(getTestCTX(), &runtimeapi.StopContainerRequest{ContainerId: id, Timeout: int64(0)})
	require.NoError(t, err)

	// Verify container log symlink deletion
	// symlink is also tentatively deleted at startup
	_, err = ds.RemoveContainer(getTestCTX(), &runtimeapi.RemoveContainerRequest{ContainerId: id})
	require.NoError(t, err)
	assert.Equal(t, []string{kubeletContainerLogPath, kubeletContainerLogPath}, fakeOS.Removes)
}

// TestContainerCreationConflict tests the logic to work around docker container
// creation naming conflict bug.
func TestContainerCreationConflict(t *testing.T) {
	sConfig := makeSandboxConfig("foo", "bar", "1", 0)
	config := makeContainerConfig(sConfig, "pause", "iamimage", 0, map[string]string{}, map[string]string{})
	containerName := makeContainerName(sConfig, config)
	const sandboxId = "sandboxid"
	const containerId = "containerid"
	conflictError := fmt.Errorf("Error response from daemon: Conflict. The name \"/%s\" is already in use by container %s. You have to remove (or rename) that container to be able to reuse that name.",
		containerName, containerId)
	noContainerError := fmt.Errorf("Error response from daemon: No such container: %s", containerId)
	randomError := fmt.Errorf("random error")

	for desc, test := range map[string]struct {
		createError  error
		removeError  error
		expectError  error
		expectCalls  []string
		expectFields int
	}{
		"no create error": {
			expectCalls:  []string{"create"},
			expectFields: 6,
		},
		"random create error": {
			createError: randomError,
			expectError: randomError,
			expectCalls: []string{"create"},
		},
		"conflict create error with successful remove": {
			createError: conflictError,
			expectError: conflictError,
			expectCalls: []string{"create", "remove"},
		},
		"conflict create error with random remove error": {
			createError: conflictError,
			removeError: randomError,
			expectError: conflictError,
			expectCalls: []string{"create", "remove"},
		},
		"conflict create error with no such container remove error": {
			createError:  conflictError,
			removeError:  noContainerError,
			expectCalls:  []string{"create", "remove", "create"},
			expectFields: 7,
		},
	} {
		t.Logf("TestCase: %s", desc)
		ds, fDocker, _ := newTestDockerService()

		if test.createError != nil {
			fDocker.InjectError("create", test.createError)
		}
		if test.removeError != nil {
			fDocker.InjectError("remove", test.removeError)
		}

		req := &runtimeapi.CreateContainerRequest{PodSandboxId: sandboxId, Config: config, SandboxConfig: sConfig}
		createResp, err := ds.CreateContainer(getTestCTX(), req)
		require.Equal(t, test.expectError, err)
		assert.NoError(t, fDocker.AssertCalls(test.expectCalls))
		if err == nil {
			c, err := fDocker.InspectContainer(createResp.ContainerId)
			assert.NoError(t, err)
			assert.Len(t, strings.Split(c.Name, nameDelimiter), test.expectFields)
		}
	}
}
