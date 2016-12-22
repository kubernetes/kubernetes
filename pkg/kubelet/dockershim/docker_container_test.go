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
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
)

// A helper to create a basic config.
func makeContainerConfig(sConfig *runtimeapi.PodSandboxConfig, name, image string, attempt uint32, labels, annotations map[string]string) *runtimeapi.ContainerConfig {
	return &runtimeapi.ContainerConfig{
		Metadata: &runtimeapi.ContainerMetadata{
			Name:    &name,
			Attempt: &attempt,
		},
		Image:       &runtimeapi.ImageSpec{Image: &image},
		Labels:      labels,
		Annotations: annotations,
	}
}

// TestListContainers creates several containers and then list them to check
// whether the correct metadatas, states, and labels are returned.
func TestListContainers(t *testing.T) {
	ds, _, _ := newTestDockerService()
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
	var createdAt int64 = 0
	for i := range configs {
		// We don't care about the sandbox id; pass a bogus one.
		sandboxID := fmt.Sprintf("sandboxid%d", i)
		id, err := ds.CreateContainer(sandboxID, configs[i], sConfigs[i])
		assert.NoError(t, err)
		err = ds.StartContainer(id)
		assert.NoError(t, err)

		imageRef := "" // FakeDockerClient doesn't populate ImageRef yet.
		// Prepend to the expected list because ListContainers returns
		// the most recent containers first.
		expected = append([]*runtimeapi.Container{{
			Metadata:     configs[i].Metadata,
			Id:           &id,
			PodSandboxId: &sandboxID,
			State:        &state,
			CreatedAt:    &createdAt,
			Image:        configs[i].Image,
			ImageRef:     &imageRef,
			Labels:       configs[i].Labels,
			Annotations:  configs[i].Annotations,
		}}, expected...)
	}
	containers, err := ds.ListContainers(nil)
	assert.NoError(t, err)
	assert.Len(t, containers, len(expected))
	assert.Equal(t, expected, containers)
}

// TestContainerStatus tests the basic lifecycle operations and verify that
// the status returned reflects the operations performed.
func TestContainerStatus(t *testing.T) {
	ds, fDocker, fClock := newTestDockerService()
	sConfig := makeSandboxConfig("foo", "bar", "1", 0)
	labels := map[string]string{"abc.xyz": "foo"}
	annotations := map[string]string{"foo.bar.baz": "abc"}
	config := makeContainerConfig(sConfig, "pause", "iamimage", 0, labels, annotations)

	var defaultTime time.Time
	dt := defaultTime.UnixNano()
	ct, st, ft := dt, dt, dt
	state := runtimeapi.ContainerState_CONTAINER_CREATED
	// The following variables are not set in FakeDockerClient.
	imageRef := DockerImageIDPrefix + ""
	exitCode := int32(0)
	var reason, message string

	expected := &runtimeapi.ContainerStatus{
		State:       &state,
		CreatedAt:   &ct,
		StartedAt:   &st,
		FinishedAt:  &ft,
		Metadata:    config.Metadata,
		Image:       config.Image,
		ImageRef:    &imageRef,
		ExitCode:    &exitCode,
		Reason:      &reason,
		Message:     &message,
		Mounts:      []*runtimeapi.Mount{},
		Labels:      config.Labels,
		Annotations: config.Annotations,
	}

	// Create the container.
	fClock.SetTime(time.Now().Add(-1 * time.Hour))
	*expected.CreatedAt = fClock.Now().UnixNano()
	const sandboxId = "sandboxid"
	id, err := ds.CreateContainer(sandboxId, config, sConfig)

	// Check internal labels
	c, err := fDocker.InspectContainer(id)
	assert.NoError(t, err)
	assert.Equal(t, c.Config.Labels[containerTypeLabelKey], containerTypeLabelContainer)
	assert.Equal(t, c.Config.Labels[sandboxIDLabelKey], sandboxId)

	// Set the id manually since we don't know the id until it's created.
	expected.Id = &id
	assert.NoError(t, err)
	status, err := ds.ContainerStatus(id)
	assert.NoError(t, err)
	assert.Equal(t, expected, status)

	// Advance the clock and start the container.
	fClock.SetTime(time.Now())
	*expected.StartedAt = fClock.Now().UnixNano()
	*expected.State = runtimeapi.ContainerState_CONTAINER_RUNNING

	err = ds.StartContainer(id)
	assert.NoError(t, err)
	status, err = ds.ContainerStatus(id)
	assert.Equal(t, expected, status)

	// Advance the clock and stop the container.
	fClock.SetTime(time.Now().Add(1 * time.Hour))
	*expected.FinishedAt = fClock.Now().UnixNano()
	*expected.State = runtimeapi.ContainerState_CONTAINER_EXITED
	*expected.Reason = "Completed"

	err = ds.StopContainer(id, 0)
	assert.NoError(t, err)
	status, err = ds.ContainerStatus(id)
	assert.Equal(t, expected, status)

	// Remove the container.
	err = ds.RemoveContainer(id)
	assert.NoError(t, err)
	status, err = ds.ContainerStatus(id)
	assert.Error(t, err, fmt.Sprintf("status of container: %+v", status))
}

// TestContainerLogPath tests the container log creation logic.
func TestContainerLogPath(t *testing.T) {
	ds, fDocker, _ := newTestDockerService()
	podLogPath := "/pod/1"
	containerLogPath := "0"
	kubeletContainerLogPath := filepath.Join(podLogPath, containerLogPath)
	sConfig := makeSandboxConfig("foo", "bar", "1", 0)
	sConfig.LogDirectory = &podLogPath
	config := makeContainerConfig(sConfig, "pause", "iamimage", 0, nil, nil)
	config.LogPath = &containerLogPath

	const sandboxId = "sandboxid"
	id, err := ds.CreateContainer(sandboxId, config, sConfig)

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
	err = ds.StartContainer(id)
	assert.NoError(t, err)

	err = ds.StopContainer(id, 0)
	assert.NoError(t, err)

	// Verify container log symlink deletion
	err = ds.RemoveContainer(id)
	assert.NoError(t, err)
	assert.Equal(t, fakeOS.Removes, []string{kubeletContainerLogPath})
}
