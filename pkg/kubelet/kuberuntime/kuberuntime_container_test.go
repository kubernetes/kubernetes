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

package kuberuntime

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/kubernetes/pkg/api/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
)

// TestRemoveContainer tests removing the container and its corresponding container logs.
func TestRemoveContainer(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
				},
			},
		},
	}

	// Create fake sandbox and container
	_, fakeContainers := makeAndSetFakePod(t, m, fakeRuntime, pod)
	assert.Equal(t, len(fakeContainers), 1)

	containerId := fakeContainers[0].Id
	fakeOS := m.osInterface.(*containertest.FakeOS)
	err = m.removeContainer(containerId)
	assert.NoError(t, err)
	// Verify container log is removed
	expectedContainerLogPath := filepath.Join(podLogsRootDirectory, "12345678", "foo_0.log")
	expectedContainerLogSymlink := legacyLogSymlink(containerId, "foo", "bar", "new")
	assert.Equal(t, fakeOS.Removes, []string{expectedContainerLogPath, expectedContainerLogSymlink})
	// Verify container is removed
	assert.Contains(t, fakeRuntime.Called, "RemoveContainer")
	containers, err := fakeRuntime.ListContainers(&runtimeapi.ContainerFilter{Id: containerId})
	assert.NoError(t, err)
	assert.Empty(t, containers)
}

// TestToKubeContainerStatus tests the converting the CRI container status to
// the internal type (i.e., toKubeContainerStatus()) for containers in
// different states.
func TestToKubeContainerStatus(t *testing.T) {
	cid := &kubecontainer.ContainerID{Type: "testRuntime", ID: "dummyid"}
	meta := &runtimeapi.ContainerMetadata{Name: "cname", Attempt: 3}
	imageSpec := &runtimeapi.ImageSpec{Image: "fimage"}
	var (
		createdAt  int64 = 327
		startedAt  int64 = 999
		finishedAt int64 = 1278
	)

	for desc, test := range map[string]struct {
		input    *runtimeapi.ContainerStatus
		expected *kubecontainer.ContainerStatus
	}{
		"created container": {
			input: &runtimeapi.ContainerStatus{
				Id:        cid.ID,
				Metadata:  meta,
				Image:     imageSpec,
				State:     runtimeapi.ContainerState_CONTAINER_CREATED,
				CreatedAt: createdAt,
			},
			expected: &kubecontainer.ContainerStatus{
				ID:        *cid,
				Image:     imageSpec.Image,
				State:     kubecontainer.ContainerStateCreated,
				CreatedAt: time.Unix(0, createdAt),
			},
		},
		"running container": {
			input: &runtimeapi.ContainerStatus{
				Id:        cid.ID,
				Metadata:  meta,
				Image:     imageSpec,
				State:     runtimeapi.ContainerState_CONTAINER_RUNNING,
				CreatedAt: createdAt,
				StartedAt: startedAt,
			},
			expected: &kubecontainer.ContainerStatus{
				ID:        *cid,
				Image:     imageSpec.Image,
				State:     kubecontainer.ContainerStateRunning,
				CreatedAt: time.Unix(0, createdAt),
				StartedAt: time.Unix(0, startedAt),
			},
		},
		"exited container": {
			input: &runtimeapi.ContainerStatus{
				Id:         cid.ID,
				Metadata:   meta,
				Image:      imageSpec,
				State:      runtimeapi.ContainerState_CONTAINER_EXITED,
				CreatedAt:  createdAt,
				StartedAt:  startedAt,
				FinishedAt: finishedAt,
				ExitCode:   int32(121),
				Reason:     "GotKilled",
				Message:    "The container was killed",
			},
			expected: &kubecontainer.ContainerStatus{
				ID:         *cid,
				Image:      imageSpec.Image,
				State:      kubecontainer.ContainerStateExited,
				CreatedAt:  time.Unix(0, createdAt),
				StartedAt:  time.Unix(0, startedAt),
				FinishedAt: time.Unix(0, finishedAt),
				ExitCode:   121,
				Reason:     "GotKilled",
				Message:    "The container was killed",
			},
		},
		"unknown container": {
			input: &runtimeapi.ContainerStatus{
				Id:        cid.ID,
				Metadata:  meta,
				Image:     imageSpec,
				State:     runtimeapi.ContainerState_CONTAINER_UNKNOWN,
				CreatedAt: createdAt,
				StartedAt: startedAt,
			},
			expected: &kubecontainer.ContainerStatus{
				ID:        *cid,
				Image:     imageSpec.Image,
				State:     kubecontainer.ContainerStateUnknown,
				CreatedAt: time.Unix(0, createdAt),
				StartedAt: time.Unix(0, startedAt),
			},
		},
	} {
		actual := toKubeContainerStatus(test.input, cid.Type)
		assert.Equal(t, test.expected, actual, desc)
	}
}

func makeExpetectedConfig(m *kubeGenericRuntimeManager, pod *v1.Pod, containerIndex int) *runtimeapi.ContainerConfig {
	container := &pod.Spec.Containers[containerIndex]
	podIP := ""
	restartCount := 0
	opts, _, _ := m.runtimeHelper.GenerateRunContainerOptions(pod, container, podIP)
	containerLogsPath := buildContainerLogsPath(container.Name, restartCount)
	restartCountUint32 := uint32(restartCount)
	envs := make([]*runtimeapi.KeyValue, len(opts.Envs))

	expectedConfig := &runtimeapi.ContainerConfig{
		Metadata: &runtimeapi.ContainerMetadata{
			Name:    container.Name,
			Attempt: restartCountUint32,
		},
		Image:       &runtimeapi.ImageSpec{Image: container.Image},
		Command:     container.Command,
		Args:        []string(nil),
		WorkingDir:  container.WorkingDir,
		Labels:      newContainerLabels(container, pod),
		Annotations: newContainerAnnotations(container, pod, restartCount),
		Devices:     makeDevices(opts),
		Mounts:      m.makeMounts(opts, container),
		LogPath:     containerLogsPath,
		Stdin:       container.Stdin,
		StdinOnce:   container.StdinOnce,
		Tty:         container.TTY,
		Linux:       m.generateLinuxContainerConfig(container, pod, new(int64), ""),
		Envs:        envs,
	}
	return expectedConfig
}

func TestGenerateContainerConfig(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
				},
			},
		},
	}

	expectedConfig := makeExpetectedConfig(m, pod, 0)
	containerConfig, err := m.generateContainerConfig(&pod.Spec.Containers[0], pod, 0, "", pod.Spec.Containers[0].Image)
	assert.NoError(t, err)
	assert.Equal(t, expectedConfig, containerConfig, "generate container config for kubelet runtime v1.")

	runAsUser := int64(0)
	runAsNonRootTrue := true
	podWithContainerSecurityContext := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
					SecurityContext: &v1.SecurityContext{
						RunAsNonRoot: &runAsNonRootTrue,
						RunAsUser:    &runAsUser,
					},
				},
			},
		},
	}

	expectedConfig = makeExpetectedConfig(m, podWithContainerSecurityContext, 0)
	containerConfig, err = m.generateContainerConfig(&podWithContainerSecurityContext.Spec.Containers[0], podWithContainerSecurityContext, 0, "", podWithContainerSecurityContext.Spec.Containers[0].Image)
	assert.Error(t, err)
}
