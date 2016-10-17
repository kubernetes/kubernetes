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

package testing

import (
	"fmt"
	"io"
	"reflect"
	"sync"
	"time"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

var (
	version = "0.1.0"

	FakeRuntimeName  = "fakeRuntime"
	FakePodSandboxIP = "192.168.192.168"
)

type FakePodSandbox struct {
	// PodSandboxStatus contains the runtime information for a sandbox.
	runtimeApi.PodSandboxStatus
}

type FakeContainer struct {
	// ContainerStatus contains the runtime information for a container.
	runtimeApi.ContainerStatus

	// the sandbox id of this container
	SandboxID string
}

type FakeRuntimeService struct {
	sync.Mutex

	Called []string

	Containers map[string]*FakeContainer
	Sandboxes  map[string]*FakePodSandbox
}

func (r *FakeRuntimeService) SetFakeSandboxes(sandboxes []*FakePodSandbox) {
	r.Lock()
	defer r.Unlock()

	r.Sandboxes = make(map[string]*FakePodSandbox)
	for _, sandbox := range sandboxes {
		sandboxID := sandbox.GetId()
		r.Sandboxes[sandboxID] = sandbox
	}
}

func (r *FakeRuntimeService) SetFakeContainers(containers []*FakeContainer) {
	r.Lock()
	defer r.Unlock()

	r.Containers = make(map[string]*FakeContainer)
	for _, c := range containers {
		containerID := c.GetId()
		r.Containers[containerID] = c
	}

}

func (r *FakeRuntimeService) AssertCalls(calls []string) error {
	r.Lock()
	defer r.Unlock()

	if !reflect.DeepEqual(calls, r.Called) {
		return fmt.Errorf("expected %#v, got %#v", calls, r.Called)
	}
	return nil
}

func NewFakeRuntimeService() *FakeRuntimeService {
	return &FakeRuntimeService{
		Called:     make([]string, 0),
		Containers: make(map[string]*FakeContainer),
		Sandboxes:  make(map[string]*FakePodSandbox),
	}
}

func (r *FakeRuntimeService) Version(apiVersion string) (*runtimeApi.VersionResponse, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "Version")

	return &runtimeApi.VersionResponse{
		Version:           &version,
		RuntimeName:       &FakeRuntimeName,
		RuntimeVersion:    &version,
		RuntimeApiVersion: &version,
	}, nil
}

func (r *FakeRuntimeService) RunPodSandbox(config *runtimeApi.PodSandboxConfig) (string, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "RunPodSandbox")

	// PodSandboxID should be randomized for real container runtime, but here just use
	// fixed name from BuildSandboxName() for easily making fake sandboxes.
	podSandboxID := BuildSandboxName(config.Metadata)
	createdAt := time.Now().Unix()
	readyState := runtimeApi.PodSandBoxState_READY
	r.Sandboxes[podSandboxID] = &FakePodSandbox{
		PodSandboxStatus: runtimeApi.PodSandboxStatus{
			Id:        &podSandboxID,
			Metadata:  config.Metadata,
			State:     &readyState,
			CreatedAt: &createdAt,
			Network: &runtimeApi.PodSandboxNetworkStatus{
				Ip: &FakePodSandboxIP,
			},
			Labels:      config.Labels,
			Annotations: config.Annotations,
		},
	}

	return podSandboxID, nil
}

func (r *FakeRuntimeService) StopPodSandbox(podSandboxID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "StopPodSandbox")

	notReadyState := runtimeApi.PodSandBoxState_NOTREADY
	if s, ok := r.Sandboxes[podSandboxID]; ok {
		s.State = &notReadyState
	} else {
		return fmt.Errorf("pod sandbox %s not found", podSandboxID)
	}

	return nil
}

func (r *FakeRuntimeService) RemovePodSandbox(podSandboxID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "RemovePodSandbox")

	// Remove the pod sandbox
	delete(r.Sandboxes, podSandboxID)

	return nil
}

func (r *FakeRuntimeService) PodSandboxStatus(podSandboxID string) (*runtimeApi.PodSandboxStatus, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "PodSandboxStatus")

	s, ok := r.Sandboxes[podSandboxID]
	if !ok {
		return nil, fmt.Errorf("pod sandbox %q not found", podSandboxID)
	}

	status := s.PodSandboxStatus
	return &status, nil
}

func (r *FakeRuntimeService) ListPodSandbox(filter *runtimeApi.PodSandboxFilter) ([]*runtimeApi.PodSandbox, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ListPodSandbox")

	result := make([]*runtimeApi.PodSandbox, 0)
	for id, s := range r.Sandboxes {
		if filter != nil {
			if filter.Id != nil && filter.GetId() != id {
				continue
			}
			if filter.State != nil && filter.GetState() != s.GetState() {
				continue
			}
			if filter.LabelSelector != nil && !filterInLabels(filter.LabelSelector, s.GetLabels()) {
				continue
			}
		}

		result = append(result, &runtimeApi.PodSandbox{
			Id:          s.Id,
			Metadata:    s.Metadata,
			State:       s.State,
			CreatedAt:   s.CreatedAt,
			Labels:      s.Labels,
			Annotations: s.Annotations,
		})
	}

	return result, nil
}

func (r *FakeRuntimeService) CreateContainer(podSandboxID string, config *runtimeApi.ContainerConfig, sandboxConfig *runtimeApi.PodSandboxConfig) (string, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "CreateContainer")

	// ContainerID should be randomized for real container runtime, but here just use
	// fixed BuildContainerName() for easily making fake containers.
	containerID := BuildContainerName(config.Metadata, podSandboxID)
	createdAt := time.Now().Unix()
	createdState := runtimeApi.ContainerState_CREATED
	imageRef := config.Image.GetImage()
	r.Containers[containerID] = &FakeContainer{
		ContainerStatus: runtimeApi.ContainerStatus{
			Id:          &containerID,
			Metadata:    config.Metadata,
			Image:       config.Image,
			ImageRef:    &imageRef,
			CreatedAt:   &createdAt,
			State:       &createdState,
			Labels:      config.Labels,
			Annotations: config.Annotations,
		},
		SandboxID: podSandboxID,
	}

	return containerID, nil
}

func (r *FakeRuntimeService) StartContainer(containerID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "StartContainer")

	c, ok := r.Containers[containerID]
	if !ok {
		return fmt.Errorf("container %s not found", containerID)
	}

	// Set container to running.
	startedAt := time.Now().Unix()
	runningState := runtimeApi.ContainerState_RUNNING
	c.State = &runningState
	c.StartedAt = &startedAt

	return nil
}

func (r *FakeRuntimeService) StopContainer(containerID string, timeout int64) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "StopContainer")

	c, ok := r.Containers[containerID]
	if !ok {
		return fmt.Errorf("container %q not found", containerID)
	}

	// Set container to exited state.
	finishedAt := time.Now().Unix()
	exitedState := runtimeApi.ContainerState_EXITED
	c.State = &exitedState
	c.FinishedAt = &finishedAt

	return nil
}

func (r *FakeRuntimeService) RemoveContainer(containerID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "RemoveContainer")

	// Remove the container
	delete(r.Containers, containerID)

	return nil
}

func (r *FakeRuntimeService) ListContainers(filter *runtimeApi.ContainerFilter) ([]*runtimeApi.Container, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ListContainers")

	result := make([]*runtimeApi.Container, 0)
	for _, s := range r.Containers {
		if filter != nil {
			if filter.Id != nil && filter.GetId() != s.GetId() {
				continue
			}
			if filter.PodSandboxId != nil && filter.GetPodSandboxId() != s.SandboxID {
				continue
			}
			if filter.State != nil && filter.GetState() != s.GetState() {
				continue
			}
			if filter.LabelSelector != nil && !filterInLabels(filter.LabelSelector, s.GetLabels()) {
				continue
			}
		}

		result = append(result, &runtimeApi.Container{
			Id:           s.Id,
			CreatedAt:    s.CreatedAt,
			PodSandboxId: &s.SandboxID,
			Metadata:     s.Metadata,
			State:        s.State,
			Image:        s.Image,
			ImageRef:     s.ImageRef,
			Labels:       s.Labels,
			Annotations:  s.Annotations,
		})
	}

	return result, nil
}

func (r *FakeRuntimeService) ContainerStatus(containerID string) (*runtimeApi.ContainerStatus, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ContainerStatus")

	c, ok := r.Containers[containerID]
	if !ok {
		return nil, fmt.Errorf("container %q not found", containerID)
	}

	status := c.ContainerStatus
	return &status, nil
}

func (r *FakeRuntimeService) Exec(containerID string, cmd []string, tty bool, stdin io.Reader, stdout, stderr io.WriteCloser) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "Exec")
	return nil
}

func (r *FakeRuntimeService) UpdateRuntimeConfig(runtimeCOnfig *runtimeApi.RuntimeConfig) error {
	return nil
}
