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
	"reflect"
	"sync"
	"time"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

var (
	FakeVersion = "0.1.0"

	FakeRuntimeName   = "fakeRuntime"
	FakePodSandboxIPs = []string{"192.168.192.168"}
)

type FakePodSandbox struct {
	// PodSandboxStatus contains the runtime information for a sandbox.
	runtimeapi.PodSandboxStatus
	// RuntimeHandler is the runtime handler that was issued with the RunPodSandbox request.
	RuntimeHandler string
}

type FakeContainer struct {
	// ContainerStatus contains the runtime information for a container.
	runtimeapi.ContainerStatus

	// LinuxResources contains the resources specific to linux containers.
	LinuxResources *runtimeapi.LinuxContainerResources

	// the sandbox id of this container
	SandboxID string
}

type FakeRuntimeService struct {
	sync.Mutex

	Called []string
	Errors map[string][]error

	FakeStatus         *runtimeapi.RuntimeStatus
	Containers         map[string]*FakeContainer
	Sandboxes          map[string]*FakePodSandbox
	FakeContainerStats map[string]*runtimeapi.ContainerStats
}

func (r *FakeRuntimeService) GetContainerID(sandboxID, name string, attempt uint32) (string, error) {
	r.Lock()
	defer r.Unlock()

	for id, c := range r.Containers {
		if c.SandboxID == sandboxID && c.Metadata.Name == name && c.Metadata.Attempt == attempt {
			return id, nil
		}
	}
	return "", fmt.Errorf("container (name, attempt, sandboxID)=(%q, %d, %q) not found", name, attempt, sandboxID)
}

func (r *FakeRuntimeService) SetFakeSandboxes(sandboxes []*FakePodSandbox) {
	r.Lock()
	defer r.Unlock()

	r.Sandboxes = make(map[string]*FakePodSandbox)
	for _, sandbox := range sandboxes {
		sandboxID := sandbox.Id
		r.Sandboxes[sandboxID] = sandbox
	}
}

func (r *FakeRuntimeService) SetFakeContainers(containers []*FakeContainer) {
	r.Lock()
	defer r.Unlock()

	r.Containers = make(map[string]*FakeContainer)
	for _, c := range containers {
		containerID := c.Id
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

func (r *FakeRuntimeService) GetCalls() []string {
	r.Lock()
	defer r.Unlock()
	return append([]string{}, r.Called...)
}

func (r *FakeRuntimeService) InjectError(f string, err error) {
	r.Lock()
	defer r.Unlock()
	r.Errors[f] = append(r.Errors[f], err)
}

// caller of popError must grab a lock.
func (r *FakeRuntimeService) popError(f string) error {
	if r.Errors == nil {
		return nil
	}
	errs := r.Errors[f]
	if len(errs) == 0 {
		return nil
	}
	err, errs := errs[0], errs[1:]
	r.Errors[f] = errs
	return err
}

func NewFakeRuntimeService() *FakeRuntimeService {
	return &FakeRuntimeService{
		Called:             make([]string, 0),
		Errors:             make(map[string][]error),
		Containers:         make(map[string]*FakeContainer),
		Sandboxes:          make(map[string]*FakePodSandbox),
		FakeContainerStats: make(map[string]*runtimeapi.ContainerStats),
	}
}

func (r *FakeRuntimeService) Version(apiVersion string) (*runtimeapi.VersionResponse, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "Version")
	if err := r.popError("Version"); err != nil {
		return nil, err
	}

	return &runtimeapi.VersionResponse{
		Version:           FakeVersion,
		RuntimeName:       FakeRuntimeName,
		RuntimeVersion:    FakeVersion,
		RuntimeApiVersion: FakeVersion,
	}, nil
}

func (r *FakeRuntimeService) Status() (*runtimeapi.RuntimeStatus, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "Status")
	if err := r.popError("Status"); err != nil {
		return nil, err
	}

	return r.FakeStatus, nil
}

func (r *FakeRuntimeService) RunPodSandbox(config *runtimeapi.PodSandboxConfig, runtimeHandler string) (string, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "RunPodSandbox")
	if err := r.popError("RunPodSandbox"); err != nil {
		return "", err
	}

	// PodSandboxID should be randomized for real container runtime, but here just use
	// fixed name from BuildSandboxName() for easily making fake sandboxes.
	podSandboxID := BuildSandboxName(config.Metadata)
	createdAt := time.Now().UnixNano()
	r.Sandboxes[podSandboxID] = &FakePodSandbox{
		PodSandboxStatus: runtimeapi.PodSandboxStatus{
			Id:        podSandboxID,
			Metadata:  config.Metadata,
			State:     runtimeapi.PodSandboxState_SANDBOX_READY,
			CreatedAt: createdAt,
			Network: &runtimeapi.PodSandboxNetworkStatus{
				Ip: FakePodSandboxIPs[0],
			},
			// Without setting sandboxStatus's Linux.Namespaces.Options, kubeGenericRuntimeManager's podSandboxChanged will consider it as network
			// namespace changed and always recreate sandbox which causes pod creation failed.
			// Ref `sandboxStatus.GetLinux().GetNamespaces().GetOptions().GetNetwork() != networkNamespaceForPod(pod)` in podSandboxChanged function.
			Linux: &runtimeapi.LinuxPodSandboxStatus{
				Namespaces: &runtimeapi.Namespace{
					Options: config.GetLinux().GetSecurityContext().GetNamespaceOptions(),
				},
			},
			Labels:         config.Labels,
			Annotations:    config.Annotations,
			RuntimeHandler: runtimeHandler,
		},
		RuntimeHandler: runtimeHandler,
	}
	// assign additional IPs
	additionalIPs := FakePodSandboxIPs[1:]
	additionalPodIPs := make([]*runtimeapi.PodIP, 0, len(additionalIPs))
	for _, ip := range additionalIPs {
		additionalPodIPs = append(additionalPodIPs, &runtimeapi.PodIP{
			Ip: ip,
		})
	}
	r.Sandboxes[podSandboxID].PodSandboxStatus.Network.AdditionalIps = additionalPodIPs
	return podSandboxID, nil
}

func (r *FakeRuntimeService) StopPodSandbox(podSandboxID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "StopPodSandbox")
	if err := r.popError("StopPodSandbox"); err != nil {
		return err
	}

	if s, ok := r.Sandboxes[podSandboxID]; ok {
		s.State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
	} else {
		return fmt.Errorf("pod sandbox %s not found", podSandboxID)
	}

	return nil
}

func (r *FakeRuntimeService) RemovePodSandbox(podSandboxID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "RemovePodSandbox")
	if err := r.popError("RemovePodSandbox"); err != nil {
		return err
	}

	// Remove the pod sandbox
	delete(r.Sandboxes, podSandboxID)

	return nil
}

func (r *FakeRuntimeService) PodSandboxStatus(podSandboxID string) (*runtimeapi.PodSandboxStatus, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "PodSandboxStatus")
	if err := r.popError("PodSandboxStatus"); err != nil {
		return nil, err
	}

	s, ok := r.Sandboxes[podSandboxID]
	if !ok {
		return nil, fmt.Errorf("pod sandbox %q not found", podSandboxID)
	}

	status := s.PodSandboxStatus
	return &status, nil
}

func (r *FakeRuntimeService) ListPodSandbox(filter *runtimeapi.PodSandboxFilter) ([]*runtimeapi.PodSandbox, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ListPodSandbox")
	if err := r.popError("ListPodSandbox"); err != nil {
		return nil, err
	}

	result := make([]*runtimeapi.PodSandbox, 0)
	for id, s := range r.Sandboxes {
		if filter != nil {
			if filter.Id != "" && filter.Id != id {
				continue
			}
			if filter.State != nil && filter.GetState().State != s.State {
				continue
			}
			if filter.LabelSelector != nil && !filterInLabels(filter.LabelSelector, s.GetLabels()) {
				continue
			}
		}

		result = append(result, &runtimeapi.PodSandbox{
			Id:             s.Id,
			Metadata:       s.Metadata,
			State:          s.State,
			CreatedAt:      s.CreatedAt,
			Labels:         s.Labels,
			Annotations:    s.Annotations,
			RuntimeHandler: s.RuntimeHandler,
		})
	}

	return result, nil
}

func (r *FakeRuntimeService) PortForward(*runtimeapi.PortForwardRequest) (*runtimeapi.PortForwardResponse, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "PortForward")
	if err := r.popError("PortForward"); err != nil {
		return nil, err
	}

	return &runtimeapi.PortForwardResponse{}, nil
}

func (r *FakeRuntimeService) CreateContainer(podSandboxID string, config *runtimeapi.ContainerConfig, sandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "CreateContainer")
	if err := r.popError("CreateContainer"); err != nil {
		return "", err
	}

	// ContainerID should be randomized for real container runtime, but here just use
	// fixed BuildContainerName() for easily making fake containers.
	containerID := BuildContainerName(config.Metadata, podSandboxID)
	createdAt := time.Now().UnixNano()
	createdState := runtimeapi.ContainerState_CONTAINER_CREATED
	imageRef := config.Image.Image
	r.Containers[containerID] = &FakeContainer{
		ContainerStatus: runtimeapi.ContainerStatus{
			Id:          containerID,
			Metadata:    config.Metadata,
			Image:       config.Image,
			ImageRef:    imageRef,
			CreatedAt:   createdAt,
			State:       createdState,
			Labels:      config.Labels,
			Annotations: config.Annotations,
		},
		SandboxID:      podSandboxID,
		LinuxResources: config.GetLinux().GetResources(),
	}

	return containerID, nil
}

func (r *FakeRuntimeService) StartContainer(containerID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "StartContainer")
	if err := r.popError("StartContainer"); err != nil {
		return err
	}

	c, ok := r.Containers[containerID]
	if !ok {
		return fmt.Errorf("container %s not found", containerID)
	}

	// Set container to running.
	c.State = runtimeapi.ContainerState_CONTAINER_RUNNING
	c.StartedAt = time.Now().UnixNano()

	return nil
}

func (r *FakeRuntimeService) StopContainer(containerID string, timeout int64) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "StopContainer")
	if err := r.popError("StopContainer"); err != nil {
		return err
	}

	c, ok := r.Containers[containerID]
	if !ok {
		return fmt.Errorf("container %q not found", containerID)
	}

	// Set container to exited state.
	finishedAt := time.Now().UnixNano()
	exitedState := runtimeapi.ContainerState_CONTAINER_EXITED
	c.State = exitedState
	c.FinishedAt = finishedAt

	return nil
}

func (r *FakeRuntimeService) RemoveContainer(containerID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "RemoveContainer")
	if err := r.popError("RemoveContainer"); err != nil {
		return err
	}

	// Remove the container
	delete(r.Containers, containerID)

	return nil
}

func (r *FakeRuntimeService) ListContainers(filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ListContainers")
	if err := r.popError("ListContainers"); err != nil {
		return nil, err
	}

	result := make([]*runtimeapi.Container, 0)
	for _, s := range r.Containers {
		if filter != nil {
			if filter.Id != "" && filter.Id != s.Id {
				continue
			}
			if filter.PodSandboxId != "" && filter.PodSandboxId != s.SandboxID {
				continue
			}
			if filter.State != nil && filter.GetState().State != s.State {
				continue
			}
			if filter.LabelSelector != nil && !filterInLabels(filter.LabelSelector, s.GetLabels()) {
				continue
			}
		}

		result = append(result, &runtimeapi.Container{
			Id:           s.Id,
			CreatedAt:    s.CreatedAt,
			PodSandboxId: s.SandboxID,
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

func (r *FakeRuntimeService) ContainerStatus(containerID string) (*runtimeapi.ContainerStatus, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ContainerStatus")
	if err := r.popError("ContainerStatus"); err != nil {
		return nil, err
	}

	c, ok := r.Containers[containerID]
	if !ok {
		return nil, fmt.Errorf("container %q not found", containerID)
	}

	status := c.ContainerStatus
	return &status, nil
}

func (r *FakeRuntimeService) UpdateContainerResources(string, *runtimeapi.LinuxContainerResources) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "UpdateContainerResources")
	return r.popError("UpdateContainerResources")
}

func (r *FakeRuntimeService) ExecSync(containerID string, cmd []string, timeout time.Duration) (stdout []byte, stderr []byte, err error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ExecSync")
	err = r.popError("ExecSync")
	return
}

func (r *FakeRuntimeService) Exec(*runtimeapi.ExecRequest) (*runtimeapi.ExecResponse, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "Exec")
	if err := r.popError("Exec"); err != nil {
		return nil, err
	}

	return &runtimeapi.ExecResponse{}, nil
}

func (r *FakeRuntimeService) Attach(req *runtimeapi.AttachRequest) (*runtimeapi.AttachResponse, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "Attach")
	if err := r.popError("Attach"); err != nil {
		return nil, err
	}

	return &runtimeapi.AttachResponse{}, nil
}

func (r *FakeRuntimeService) UpdateRuntimeConfig(runtimeCOnfig *runtimeapi.RuntimeConfig) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "UpdateRuntimeConfig")
	return r.popError("UpdateRuntimeConfig")
}

func (r *FakeRuntimeService) SetFakeContainerStats(containerStats []*runtimeapi.ContainerStats) {
	r.Lock()
	defer r.Unlock()

	r.FakeContainerStats = make(map[string]*runtimeapi.ContainerStats)
	for _, s := range containerStats {
		r.FakeContainerStats[s.Attributes.Id] = s
	}
}

func (r *FakeRuntimeService) ContainerStats(containerID string) (*runtimeapi.ContainerStats, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ContainerStats")
	if err := r.popError("ContainerStats"); err != nil {
		return nil, err
	}

	s, found := r.FakeContainerStats[containerID]
	if !found {
		return nil, fmt.Errorf("no stats for container %q", containerID)
	}
	return s, nil
}

func (r *FakeRuntimeService) ListContainerStats(filter *runtimeapi.ContainerStatsFilter) ([]*runtimeapi.ContainerStats, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ListContainerStats")
	if err := r.popError("ListContainerStats"); err != nil {
		return nil, err
	}

	var result []*runtimeapi.ContainerStats
	for _, c := range r.Containers {
		if filter != nil {
			if filter.Id != "" && filter.Id != c.Id {
				continue
			}
			if filter.PodSandboxId != "" && filter.PodSandboxId != c.SandboxID {
				continue
			}
			if filter.LabelSelector != nil && !filterInLabels(filter.LabelSelector, c.GetLabels()) {
				continue
			}
		}
		s, found := r.FakeContainerStats[c.Id]
		if !found {
			continue
		}
		result = append(result, s)
	}

	return result, nil
}

func (r *FakeRuntimeService) ReopenContainerLog(containerID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ReopenContainerLog")

	if err := r.popError("ReopenContainerLog"); err != nil {
		return err
	}

	return nil
}
