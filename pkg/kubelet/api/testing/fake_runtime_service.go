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

	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

type fakeRuntimeService struct {
}

func NewFakeRuntimeService() internalApi.RuntimeService {
	return &fakeRuntimeService{}
}

func (r *fakeRuntimeService) Version(apiVersion string) (*runtimeApi.VersionResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (r *fakeRuntimeService) CreatePodSandbox(config *runtimeApi.PodSandboxConfig) (string, error) {
	return "", fmt.Errorf("not implemented")
}

func (r *fakeRuntimeService) StopPodSandbox(podSandboxID string) error {
	return fmt.Errorf("not implemented")
}

func (r *fakeRuntimeService) DeletePodSandbox(podSandboxID string) error {
	return fmt.Errorf("not implemented")
}

func (r *fakeRuntimeService) PodSandboxStatus(podSandboxID string) (*runtimeApi.PodSandboxStatus, error) {
	return nil, fmt.Errorf("not implemented")
}

func (r *fakeRuntimeService) ListPodSandbox(filter *runtimeApi.PodSandboxFilter) ([]*runtimeApi.PodSandbox, error) {
	return nil, fmt.Errorf("not implemented")
}

func (r *fakeRuntimeService) CreateContainer(podSandboxID string, config *runtimeApi.ContainerConfig, sandboxConfig *runtimeApi.PodSandboxConfig) (string, error) {
	return "", fmt.Errorf("not implemented")
}

func (r *fakeRuntimeService) StartContainer(rawContainerID string) error {
	return fmt.Errorf("not implemented")
}

func (r *fakeRuntimeService) StopContainer(rawContainerID string, timeout int64) error {
	return fmt.Errorf("not implemented")
}

func (r *fakeRuntimeService) RemoveContainer(rawContainerID string) error {
	return fmt.Errorf("not implemented")
}

func (r *fakeRuntimeService) ListContainers(filter *runtimeApi.ContainerFilter) ([]*runtimeApi.Container, error) {
	return nil, fmt.Errorf("not implemented")
}

func (r *fakeRuntimeService) ContainerStatus(rawContainerID string) (*runtimeApi.ContainerStatus, error) {
	return nil, fmt.Errorf("not implemented")
}

func (r *fakeRuntimeService) Exec(rawContainerID string, cmd []string, tty bool, stdin io.Reader, stdout, stderr io.WriteCloser) error {
	return fmt.Errorf("not implemented")
}
