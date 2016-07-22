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

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// CreatePodSandbox creates a pod-level sandbox.
// The definition of PodSandbox is at https://github.com/kubernetes/kubernetes/pull/25899
func (ds *dockerService) CreatePodSandbox(config *runtimeApi.PodSandboxConfig) (string, error) {
	return "", fmt.Errorf("Not implemented")
}

// StopPodSandbox stops the sandbox. If there are any running containers in the
// sandbox, they should be force terminated.
func (ds *dockerService) StopPodSandbox(podSandboxID string) error {
	return fmt.Errorf("Not implemented")
}

// DeletePodSandbox deletes the sandbox. If there are running containers in the
// sandbox, they should be forcibly deleted.
func (ds *dockerService) DeletePodSandbox(podSandboxID string) error {
	return fmt.Errorf("Not implemented")
}

// PodSandboxStatus returns the Status of the PodSandbox.
func (ds *dockerService) PodSandboxStatus(podSandboxID string) (*runtimeApi.PodSandboxStatus, error) {
	return nil, fmt.Errorf("Not implemented")
}

// ListPodSandbox returns a list of Sandbox.
func (ds *dockerService) ListPodSandbox(filter *runtimeApi.PodSandboxFilter) ([]*runtimeApi.PodSandbox, error) {
	return nil, fmt.Errorf("Not implemented")
}
