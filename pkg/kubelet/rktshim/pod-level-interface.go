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

package rktshim

import (
	kubeletApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// PodSandboxManager provides basic operations to create/delete and examine
// the pod sandboxes in a blocking manner.
type PodSandboxManager struct{}

// TODO(tmrts): Fill the configuration struct fields.
type PodSandboxManagerConfig struct{}

// NewPodSandboxManager creates a PodSandboxManager.
func NewPodSandboxManager(PodSandboxManagerConfig) (kubeletApi.PodSandboxManager, error) {
	return &PodSandboxManager{}, nil
}

// RunPodSandbox creates and starts a pod sandbox given a pod sandbox configuration.
func (*PodSandboxManager) RunPodSandbox(*runtimeApi.PodSandboxConfig) (string, error) {
	panic("not implemented")
}

// StopPodSandbox stops a pod sandbox and the apps inside the sandbox.
func (*PodSandboxManager) StopPodSandbox(string) error {
	panic("not implemented")
}

// RemovePodSandbox deletes the pod sandbox and the apps inside the sandbox.
func (*PodSandboxManager) RemovePodSandbox(string) error {
	panic("not implemented")
}

// PodSandboxStatus queries the status of the pod sandbox.
func (*PodSandboxManager) PodSandboxStatus(string) (*runtimeApi.PodSandboxStatus, error) {
	panic("not implemented")
}

// ListPodSandbox lists existing sandboxes, filtered by the PodSandboxFilter.
func (*PodSandboxManager) ListPodSandbox(*runtimeApi.PodSandboxFilter) ([]*runtimeApi.PodSandbox, error) {
	panic("not implemented")
}

// PortForward prepares a streaming endpoint to forward ports from a PodSandbox, and returns the address.
func (*PodSandboxManager) PortForward(*runtimeApi.PortForwardRequest) (*runtimeApi.PortForwardResponse, error) {
	panic("not implemented")
}
