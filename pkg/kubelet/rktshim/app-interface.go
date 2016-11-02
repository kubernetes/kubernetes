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
	"time"

	kubeletApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// Runtime provides an API for lifecycle, inspection and introspection
// operations in a blocking manner using the App level API provided by rkt.
type Runtime struct{}

// TODO(tmrts): Fill out the creation configuration fields.
type RuntimeConfig struct{}

// NewRuntime creates a container.Runtime instance using the Runtime.
func NewRuntime(RuntimeConfig) (kubeletApi.ContainerManager, error) {
	return &Runtime{}, nil
}

// CreateContainer creates an app inside the provided pod sandbox and returns the RawContainerID.
func (*Runtime) CreateContainer(string, *runtimeApi.ContainerConfig, *runtimeApi.PodSandboxConfig) (string, error) {
	panic("not implemented")
}

// StartContainer starts a created app.
func (*Runtime) StartContainer(string) error {
	panic("not implemented")
}

// StopContainer stops a running app with a grace period (i.e. timeout).
func (*Runtime) StopContainer(string, int64) error {
	panic("not implemented")
}

// RemoveContainer removes the app from a pod sandbox.
func (*Runtime) RemoveContainer(string) error {
	panic("not implemented")
}

// ListContainers lists out the apps residing inside the pod sandbox using the ContainerFilter.
func (*Runtime) ListContainers(*runtimeApi.ContainerFilter) ([]*runtimeApi.Container, error) {
	panic("not implemented")
}

// ContainerStatus returns the RawContainerStatus of an app inside the pod sandbox.
func (*Runtime) ContainerStatus(string) (*runtimeApi.ContainerStatus, error) {
	panic("not implemented")
}

// ExecSync executes a command in the container, and returns the stdout output.
// If command exits with a non-zero exit code, an error is returned.
func (*Runtime) ExecSync(containerID string, cmd []string, timeout time.Duration) (stdout []byte, stderr []byte, err error) {
	panic("not implemented")
}

// Exec prepares a streaming endpoint to execute a command in the container, and returns the address.
func (*Runtime) Exec(*runtimeApi.ExecRequest) (*runtimeApi.ExecResponse, error) {
	panic("not implemented")
}

// Attach prepares a streaming endpoint to attach to a running container, and returns the address.
func (*Runtime) Attach(req *runtimeApi.AttachRequest) (*runtimeApi.AttachResponse, error) {
	panic("not implemented")
}
