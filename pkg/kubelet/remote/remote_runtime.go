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

package remote

import (
	"io"
	"time"

	"fmt"
	"github.com/golang/glog"
	"google.golang.org/grpc"
	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// RemoteRuntimeService is a gRPC implementation of internalApi.RuntimeService.
type RemoteRuntimeService struct {
	timeout       time.Duration
	runtimeClient runtimeApi.RuntimeServiceClient
}

// NewRemoteRuntimeService creates a new internalApi.RuntimeService.
func NewRemoteRuntimeService(addr string, connectionTimout time.Duration) (internalApi.RuntimeService, error) {
	glog.V(3).Infof("Connecting to runtime service %s", addr)
	conn, err := grpc.Dial(addr, grpc.WithInsecure(), grpc.WithDialer(dial))
	if err != nil {
		glog.Errorf("Connect remote runtime %s failed: %v", addr, err)
		return nil, err
	}

	return &RemoteRuntimeService{
		timeout:       connectionTimout,
		runtimeClient: runtimeApi.NewRuntimeServiceClient(conn),
	}, nil
}

// Version returns the runtime name, runtime version and runtime API version.
func (r *RemoteRuntimeService) Version(apiVersion string) (*runtimeApi.VersionResponse, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	typedVersion, err := r.runtimeClient.Version(ctx, &runtimeApi.VersionRequest{
		Version: &apiVersion,
	})
	if err != nil {
		glog.Errorf("Version from runtime service failed: %v", err)
		return nil, err
	}

	return typedVersion, err
}

// CreatePodSandbox creates a pod-level sandbox.
func (r *RemoteRuntimeService) CreatePodSandbox(config *runtimeApi.PodSandboxConfig) (string, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.CreatePodSandbox(ctx, &runtimeApi.CreatePodSandboxRequest{
		Config: config,
	})
	if err != nil {
		glog.Errorf("CreatePodSandbox from runtime service failed: %v", err)
		return "", err
	}

	return resp.GetPodSandboxId(), nil
}

// StopPodSandbox stops the sandbox. If there are any running containers in the
// sandbox, they should be forced to termination.
func (r *RemoteRuntimeService) StopPodSandbox(podSandBoxID string) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.StopPodSandbox(ctx, &runtimeApi.StopPodSandboxRequest{
		PodSandboxId: &podSandBoxID,
	})
	if err != nil {
		glog.Errorf("StopPodSandbox %q from runtime service failed: %v", podSandBoxID, err)
		return err
	}

	return nil
}

// RemovePodSandbox removes the sandbox. If there are any containers in the
// sandbox, they should be forcibly removed.
func (r *RemoteRuntimeService) RemovePodSandbox(podSandBoxID string) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.RemovePodSandbox(ctx, &runtimeApi.RemovePodSandboxRequest{
		PodSandboxId: &podSandBoxID,
	})
	if err != nil {
		glog.Errorf("RemovePodSandbox %q from runtime service failed: %v", podSandBoxID, err)
		return err
	}

	return nil
}

// PodSandboxStatus returns the status of the PodSandbox.
func (r *RemoteRuntimeService) PodSandboxStatus(podSandBoxID string) (*runtimeApi.PodSandboxStatus, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.PodSandboxStatus(ctx, &runtimeApi.PodSandboxStatusRequest{
		PodSandboxId: &podSandBoxID,
	})
	if err != nil {
		glog.Errorf("PodSandboxStatus %q from runtime service failed: %v", podSandBoxID, err)
		return nil, err
	}

	return resp.Status, nil
}

// ListPodSandbox returns a list of PodSandboxes.
func (r *RemoteRuntimeService) ListPodSandbox(filter *runtimeApi.PodSandboxFilter) ([]*runtimeApi.PodSandbox, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.ListPodSandbox(ctx, &runtimeApi.ListPodSandboxRequest{
		Filter: filter,
	})
	if err != nil {
		glog.Errorf("ListPodSandbox with filter %q from runtime service failed: %v", filter, err)
		return nil, err
	}

	return resp.Items, nil
}

// CreateContainer creates a new container in the specified PodSandbox.
func (r *RemoteRuntimeService) CreateContainer(podSandBoxID string, config *runtimeApi.ContainerConfig, sandboxConfig *runtimeApi.PodSandboxConfig) (string, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.CreateContainer(ctx, &runtimeApi.CreateContainerRequest{
		PodSandboxId:  &podSandBoxID,
		Config:        config,
		SandboxConfig: sandboxConfig,
	})
	if err != nil {
		glog.Errorf("CreateContainer in sandbox %q from runtime service failed: %v", podSandBoxID, err)
		return "", err
	}

	return resp.GetContainerId(), nil
}

// StartContainer starts the container.
func (r *RemoteRuntimeService) StartContainer(containerID string) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.StartContainer(ctx, &runtimeApi.StartContainerRequest{
		ContainerId: &containerID,
	})
	if err != nil {
		glog.Errorf("StartContainer %q from runtime service failed: %v", containerID, err)
		return err
	}

	return nil
}

// StopContainer stops a running container with a grace period (i.e., timeout).
func (r *RemoteRuntimeService) StopContainer(containerID string, timeout int64) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.StopContainer(ctx, &runtimeApi.StopContainerRequest{
		ContainerId: &containerID,
		Timeout:     &timeout,
	})
	if err != nil {
		glog.Errorf("StopContainer %q from runtime service failed: %v", containerID, err)
		return err
	}

	return nil
}

// RemoveContainer removes the container. If the container is running, the container
// should be forced to removal.
func (r *RemoteRuntimeService) RemoveContainer(containerID string) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.RemoveContainer(ctx, &runtimeApi.RemoveContainerRequest{
		ContainerId: &containerID,
	})
	if err != nil {
		glog.Errorf("RemoveContainer %q from runtime service failed: %v", containerID, err)
		return err
	}

	return nil
}

// ListContainers lists containers by filters.
func (r *RemoteRuntimeService) ListContainers(filter *runtimeApi.ContainerFilter) ([]*runtimeApi.Container, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.ListContainers(ctx, &runtimeApi.ListContainersRequest{
		Filter: filter,
	})
	if err != nil {
		glog.Errorf("ListContainers with filter %q from runtime service failed: %v", filter, err)
		return nil, err
	}

	return resp.Containers, nil
}

// ContainerStatus returns the container status.
func (r *RemoteRuntimeService) ContainerStatus(containerID string) (*runtimeApi.ContainerStatus, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.ContainerStatus(ctx, &runtimeApi.ContainerStatusRequest{
		ContainerId: &containerID,
	})
	if err != nil {
		glog.Errorf("ContainerStatus %q from runtime service failed: %v", containerID, err)
		return nil, err
	}

	return resp.Status, nil
}

// Exec executes a command in the container.
// TODO: support terminal resizing for exec, refer https://github.com/kubernetes/kubernetes/issues/29579.
func (r *RemoteRuntimeService) Exec(containerID string, cmd []string, tty bool, stdin io.Reader, stdout, stderr io.WriteCloser) error {
	return fmt.Errorf("Not implemented")
}
