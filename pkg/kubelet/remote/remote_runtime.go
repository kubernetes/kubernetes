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
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"
	"google.golang.org/grpc"
	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
)

// RemoteRuntimeService is a gRPC implementation of internalApi.RuntimeService.
type RemoteRuntimeService struct {
	timeout       time.Duration
	runtimeClient runtimeApi.RuntimeServiceClient
}

// NewRemoteRuntimeService creates a new internalApi.RuntimeService.
func NewRemoteRuntimeService(addr string, connectionTimout time.Duration) (internalApi.RuntimeService, error) {
	glog.Infof("Connecting to runtime service %s", addr)
	conn, err := grpc.Dial(addr, grpc.WithInsecure(), grpc.WithTimeout(connectionTimout), grpc.WithDialer(dial))
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

// RunPodSandbox creates and starts a pod-level sandbox. Runtimes should ensure
// the sandbox is in ready state.
func (r *RemoteRuntimeService) RunPodSandbox(config *runtimeApi.PodSandboxConfig) (string, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.RunPodSandbox(ctx, &runtimeApi.RunPodSandboxRequest{
		Config: config,
	})
	if err != nil {
		glog.Errorf("RunPodSandbox from runtime service failed: %v", err)
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

// ExecSync executes a command in the container, and returns the stdout output.
// If command exits with a non-zero exit code, an error is returned.
func (r *RemoteRuntimeService) ExecSync(containerID string, cmd []string, timeout time.Duration) (stdout []byte, stderr []byte, err error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	timeoutSeconds := int64(timeout.Seconds())
	req := &runtimeApi.ExecSyncRequest{
		ContainerId: &containerID,
		Cmd:         cmd,
		Timeout:     &timeoutSeconds,
	}
	resp, err := r.runtimeClient.ExecSync(ctx, req)
	if err != nil {
		glog.Errorf("ExecSync %s '%s' from runtime service failed: %v", containerID, strings.Join(cmd, " "), err)
		return nil, nil, err
	}

	err = nil
	if resp.GetExitCode() != 0 {
		err = utilexec.CodeExitError{
			Err:  fmt.Errorf("command '%s' exited with %d: %s", strings.Join(cmd, " "), resp.GetExitCode(), resp.GetStderr()),
			Code: int(resp.GetExitCode()),
		}
	}

	return resp.GetStdout(), resp.GetStderr(), err
}

// Exec prepares a streaming endpoint to execute a command in the container, and returns the address.
func (r *RemoteRuntimeService) Exec(req *runtimeApi.ExecRequest) (*runtimeApi.ExecResponse, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.Exec(ctx, req)
	if err != nil {
		glog.Errorf("Exec %s '%s' from runtime service failed: %v", req.GetContainerId(), strings.Join(req.GetCmd(), " "), err)
		return nil, err
	}

	return resp, nil
}

// Attach prepares a streaming endpoint to attach to a running container, and returns the address.
func (r *RemoteRuntimeService) Attach(req *runtimeApi.AttachRequest) (*runtimeApi.AttachResponse, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.Attach(ctx, req)
	if err != nil {
		glog.Errorf("Attach %s from runtime service failed: %v", req.GetContainerId(), err)
		return nil, err
	}

	return resp, nil
}

// PortForward prepares a streaming endpoint to forward ports from a PodSandbox, and returns the address.
func (r *RemoteRuntimeService) PortForward(req *runtimeApi.PortForwardRequest) (*runtimeApi.PortForwardResponse, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.PortForward(ctx, req)
	if err != nil {
		glog.Errorf("PortForward %s from runtime service failed: %v", req.GetPodSandboxId(), err)
		return nil, err
	}

	return resp, nil
}

// UpdateRuntimeConfig updates the config of a runtime service. The only
// update payload currently supported is the pod CIDR assigned to a node,
// and the runtime service just proxies it down to the network plugin.
func (r *RemoteRuntimeService) UpdateRuntimeConfig(runtimeConfig *runtimeApi.RuntimeConfig) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	// Response doesn't contain anything of interest. This translates to an
	// Event notification to the network plugin, which can't fail, so we're
	// really looking to surface destination unreachable.
	_, err := r.runtimeClient.UpdateRuntimeConfig(ctx, &runtimeApi.UpdateRuntimeConfigRequest{
		RuntimeConfig: runtimeConfig,
	})

	if err != nil {
		return err
	}

	return nil
}

// Status returns the status of the runtime.
func (r *RemoteRuntimeService) Status() (*runtimeApi.RuntimeStatus, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.Status(ctx, &runtimeApi.StatusRequest{})
	if err != nil {
		glog.Errorf("Status from runtime service failed: %v", err)
		return nil, err
	}

	return resp.Status, nil
}
