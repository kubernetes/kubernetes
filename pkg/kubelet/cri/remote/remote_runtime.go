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
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	"k8s.io/component-base/logs/logreduction"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/cri/remote/util"
	"k8s.io/kubernetes/pkg/probe/exec"
	utilexec "k8s.io/utils/exec"
)

// remoteRuntimeService is a gRPC implementation of internalapi.RuntimeService.
type remoteRuntimeService struct {
	timeout       time.Duration
	runtimeClient runtimeapi.RuntimeServiceClient
	// Cache last per-container error message to reduce log spam
	logReduction *logreduction.LogReduction
}

const (
	// How frequently to report identical errors
	identicalErrorDelay = 1 * time.Minute
)

// NewRemoteRuntimeService creates a new internalapi.RuntimeService.
func NewRemoteRuntimeService(endpoint string, connectionTimeout time.Duration) (internalapi.RuntimeService, error) {
	klog.V(3).InfoS("Connecting to runtime service", "endpoint", endpoint)
	addr, dialer, err := util.GetAddressAndDialer(endpoint)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), connectionTimeout)
	defer cancel()

	conn, err := grpc.DialContext(ctx, addr, grpc.WithInsecure(), grpc.WithContextDialer(dialer), grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxMsgSize)))
	if err != nil {
		klog.ErrorS(err, "Connect remote runtime failed", "address", addr)
		return nil, err
	}

	return &remoteRuntimeService{
		timeout:       connectionTimeout,
		runtimeClient: runtimeapi.NewRuntimeServiceClient(conn),
		logReduction:  logreduction.NewLogReduction(identicalErrorDelay),
	}, nil
}

// Version returns the runtime name, runtime version and runtime API version.
func (r *remoteRuntimeService) Version(apiVersion string) (*runtimeapi.VersionResponse, error) {
	klog.V(10).InfoS("[RemoteRuntimeService] Version", "apiVersion", apiVersion, "timeout", r.timeout)

	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	typedVersion, err := r.runtimeClient.Version(ctx, &runtimeapi.VersionRequest{
		Version: apiVersion,
	})
	if err != nil {
		klog.ErrorS(err, "Version from runtime service failed")
		return nil, err
	}

	klog.V(10).InfoS("[RemoteRuntimeService] Version Response", "apiVersion", typedVersion)

	if typedVersion.Version == "" || typedVersion.RuntimeName == "" || typedVersion.RuntimeApiVersion == "" || typedVersion.RuntimeVersion == "" {
		return nil, fmt.Errorf("not all fields are set in VersionResponse (%q)", *typedVersion)
	}

	return typedVersion, err
}

// RunPodSandbox creates and starts a pod-level sandbox. Runtimes should ensure
// the sandbox is in ready state.
func (r *remoteRuntimeService) RunPodSandbox(config *runtimeapi.PodSandboxConfig, runtimeHandler string) (string, error) {
	// Use 2 times longer timeout for sandbox operation (4 mins by default)
	// TODO: Make the pod sandbox timeout configurable.
	timeout := r.timeout * 2

	klog.V(10).InfoS("[RemoteRuntimeService] RunPodSandbox", "config", config, "runtimeHandler", runtimeHandler, "timeout", timeout)

	ctx, cancel := getContextWithTimeout(timeout)
	defer cancel()

	resp, err := r.runtimeClient.RunPodSandbox(ctx, &runtimeapi.RunPodSandboxRequest{
		Config:         config,
		RuntimeHandler: runtimeHandler,
	})
	if err != nil {
		klog.ErrorS(err, "RunPodSandbox from runtime service failed")
		return "", err
	}

	if resp.PodSandboxId == "" {
		errorMessage := fmt.Sprintf("PodSandboxId is not set for sandbox %q", config.GetMetadata())
		err := errors.New(errorMessage)
		klog.ErrorS(err, "RunPodSandbox failed")
		return "", err
	}

	klog.V(10).InfoS("[RemoteRuntimeService] RunPodSandbox Response", "podSandboxID", resp.PodSandboxId)

	return resp.PodSandboxId, nil
}

// StopPodSandbox stops the sandbox. If there are any running containers in the
// sandbox, they should be forced to termination.
func (r *remoteRuntimeService) StopPodSandbox(podSandBoxID string) error {
	klog.V(10).InfoS("[RemoteRuntimeService] StopPodSandbox", "podSandboxID", podSandBoxID, "timeout", r.timeout)

	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.StopPodSandbox(ctx, &runtimeapi.StopPodSandboxRequest{
		PodSandboxId: podSandBoxID,
	})
	if err != nil {
		klog.ErrorS(err, "StopPodSandbox from runtime service failed", "podSandboxID", podSandBoxID)
		return err
	}

	klog.V(10).InfoS("[RemoteRuntimeService] StopPodSandbox Response", "podSandboxID", podSandBoxID)

	return nil
}

// RemovePodSandbox removes the sandbox. If there are any containers in the
// sandbox, they should be forcibly removed.
func (r *remoteRuntimeService) RemovePodSandbox(podSandBoxID string) error {
	klog.V(10).InfoS("[RemoteRuntimeService] RemovePodSandbox", "podSandboxID", podSandBoxID, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.RemovePodSandbox(ctx, &runtimeapi.RemovePodSandboxRequest{
		PodSandboxId: podSandBoxID,
	})
	if err != nil {
		klog.ErrorS(err, "RemovePodSandbox from runtime service failed", "podSandboxID", podSandBoxID)
		return err
	}

	klog.V(10).InfoS("[RemoteRuntimeService] RemovePodSandbox Response", "podSandboxID", podSandBoxID)

	return nil
}

// PodSandboxStatus returns the status of the PodSandbox.
func (r *remoteRuntimeService) PodSandboxStatus(podSandBoxID string) (*runtimeapi.PodSandboxStatus, error) {
	klog.V(10).InfoS("[RemoteRuntimeService] PodSandboxStatus", "podSandboxID", podSandBoxID, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.PodSandboxStatus(ctx, &runtimeapi.PodSandboxStatusRequest{
		PodSandboxId: podSandBoxID,
	})
	if err != nil {
		return nil, err
	}

	klog.V(10).InfoS("[RemoteRuntimeService] PodSandboxStatus Response", "podSandboxID", podSandBoxID, "status", resp.Status)

	if resp.Status != nil {
		if err := verifySandboxStatus(resp.Status); err != nil {
			return nil, err
		}
	}

	return resp.Status, nil
}

// ListPodSandbox returns a list of PodSandboxes.
func (r *remoteRuntimeService) ListPodSandbox(filter *runtimeapi.PodSandboxFilter) ([]*runtimeapi.PodSandbox, error) {
	klog.V(10).InfoS("[RemoteRuntimeService] ListPodSandbox", "filter", filter, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.ListPodSandbox(ctx, &runtimeapi.ListPodSandboxRequest{
		Filter: filter,
	})
	if err != nil {
		klog.ErrorS(err, "ListPodSandbox with filter from runtime service failed", "filter", filter)
		return nil, err
	}

	klog.V(10).InfoS("[RemoteRuntimeService] ListPodSandbox Response", "filter", filter, "items", resp.Items)

	return resp.Items, nil
}

// CreateContainer creates a new container in the specified PodSandbox.
func (r *remoteRuntimeService) CreateContainer(podSandBoxID string, config *runtimeapi.ContainerConfig, sandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	klog.V(10).InfoS("[RemoteRuntimeService] CreateContainer", "podSandboxID", podSandBoxID, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.CreateContainer(ctx, &runtimeapi.CreateContainerRequest{
		PodSandboxId:  podSandBoxID,
		Config:        config,
		SandboxConfig: sandboxConfig,
	})
	if err != nil {
		klog.ErrorS(err, "CreateContainer in sandbox from runtime service failed", "podSandboxID", podSandBoxID)
		return "", err
	}

	klog.V(10).InfoS("[RemoteRuntimeService] CreateContainer", "podSandboxID", podSandBoxID, "containerID", resp.ContainerId)
	if resp.ContainerId == "" {
		errorMessage := fmt.Sprintf("ContainerId is not set for container %q", config.GetMetadata())
		err := errors.New(errorMessage)
		klog.ErrorS(err, "CreateContainer failed")
		return "", err
	}

	return resp.ContainerId, nil
}

// StartContainer starts the container.
func (r *remoteRuntimeService) StartContainer(containerID string) error {
	klog.V(10).InfoS("[RemoteRuntimeService] StartContainer", "containerID", containerID, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.StartContainer(ctx, &runtimeapi.StartContainerRequest{
		ContainerId: containerID,
	})
	if err != nil {
		klog.ErrorS(err, "StartContainer from runtime service failed", "containerID", containerID)
		return err
	}
	klog.V(10).InfoS("[RemoteRuntimeService] StartContainer Response", "containerID", containerID)

	return nil
}

// StopContainer stops a running container with a grace period (i.e., timeout).
func (r *remoteRuntimeService) StopContainer(containerID string, timeout int64) error {
	klog.V(10).InfoS("[RemoteRuntimeService] StopContainer", "containerID", containerID, "timeout", timeout)
	// Use timeout + default timeout (2 minutes) as timeout to leave extra time
	// for SIGKILL container and request latency.
	t := r.timeout + time.Duration(timeout)*time.Second
	ctx, cancel := getContextWithTimeout(t)
	defer cancel()

	r.logReduction.ClearID(containerID)
	_, err := r.runtimeClient.StopContainer(ctx, &runtimeapi.StopContainerRequest{
		ContainerId: containerID,
		Timeout:     timeout,
	})
	if err != nil {
		klog.ErrorS(err, "StopContainer from runtime service failed", "containerID", containerID)
		return err
	}
	klog.V(10).InfoS("[RemoteRuntimeService] StopContainer Response", "containerID", containerID)

	return nil
}

// RemoveContainer removes the container. If the container is running, the container
// should be forced to removal.
func (r *remoteRuntimeService) RemoveContainer(containerID string) error {
	klog.V(10).InfoS("[RemoteRuntimeService] RemoveContainer", "containerID", containerID, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	r.logReduction.ClearID(containerID)
	_, err := r.runtimeClient.RemoveContainer(ctx, &runtimeapi.RemoveContainerRequest{
		ContainerId: containerID,
	})
	if err != nil {
		klog.ErrorS(err, "RemoveContainer from runtime service failed", "containerID", containerID)
		return err
	}
	klog.V(10).InfoS("[RemoteRuntimeService] RemoveContainer Response", "containerID", containerID)

	return nil
}

// ListContainers lists containers by filters.
func (r *remoteRuntimeService) ListContainers(filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
	klog.V(10).InfoS("[RemoteRuntimeService] ListContainers", "filter", filter, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.ListContainers(ctx, &runtimeapi.ListContainersRequest{
		Filter: filter,
	})
	if err != nil {
		klog.ErrorS(err, "ListContainers with filter from runtime service failed", "filter", filter)
		return nil, err
	}
	klog.V(10).InfoS("[RemoteRuntimeService] ListContainers Response", "filter", filter, "containers", resp.Containers)

	return resp.Containers, nil
}

// ContainerStatus returns the container status.
func (r *remoteRuntimeService) ContainerStatus(containerID string) (*runtimeapi.ContainerStatus, error) {
	klog.V(10).InfoS("[RemoteRuntimeService] ContainerStatus", "containerID", containerID, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.ContainerStatus(ctx, &runtimeapi.ContainerStatusRequest{
		ContainerId: containerID,
	})
	if err != nil {
		// Don't spam the log with endless messages about the same failure.
		if r.logReduction.ShouldMessageBePrinted(err.Error(), containerID) {
			klog.ErrorS(err, "ContainerStatus from runtime service failed", "containerID", containerID)
		}
		return nil, err
	}
	r.logReduction.ClearID(containerID)
	klog.V(10).InfoS("[RemoteRuntimeService] ContainerStatus Response", "containerID", containerID, "status", resp.Status)

	if resp.Status != nil {
		if err := verifyContainerStatus(resp.Status); err != nil {
			klog.ErrorS(err, "verify ContainerStatus failed", "containerID", containerID)
			return nil, err
		}
	}

	return resp.Status, nil
}

// UpdateContainerResources updates a containers resource config
func (r *remoteRuntimeService) UpdateContainerResources(containerID string, resources *runtimeapi.LinuxContainerResources) error {
	klog.V(10).InfoS("[RemoteRuntimeService] UpdateContainerResources", "containerID", containerID, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.UpdateContainerResources(ctx, &runtimeapi.UpdateContainerResourcesRequest{
		ContainerId: containerID,
		Linux:       resources,
	})
	if err != nil {
		klog.ErrorS(err, "UpdateContainerResources from runtime service failed", "containerID", containerID)
		return err
	}
	klog.V(10).InfoS("[RemoteRuntimeService] UpdateContainerResources Response", "containerID", containerID)

	return nil
}

// ExecSync executes a command in the container, and returns the stdout output.
// If command exits with a non-zero exit code, an error is returned.
func (r *remoteRuntimeService) ExecSync(containerID string, cmd []string, timeout time.Duration) (stdout []byte, stderr []byte, err error) {
	klog.V(10).InfoS("[RemoteRuntimeService] ExecSync", "containerID", containerID, "timeout", timeout)
	// Do not set timeout when timeout is 0.
	var ctx context.Context
	var cancel context.CancelFunc
	if timeout != 0 {
		// Use timeout + default timeout (2 minutes) as timeout to leave some time for
		// the runtime to do cleanup.
		ctx, cancel = getContextWithTimeout(r.timeout + timeout)
	} else {
		ctx, cancel = getContextWithCancel()
	}
	defer cancel()

	timeoutSeconds := int64(timeout.Seconds())
	req := &runtimeapi.ExecSyncRequest{
		ContainerId: containerID,
		Cmd:         cmd,
		Timeout:     timeoutSeconds,
	}
	resp, err := r.runtimeClient.ExecSync(ctx, req)
	if err != nil {
		klog.ErrorS(err, "ExecSync cmd from runtime service failed", "containerID", containerID, "cmd", cmd)

		// interpret DeadlineExceeded gRPC errors as timedout probes
		if status.Code(err) == codes.DeadlineExceeded {
			err = exec.NewTimeoutError(fmt.Errorf("command %q timed out", strings.Join(cmd, " ")), timeout)
		}

		return nil, nil, err
	}

	klog.V(10).InfoS("[RemoteRuntimeService] ExecSync Response", "containerID", containerID, "exitCode", resp.ExitCode)
	err = nil
	if resp.ExitCode != 0 {
		err = utilexec.CodeExitError{
			Err:  fmt.Errorf("command '%s' exited with %d: %s", strings.Join(cmd, " "), resp.ExitCode, resp.Stderr),
			Code: int(resp.ExitCode),
		}
	}

	return resp.Stdout, resp.Stderr, err
}

// Exec prepares a streaming endpoint to execute a command in the container, and returns the address.
func (r *remoteRuntimeService) Exec(req *runtimeapi.ExecRequest) (*runtimeapi.ExecResponse, error) {
	klog.V(10).InfoS("[RemoteRuntimeService] Exec", "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.Exec(ctx, req)
	if err != nil {
		klog.ErrorS(err, "Exec cmd from runtime service failed", "containerID", req.ContainerId, "cmd", req.Cmd)
		return nil, err
	}
	klog.V(10).InfoS("[RemoteRuntimeService] Exec Response")

	if resp.Url == "" {
		errorMessage := "URL is not set"
		err := errors.New(errorMessage)
		klog.ErrorS(err, "Exec failed")
		return nil, err
	}

	return resp, nil
}

// Attach prepares a streaming endpoint to attach to a running container, and returns the address.
func (r *remoteRuntimeService) Attach(req *runtimeapi.AttachRequest) (*runtimeapi.AttachResponse, error) {
	klog.V(10).InfoS("[RemoteRuntimeService] Attach", "containerID", req.ContainerId, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.Attach(ctx, req)
	if err != nil {
		klog.ErrorS(err, "Attach container from runtime service failed", "containerID", req.ContainerId)
		return nil, err
	}
	klog.V(10).InfoS("[RemoteRuntimeService] Attach Response", "containerID", req.ContainerId)

	if resp.Url == "" {
		errorMessage := "URL is not set"
		err := errors.New(errorMessage)
		klog.ErrorS(err, "Attach failed")
		return nil, err
	}
	return resp, nil
}

// PortForward prepares a streaming endpoint to forward ports from a PodSandbox, and returns the address.
func (r *remoteRuntimeService) PortForward(req *runtimeapi.PortForwardRequest) (*runtimeapi.PortForwardResponse, error) {
	klog.V(10).InfoS("[RemoteRuntimeService] PortForward", "podSandboxID", req.PodSandboxId, "port", req.Port, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.PortForward(ctx, req)
	if err != nil {
		klog.ErrorS(err, "PortForward from runtime service failed", "podSandboxID", req.PodSandboxId)
		return nil, err
	}
	klog.V(10).InfoS("[RemoteRuntimeService] PortForward Response", "podSandboxID", req.PodSandboxId)

	if resp.Url == "" {
		errorMessage := "URL is not set"
		err := errors.New(errorMessage)
		klog.ErrorS(err, "PortForward failed")
		return nil, err
	}

	return resp, nil
}

// UpdateRuntimeConfig updates the config of a runtime service. The only
// update payload currently supported is the pod CIDR assigned to a node,
// and the runtime service just proxies it down to the network plugin.
func (r *remoteRuntimeService) UpdateRuntimeConfig(runtimeConfig *runtimeapi.RuntimeConfig) error {
	klog.V(10).InfoS("[RemoteRuntimeService] UpdateRuntimeConfig", "runtimeConfig", runtimeConfig, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	// Response doesn't contain anything of interest. This translates to an
	// Event notification to the network plugin, which can't fail, so we're
	// really looking to surface destination unreachable.
	_, err := r.runtimeClient.UpdateRuntimeConfig(ctx, &runtimeapi.UpdateRuntimeConfigRequest{
		RuntimeConfig: runtimeConfig,
	})

	if err != nil {
		return err
	}
	klog.V(10).InfoS("[RemoteRuntimeService] UpdateRuntimeConfig Response", "runtimeConfig", runtimeConfig)

	return nil
}

// Status returns the status of the runtime.
func (r *remoteRuntimeService) Status() (*runtimeapi.RuntimeStatus, error) {
	klog.V(10).InfoS("[RemoteRuntimeService] Status", "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.Status(ctx, &runtimeapi.StatusRequest{})
	if err != nil {
		klog.ErrorS(err, "Status from runtime service failed")
		return nil, err
	}

	klog.V(10).InfoS("[RemoteRuntimeService] Status Response", "status", resp.Status)

	if resp.Status == nil || len(resp.Status.Conditions) < 2 {
		errorMessage := "RuntimeReady or NetworkReady condition are not set"
		err := errors.New(errorMessage)
		klog.ErrorS(err, "Status failed")
		return nil, err
	}

	return resp.Status, nil
}

// ContainerStats returns the stats of the container.
func (r *remoteRuntimeService) ContainerStats(containerID string) (*runtimeapi.ContainerStats, error) {
	klog.V(10).InfoS("[RemoteRuntimeService] ContainerStats", "containerID", containerID, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.ContainerStats(ctx, &runtimeapi.ContainerStatsRequest{
		ContainerId: containerID,
	})
	if err != nil {
		if r.logReduction.ShouldMessageBePrinted(err.Error(), containerID) {
			klog.ErrorS(err, "ContainerStats from runtime service failed", "containerID", containerID)
		}
		return nil, err
	}
	r.logReduction.ClearID(containerID)
	klog.V(10).InfoS("[RemoteRuntimeService] ContainerStats Response", "containerID", containerID, "stats", resp.GetStats())

	return resp.GetStats(), nil
}

// ListContainerStats returns the list of ContainerStats given the filter.
func (r *remoteRuntimeService) ListContainerStats(filter *runtimeapi.ContainerStatsFilter) ([]*runtimeapi.ContainerStats, error) {
	klog.V(10).InfoS("[RemoteRuntimeService] ListContainerStats", "filter", filter)
	// Do not set timeout, because writable layer stats collection takes time.
	// TODO(random-liu): Should we assume runtime should cache the result, and set timeout here?
	ctx, cancel := getContextWithCancel()
	defer cancel()

	resp, err := r.runtimeClient.ListContainerStats(ctx, &runtimeapi.ListContainerStatsRequest{
		Filter: filter,
	})
	if err != nil {
		klog.ErrorS(err, "ListContainerStats with filter from runtime service failed", "filter", filter)
		return nil, err
	}
	klog.V(10).InfoS("[RemoteRuntimeService] ListContainerStats Response", "filter", filter, "stats", resp.GetStats())

	return resp.GetStats(), nil
}

// ReopenContainerLog reopens the container log file.
func (r *remoteRuntimeService) ReopenContainerLog(containerID string) error {
	klog.V(10).InfoS("[RemoteRuntimeService] ReopenContainerLog", "containerID", containerID, "timeout", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.ReopenContainerLog(ctx, &runtimeapi.ReopenContainerLogRequest{ContainerId: containerID})
	if err != nil {
		klog.ErrorS(err, "ReopenContainerLog from runtime service failed", "containerID", containerID)
		return err
	}

	klog.V(10).InfoS("[RemoteRuntimeService] ReopenContainerLog Response", "containerID", containerID)
	return nil
}
