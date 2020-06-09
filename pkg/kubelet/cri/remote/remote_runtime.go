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
	"k8s.io/klog/v2"

	"k8s.io/component-base/logs/logreduction"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/cri/remote/util"
	utilexec "k8s.io/utils/exec"
)

// RemoteRuntimeService is a gRPC implementation of internalapi.RuntimeService.
type RemoteRuntimeService struct {
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
	klog.V(3).Infof("Connecting to runtime service %s", endpoint)
	addr, dialer, err := util.GetAddressAndDialer(endpoint)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), connectionTimeout)
	defer cancel()

	conn, err := grpc.DialContext(ctx, addr, grpc.WithInsecure(), grpc.WithContextDialer(dialer), grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxMsgSize)))
	if err != nil {
		klog.Errorf("Connect remote runtime %s failed: %v", addr, err)
		return nil, err
	}

	return &RemoteRuntimeService{
		timeout:       connectionTimeout,
		runtimeClient: runtimeapi.NewRuntimeServiceClient(conn),
		logReduction:  logreduction.NewLogReduction(identicalErrorDelay),
	}, nil
}

// Version returns the runtime name, runtime version and runtime API version.
func (r *RemoteRuntimeService) Version(apiVersion string) (*runtimeapi.VersionResponse, error) {
	klog.V(10).Infof("[RemoteRuntimeService] Version (apiVersion=%v, timeout=%v)", apiVersion, r.timeout)

	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	typedVersion, err := r.runtimeClient.Version(ctx, &runtimeapi.VersionRequest{
		Version: apiVersion,
	})
	if err != nil {
		klog.Errorf("Version from runtime service failed: %v", err)
		return nil, err
	}

	klog.V(10).Infof("[RemoteRuntimeService] Version Response (typedVersion=%v)", typedVersion)

	if typedVersion.Version == "" || typedVersion.RuntimeName == "" || typedVersion.RuntimeApiVersion == "" || typedVersion.RuntimeVersion == "" {
		return nil, fmt.Errorf("not all fields are set in VersionResponse (%q)", *typedVersion)
	}

	return typedVersion, err
}

// RunPodSandbox creates and starts a pod-level sandbox. Runtimes should ensure
// the sandbox is in ready state.
func (r *RemoteRuntimeService) RunPodSandbox(config *runtimeapi.PodSandboxConfig, runtimeHandler string) (string, error) {
	// Use 2 times longer timeout for sandbox operation (4 mins by default)
	// TODO: Make the pod sandbox timeout configurable.
	timeout := r.timeout * 2

	klog.V(10).Infof("[RemoteRuntimeService] RunPodSandbox (config=%v, runtimeHandler=%v, timeout=%v)", config, runtimeHandler, timeout)

	ctx, cancel := getContextWithTimeout(timeout)
	defer cancel()

	resp, err := r.runtimeClient.RunPodSandbox(ctx, &runtimeapi.RunPodSandboxRequest{
		Config:         config,
		RuntimeHandler: runtimeHandler,
	})
	if err != nil {
		klog.Errorf("RunPodSandbox from runtime service failed: %v", err)
		return "", err
	}

	if resp.PodSandboxId == "" {
		errorMessage := fmt.Sprintf("PodSandboxId is not set for sandbox %q", config.GetMetadata())
		klog.Errorf("RunPodSandbox failed: %s", errorMessage)
		return "", errors.New(errorMessage)
	}

	klog.V(10).Infof("[RemoteRuntimeService] RunPodSandbox Response (PodSandboxId=%v)", resp.PodSandboxId)

	return resp.PodSandboxId, nil
}

// StopPodSandbox stops the sandbox. If there are any running containers in the
// sandbox, they should be forced to termination.
func (r *RemoteRuntimeService) StopPodSandbox(podSandBoxID string) error {
	klog.V(10).Infof("[RemoteRuntimeService] StopPodSandbox (podSandboxID=%v, timeout=%v)", podSandBoxID, r.timeout)

	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.StopPodSandbox(ctx, &runtimeapi.StopPodSandboxRequest{
		PodSandboxId: podSandBoxID,
	})
	if err != nil {
		klog.Errorf("StopPodSandbox %q from runtime service failed: %v", podSandBoxID, err)
		return err
	}

	klog.V(10).Infof("[RemoteRuntimeService] StopPodSandbox Response (podSandboxID=%v)", podSandBoxID)

	return nil
}

// RemovePodSandbox removes the sandbox. If there are any containers in the
// sandbox, they should be forcibly removed.
func (r *RemoteRuntimeService) RemovePodSandbox(podSandBoxID string) error {
	klog.V(10).Infof("[RemoteRuntimeService] RemovePodSandbox (podSandboxID=%v, timeout=%v)", podSandBoxID, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.RemovePodSandbox(ctx, &runtimeapi.RemovePodSandboxRequest{
		PodSandboxId: podSandBoxID,
	})
	if err != nil {
		klog.Errorf("RemovePodSandbox %q from runtime service failed: %v", podSandBoxID, err)
		return err
	}

	klog.V(10).Infof("[RemoteRuntimeService] RemovePodSandbox Response (podSandboxID=%v)", podSandBoxID)

	return nil
}

// PodSandboxStatus returns the status of the PodSandbox.
func (r *RemoteRuntimeService) PodSandboxStatus(podSandBoxID string) (*runtimeapi.PodSandboxStatus, error) {
	klog.V(10).Infof("[RemoteRuntimeService] PodSandboxStatus (podSandboxID=%v, timeout=%v)", podSandBoxID, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.PodSandboxStatus(ctx, &runtimeapi.PodSandboxStatusRequest{
		PodSandboxId: podSandBoxID,
	})
	if err != nil {
		return nil, err
	}

	klog.V(10).Infof("[RemoteRuntimeService] PodSandboxStatus Response (podSandboxID=%v, status=%v)", podSandBoxID, resp.Status)

	if resp.Status != nil {
		if err := verifySandboxStatus(resp.Status); err != nil {
			return nil, err
		}
	}

	return resp.Status, nil
}

// ListPodSandbox returns a list of PodSandboxes.
func (r *RemoteRuntimeService) ListPodSandbox(filter *runtimeapi.PodSandboxFilter) ([]*runtimeapi.PodSandbox, error) {
	klog.V(10).Infof("[RemoteRuntimeService] ListPodSandbox (filter=%v, timeout=%v)", filter, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.ListPodSandbox(ctx, &runtimeapi.ListPodSandboxRequest{
		Filter: filter,
	})
	if err != nil {
		klog.Errorf("ListPodSandbox with filter %+v from runtime service failed: %v", filter, err)
		return nil, err
	}

	klog.V(10).Infof("[RemoteRuntimeService] ListPodSandbox Response (filter=%v, items=%v)", filter, resp.Items)

	return resp.Items, nil
}

// CreateContainer creates a new container in the specified PodSandbox.
func (r *RemoteRuntimeService) CreateContainer(podSandBoxID string, config *runtimeapi.ContainerConfig, sandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	klog.V(10).Infof("[RemoteRuntimeService] CreateContainer (podSandBoxID=%v, timeout=%v)", podSandBoxID, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.CreateContainer(ctx, &runtimeapi.CreateContainerRequest{
		PodSandboxId:  podSandBoxID,
		Config:        config,
		SandboxConfig: sandboxConfig,
	})
	if err != nil {
		klog.Errorf("CreateContainer in sandbox %q from runtime service failed: %v", podSandBoxID, err)
		return "", err
	}

	klog.V(10).Infof("[RemoteRuntimeService] CreateContainer (podSandBoxID=%v, ContainerId=%v)", podSandBoxID, resp.ContainerId)
	if resp.ContainerId == "" {
		errorMessage := fmt.Sprintf("ContainerId is not set for container %q", config.GetMetadata())
		klog.Errorf("CreateContainer failed: %s", errorMessage)
		return "", errors.New(errorMessage)
	}

	return resp.ContainerId, nil
}

// StartContainer starts the container.
func (r *RemoteRuntimeService) StartContainer(containerID string) error {
	klog.V(10).Infof("[RemoteRuntimeService] StartContainer (containerID=%v, timeout=%v)", containerID, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.StartContainer(ctx, &runtimeapi.StartContainerRequest{
		ContainerId: containerID,
	})
	if err != nil {
		klog.Errorf("StartContainer %q from runtime service failed: %v", containerID, err)
		return err
	}
	klog.V(10).Infof("[RemoteRuntimeService] StartContainer Response (containerID=%v)", containerID)

	return nil
}

// StopContainer stops a running container with a grace period (i.e., timeout).
func (r *RemoteRuntimeService) StopContainer(containerID string, timeout int64) error {
	klog.V(10).Infof("[RemoteRuntimeService] StopContainer (containerID=%v, timeout=%v)", containerID, timeout)
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
		klog.Errorf("StopContainer %q from runtime service failed: %v", containerID, err)
		return err
	}
	klog.V(10).Infof("[RemoteRuntimeService] StopContainer Response (containerID=%v)", containerID)

	return nil
}

// RemoveContainer removes the container. If the container is running, the container
// should be forced to removal.
func (r *RemoteRuntimeService) RemoveContainer(containerID string) error {
	klog.V(10).Infof("[RemoteRuntimeService] RemoveContainer (containerID=%v, timeout=%v)", containerID, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	r.logReduction.ClearID(containerID)
	_, err := r.runtimeClient.RemoveContainer(ctx, &runtimeapi.RemoveContainerRequest{
		ContainerId: containerID,
	})
	if err != nil {
		klog.Errorf("RemoveContainer %q from runtime service failed: %v", containerID, err)
		return err
	}
	klog.V(10).Infof("[RemoteRuntimeService] RemoveContainer Response (containerID=%v)", containerID)

	return nil
}

// ListContainers lists containers by filters.
func (r *RemoteRuntimeService) ListContainers(filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
	klog.V(10).Infof("[RemoteRuntimeService] ListContainers (filter=%v, timeout=%v)", filter, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.ListContainers(ctx, &runtimeapi.ListContainersRequest{
		Filter: filter,
	})
	if err != nil {
		klog.Errorf("ListContainers with filter %+v from runtime service failed: %v", filter, err)
		return nil, err
	}
	klog.V(10).Infof("[RemoteRuntimeService] ListContainers Response (filter=%v, containers=%v)", filter, resp.Containers)

	return resp.Containers, nil
}

// ContainerStatus returns the container status.
func (r *RemoteRuntimeService) ContainerStatus(containerID string) (*runtimeapi.ContainerStatus, error) {
	klog.V(10).Infof("[RemoteRuntimeService] ContainerStatus (containerID=%v, timeout=%v)", containerID, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.ContainerStatus(ctx, &runtimeapi.ContainerStatusRequest{
		ContainerId: containerID,
	})
	if err != nil {
		// Don't spam the log with endless messages about the same failure.
		if r.logReduction.ShouldMessageBePrinted(err.Error(), containerID) {
			klog.Errorf("ContainerStatus %q from runtime service failed: %v", containerID, err)
		}
		return nil, err
	}
	r.logReduction.ClearID(containerID)
	klog.V(10).Infof("[RemoteRuntimeService] ContainerStatus Response (containerID=%v, status=%v)", containerID, resp.Status)

	if resp.Status != nil {
		if err := verifyContainerStatus(resp.Status); err != nil {
			klog.Errorf("ContainerStatus of %q failed: %v", containerID, err)
			return nil, err
		}
	}

	return resp.Status, nil
}

// UpdateContainerResources updates a containers resource config
func (r *RemoteRuntimeService) UpdateContainerResources(containerID string, resources *runtimeapi.LinuxContainerResources) error {
	klog.V(10).Infof("[RemoteRuntimeService] UpdateContainerResources (containerID=%v, timeout=%v)", containerID, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.UpdateContainerResources(ctx, &runtimeapi.UpdateContainerResourcesRequest{
		ContainerId: containerID,
		Linux:       resources,
	})
	if err != nil {
		klog.Errorf("UpdateContainerResources %q from runtime service failed: %v", containerID, err)
		return err
	}
	klog.V(10).Infof("[RemoteRuntimeService] UpdateContainerResources Response (containerID=%v)", containerID)

	return nil
}

// ExecSync executes a command in the container, and returns the stdout output.
// If command exits with a non-zero exit code, an error is returned.
func (r *RemoteRuntimeService) ExecSync(containerID string, cmd []string, timeout time.Duration) (stdout []byte, stderr []byte, err error) {
	klog.V(10).Infof("[RemoteRuntimeService] ExecSync (containerID=%v, timeout=%v)", containerID, timeout)
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
		klog.Errorf("ExecSync %s '%s' from runtime service failed: %v", containerID, strings.Join(cmd, " "), err)
		return nil, nil, err
	}

	klog.V(10).Infof("[RemoteRuntimeService] ExecSync Response (containerID=%v, ExitCode=%v)", containerID, resp.ExitCode)
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
func (r *RemoteRuntimeService) Exec(req *runtimeapi.ExecRequest) (*runtimeapi.ExecResponse, error) {
	klog.V(10).Infof("[RemoteRuntimeService] Exec (timeout=%v)", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.Exec(ctx, req)
	if err != nil {
		klog.Errorf("Exec %s '%s' from runtime service failed: %v", req.ContainerId, strings.Join(req.Cmd, " "), err)
		return nil, err
	}
	klog.V(10).Info("[RemoteRuntimeService] Exec Response")

	if resp.Url == "" {
		errorMessage := "URL is not set"
		klog.Errorf("Exec failed: %s", errorMessage)
		return nil, errors.New(errorMessage)
	}

	return resp, nil
}

// Attach prepares a streaming endpoint to attach to a running container, and returns the address.
func (r *RemoteRuntimeService) Attach(req *runtimeapi.AttachRequest) (*runtimeapi.AttachResponse, error) {
	klog.V(10).Infof("[RemoteRuntimeService] Attach (containerId=%v, timeout=%v)", req.ContainerId, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.Attach(ctx, req)
	if err != nil {
		klog.Errorf("Attach %s from runtime service failed: %v", req.ContainerId, err)
		return nil, err
	}
	klog.V(10).Infof("[RemoteRuntimeService] Attach Response (containerId=%v)", req.ContainerId)

	if resp.Url == "" {
		errorMessage := "URL is not set"
		klog.Errorf("Attach failed: %s", errorMessage)
		return nil, errors.New(errorMessage)
	}
	return resp, nil
}

// PortForward prepares a streaming endpoint to forward ports from a PodSandbox, and returns the address.
func (r *RemoteRuntimeService) PortForward(req *runtimeapi.PortForwardRequest) (*runtimeapi.PortForwardResponse, error) {
	klog.V(10).Infof("[RemoteRuntimeService] PortForward (podSandboxID=%v, port=%v, timeout=%v)", req.PodSandboxId, req.Port, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.PortForward(ctx, req)
	if err != nil {
		klog.Errorf("PortForward %s from runtime service failed: %v", req.PodSandboxId, err)
		return nil, err
	}
	klog.V(10).Infof("[RemoteRuntimeService] PortForward Response (podSandboxID=%v)", req.PodSandboxId)

	if resp.Url == "" {
		errorMessage := "URL is not set"
		klog.Errorf("PortForward failed: %s", errorMessage)
		return nil, errors.New(errorMessage)
	}

	return resp, nil
}

// UpdateRuntimeConfig updates the config of a runtime service. The only
// update payload currently supported is the pod CIDR assigned to a node,
// and the runtime service just proxies it down to the network plugin.
func (r *RemoteRuntimeService) UpdateRuntimeConfig(runtimeConfig *runtimeapi.RuntimeConfig) error {
	klog.V(10).Infof("[RemoteRuntimeService] UpdateRuntimeConfig (runtimeConfig=%v, timeout=%v)", runtimeConfig, r.timeout)
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
	klog.V(10).Infof("[RemoteRuntimeService] UpdateRuntimeConfig Response (runtimeConfig=%v)", runtimeConfig)

	return nil
}

// Status returns the status of the runtime.
func (r *RemoteRuntimeService) Status() (*runtimeapi.RuntimeStatus, error) {
	klog.V(10).Infof("[RemoteRuntimeService] Status (timeout=%v)", r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.Status(ctx, &runtimeapi.StatusRequest{})
	if err != nil {
		klog.Errorf("Status from runtime service failed: %v", err)
		return nil, err
	}

	klog.V(10).Infof("[RemoteRuntimeService] Status Response (status=%v)", resp.Status)

	if resp.Status == nil || len(resp.Status.Conditions) < 2 {
		errorMessage := "RuntimeReady or NetworkReady condition are not set"
		klog.Errorf("Status failed: %s", errorMessage)
		return nil, errors.New(errorMessage)
	}

	return resp.Status, nil
}

// ContainerStats returns the stats of the container.
func (r *RemoteRuntimeService) ContainerStats(containerID string) (*runtimeapi.ContainerStats, error) {
	klog.V(10).Infof("[RemoteRuntimeService] ContainerStats (containerID=%v, timeout=%v)", containerID, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.ContainerStats(ctx, &runtimeapi.ContainerStatsRequest{
		ContainerId: containerID,
	})
	if err != nil {
		if r.logReduction.ShouldMessageBePrinted(err.Error(), containerID) {
			klog.Errorf("ContainerStats %q from runtime service failed: %v", containerID, err)
		}
		return nil, err
	}
	r.logReduction.ClearID(containerID)
	klog.V(10).Infof("[RemoteRuntimeService] ContainerStats Response (containerID=%v, stats=%v)", containerID, resp.GetStats())

	return resp.GetStats(), nil
}

func (r *RemoteRuntimeService) ListContainerStats(filter *runtimeapi.ContainerStatsFilter) ([]*runtimeapi.ContainerStats, error) {
	klog.V(10).Infof("[RemoteRuntimeService] ListContainerStats (filter=%v)", filter)
	// Do not set timeout, because writable layer stats collection takes time.
	// TODO(random-liu): Should we assume runtime should cache the result, and set timeout here?
	ctx, cancel := getContextWithCancel()
	defer cancel()

	resp, err := r.runtimeClient.ListContainerStats(ctx, &runtimeapi.ListContainerStatsRequest{
		Filter: filter,
	})
	if err != nil {
		klog.Errorf("ListContainerStats with filter %+v from runtime service failed: %v", filter, err)
		return nil, err
	}
	klog.V(10).Infof("[RemoteRuntimeService] ListContainerStats Response (filter=%v, stats=%v)", filter, resp.GetStats())

	return resp.GetStats(), nil
}

func (r *RemoteRuntimeService) ReopenContainerLog(containerID string) error {
	klog.V(10).Infof("[RemoteRuntimeService] ReopenContainerLog (containerID=%v, timeout=%v)", containerID, r.timeout)
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.ReopenContainerLog(ctx, &runtimeapi.ReopenContainerLogRequest{ContainerId: containerID})
	if err != nil {
		klog.Errorf("ReopenContainerLog %q from runtime service failed: %v", containerID, err)
		return err
	}

	klog.V(10).Infof("[RemoteRuntimeService] ReopenContainerLog Response (containerID=%v)", containerID)
	return nil
}
