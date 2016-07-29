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
	},
	)
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
		glog.Errorf("StopPodSandbox from runtime service failed: %v", err)
		return err
	}

	return nil
}

// DeletePodSandbox deletes the sandbox. If there are any containers in the
// sandbox, they should be forced to deletion.
func (r *RemoteRuntimeService) DeletePodSandbox(podSandBoxID string) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.DeletePodSandbox(ctx, &runtimeApi.DeletePodSandboxRequest{
		PodSandboxId: &podSandBoxID,
	})
	if err != nil {
		glog.Errorf("DeletePodSandbox from runtime service failed: %v", err)
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
		glog.Errorf("PodSandboxStatus from runtime service failed: %v", err)
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
		glog.Errorf("ListPodSandbox from runtime service failed: %v", err)
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
		glog.Errorf("CreateContainer from runtime service failed: %v", err)
		return "", err
	}

	return resp.GetContainerId(), nil
}

// StartContainer starts the container.
func (r *RemoteRuntimeService) StartContainer(rawContainerID string) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.StartContainer(ctx, &runtimeApi.StartContainerRequest{
		ContainerId: &rawContainerID,
	})
	if err != nil {
		glog.Errorf("StartContainer from runtime service failed: %v", err)
		return err
	}

	return nil
}

// StopContainer stops a running container with a grace period (i.e., timeout).
func (r *RemoteRuntimeService) StopContainer(rawContainerID string, timeout int64) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.StopContainer(ctx, &runtimeApi.StopContainerRequest{
		ContainerId: &rawContainerID,
		Timeout:     &timeout,
	})
	if err != nil {
		glog.Errorf("StopContainer from runtime service failed: %v", err)
		return err
	}

	return nil
}

// RemoveContainer removes the container. If the container is running, the container
// should be forced to removal.
func (r *RemoteRuntimeService) RemoveContainer(rawContainerID string) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.runtimeClient.RemoveContainer(ctx, &runtimeApi.RemoveContainerRequest{
		ContainerId: &rawContainerID,
	})
	if err != nil {
		glog.Errorf("RemoveContainer from runtime service failed: %v", err)
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
		glog.Errorf("ListContainers from runtime service failed: %v", err)
		return nil, err
	}

	return resp.Containers, nil
}

// ContainerStatus returns the container status.
func (r *RemoteRuntimeService) ContainerStatus(rawContainerID string) (*runtimeApi.ContainerStatus, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.runtimeClient.ContainerStatus(ctx, &runtimeApi.ContainerStatusRequest{
		ContainerId: &rawContainerID,
	})
	if err != nil {
		glog.Errorf("ContainerStatus from runtime service failed: %v", err)
		return nil, err
	}

	return resp.Status, nil
}

// Exec executes a command in the container.
// TODO: support terminal resizing for exec, refer https://github.com/kubernetes/kubernetes/issues/29579.
func (r *RemoteRuntimeService) Exec(rawContainerID string, cmd []string, tty bool, stdin io.Reader, stdout, stderr io.WriteCloser) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	stream, err := r.runtimeClient.Exec(ctx)
	if err != nil {
		glog.Errorf("Get remote runtime client stream failed: %v", err)
		return err
	}

	request := &runtimeApi.ExecRequest{
		ContainerId: &rawContainerID,
		Cmd:         cmd,
		Tty:         &tty,
	}
	err = stream.Send(request)
	if err != nil {
		glog.Errorf("Send exec request to remote runtime failed: %v", err)
		return err
	}

	errChanOut := make(chan error, 1)
	errChanIn := make(chan error, 1)
	exit := make(chan bool)

	go func(stdout, stderr io.WriteCloser) {
		defer close(errChanOut)
		defer close(exit)

		for {
			resp, err := stream.Recv()
			if err != nil && err != io.EOF {
				errChanOut <- err
				return
			}

			if resp != nil && len(resp.Stdout) > 0 && stdout != nil {
				nw, err := stdout.Write(resp.Stdout)
				if err != nil {
					errChanOut <- err
					return
				}
				if nw != len(resp.Stdout) {
					errChanOut <- io.ErrShortWrite
					return
				}
				if err == io.EOF {
					break
				}
			}

			if resp != nil && len(resp.Stderr) > 0 && stderr != nil {
				nw, err := stderr.Write(resp.Stderr)
				if err != nil {
					errChanOut <- err
					return
				}
				if nw != len(resp.Stderr) {
					errChanOut <- io.ErrShortWrite
					return
				}
				if err == io.EOF {
					break
				}
			}
		}
	}(stdout, stderr)

	if stdin != nil {
		go func(stdin io.Reader) {
			defer close(errChanIn)
			buffer := make([]byte, 256)

			for {
				nr, err := stdin.Read(buffer)
				if nr > 0 {
					request.Stdin = buffer[:nr]
					err := stream.Send(request)
					if err != nil {
						errChanIn <- err
						return
					}
				}

				if err == io.EOF {
					break
				}

				if err != nil {
					errChanIn <- err
					return
				}
			}
		}(stdin)
	}

	<-exit
	select {
	case err = <-errChanIn:
		if err != nil {
			glog.Errorf("Exec send stream error: %v", err)
		}
		return err
	case err = <-errChanOut:
		if err != nil {
			glog.Errorf("Exec receive stream error: %v", err)
		}
		return err
	}
}
