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
	"time"

	"golang.org/x/net/context"

	internalapi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/containerdshim"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
)

// ContainerdService is the interface implement CRI remote service server.
type ContainerdService interface {
	runtimeapi.RuntimeServiceServer
	runtimeapi.ImageServiceServer
}

// containerdService uses containerdshim service to implement ContainerdService.
// Notice that the contexts in the functions are not used now.
// TODO(random-liu): Change the containerdshim service to support context, and implement
// internal services and remote services with the containerdshim service.
type containerdService struct {
	runtimeService internalapi.RuntimeService
	imageService   internalapi.ImageManagerService
}

func NewContainerdService(s containerdshim.ContainerdService) ContainerdService {
	return &containerdService{runtimeService: s, imageService: s}
}

func (c *containerdService) Version(ctx context.Context, r *runtimeapi.VersionRequest) (*runtimeapi.VersionResponse, error) {
	return c.runtimeService.Version(r.Version)
}

func (c *containerdService) Status(ctx context.Context, r *runtimeapi.StatusRequest) (*runtimeapi.StatusResponse, error) {
	status, err := c.runtimeService.Status()
	if err != nil {
		return nil, err
	}
	return &runtimeapi.StatusResponse{Status: status}, nil
}

func (c *containerdService) RunPodSandbox(ctx context.Context, r *runtimeapi.RunPodSandboxRequest) (*runtimeapi.RunPodSandboxResponse, error) {
	podSandboxId, err := c.runtimeService.RunPodSandbox(r.GetConfig())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.RunPodSandboxResponse{PodSandboxId: podSandboxId}, nil
}

func (c *containerdService) StopPodSandbox(ctx context.Context, r *runtimeapi.StopPodSandboxRequest) (*runtimeapi.StopPodSandboxResponse, error) {
	err := c.runtimeService.StopPodSandbox(r.PodSandboxId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.StopPodSandboxResponse{}, nil
}

func (c *containerdService) RemovePodSandbox(ctx context.Context, r *runtimeapi.RemovePodSandboxRequest) (*runtimeapi.RemovePodSandboxResponse, error) {
	err := c.runtimeService.RemovePodSandbox(r.PodSandboxId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.RemovePodSandboxResponse{}, nil
}

func (c *containerdService) PodSandboxStatus(ctx context.Context, r *runtimeapi.PodSandboxStatusRequest) (*runtimeapi.PodSandboxStatusResponse, error) {
	podSandboxStatus, err := c.runtimeService.PodSandboxStatus(r.PodSandboxId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.PodSandboxStatusResponse{Status: podSandboxStatus}, nil
}

func (c *containerdService) ListPodSandbox(ctx context.Context, r *runtimeapi.ListPodSandboxRequest) (*runtimeapi.ListPodSandboxResponse, error) {
	items, err := c.runtimeService.ListPodSandbox(r.GetFilter())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.ListPodSandboxResponse{Items: items}, nil
}

func (c *containerdService) CreateContainer(ctx context.Context, r *runtimeapi.CreateContainerRequest) (*runtimeapi.CreateContainerResponse, error) {
	containerId, err := c.runtimeService.CreateContainer(r.PodSandboxId, r.GetConfig(), r.GetSandboxConfig())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.CreateContainerResponse{ContainerId: containerId}, nil
}

func (c *containerdService) StartContainer(ctx context.Context, r *runtimeapi.StartContainerRequest) (*runtimeapi.StartContainerResponse, error) {
	err := c.runtimeService.StartContainer(r.ContainerId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.StartContainerResponse{}, nil
}

func (c *containerdService) StopContainer(ctx context.Context, r *runtimeapi.StopContainerRequest) (*runtimeapi.StopContainerResponse, error) {
	err := c.runtimeService.StopContainer(r.ContainerId, r.Timeout)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.StopContainerResponse{}, nil
}

func (c *containerdService) RemoveContainer(ctx context.Context, r *runtimeapi.RemoveContainerRequest) (*runtimeapi.RemoveContainerResponse, error) {
	err := c.runtimeService.RemoveContainer(r.ContainerId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.RemoveContainerResponse{}, nil
}

func (c *containerdService) ListContainers(ctx context.Context, r *runtimeapi.ListContainersRequest) (*runtimeapi.ListContainersResponse, error) {
	containers, err := c.runtimeService.ListContainers(r.GetFilter())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.ListContainersResponse{Containers: containers}, nil
}

func (c *containerdService) ContainerStatus(ctx context.Context, r *runtimeapi.ContainerStatusRequest) (*runtimeapi.ContainerStatusResponse, error) {
	status, err := c.runtimeService.ContainerStatus(r.ContainerId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.ContainerStatusResponse{Status: status}, nil
}

func (c *containerdService) ExecSync(ctx context.Context, r *runtimeapi.ExecSyncRequest) (*runtimeapi.ExecSyncResponse, error) {
	stdout, stderr, err := c.runtimeService.ExecSync(r.ContainerId, r.Cmd, time.Duration(r.Timeout)*time.Second)
	var exitCode int32
	if err != nil {
		exitError, ok := err.(utilexec.ExitError)
		if !ok {
			return nil, err
		}
		exitCode = int32(exitError.ExitStatus())
	}
	return &runtimeapi.ExecSyncResponse{
		Stdout:   stdout,
		Stderr:   stderr,
		ExitCode: exitCode,
	}, nil
}

func (c *containerdService) Exec(ctx context.Context, r *runtimeapi.ExecRequest) (*runtimeapi.ExecResponse, error) {
	return c.runtimeService.Exec(r)
}

func (c *containerdService) Attach(ctx context.Context, r *runtimeapi.AttachRequest) (*runtimeapi.AttachResponse, error) {
	return c.runtimeService.Attach(r)
}

func (c *containerdService) PortForward(ctx context.Context, r *runtimeapi.PortForwardRequest) (*runtimeapi.PortForwardResponse, error) {
	return c.runtimeService.PortForward(r)
}

func (c *containerdService) UpdateRuntimeConfig(ctx context.Context, r *runtimeapi.UpdateRuntimeConfigRequest) (*runtimeapi.UpdateRuntimeConfigResponse, error) {
	err := c.runtimeService.UpdateRuntimeConfig(r.GetRuntimeConfig())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.UpdateRuntimeConfigResponse{}, nil
}

func (c *containerdService) ListImages(ctx context.Context, r *runtimeapi.ListImagesRequest) (*runtimeapi.ListImagesResponse, error) {
	images, err := c.imageService.ListImages(r.GetFilter())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.ListImagesResponse{Images: images}, nil
}

func (c *containerdService) ImageStatus(ctx context.Context, r *runtimeapi.ImageStatusRequest) (*runtimeapi.ImageStatusResponse, error) {
	image, err := c.imageService.ImageStatus(r.GetImage())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.ImageStatusResponse{Image: image}, nil
}

func (c *containerdService) PullImage(ctx context.Context, r *runtimeapi.PullImageRequest) (*runtimeapi.PullImageResponse, error) {
	image, err := c.imageService.PullImage(r.GetImage(), r.GetAuth())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.PullImageResponse{ImageRef: image}, nil
}

func (c *containerdService) RemoveImage(ctx context.Context, r *runtimeapi.RemoveImageRequest) (*runtimeapi.RemoveImageResponse, error) {
	err := c.imageService.RemoveImage(r.GetImage())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.RemoveImageResponse{}, nil
}
