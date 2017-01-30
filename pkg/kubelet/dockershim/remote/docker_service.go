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
	"k8s.io/kubernetes/pkg/kubelet/dockershim"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
)

// DockerService is the interface implement CRI remote service server.
type DockerService interface {
	runtimeapi.RuntimeServiceServer
	runtimeapi.ImageServiceServer
}

// dockerService uses dockershim service to implement DockerService.
// Notice that the contexts in the functions are not used now.
// TODO(random-liu): Change the dockershim service to support context, and implement
// internal services and remote services with the dockershim service.
type dockerService struct {
	runtimeService internalapi.RuntimeService
	imageService   internalapi.ImageManagerService
}

func NewDockerService(s dockershim.DockerService) DockerService {
	return &dockerService{runtimeService: s, imageService: s}
}

func (d *dockerService) Version(ctx context.Context, r *runtimeapi.VersionRequest) (*runtimeapi.VersionResponse, error) {
	return d.runtimeService.Version(r.Version)
}

func (d *dockerService) Status(ctx context.Context, r *runtimeapi.StatusRequest) (*runtimeapi.StatusResponse, error) {
	status, err := d.runtimeService.Status()
	if err != nil {
		return nil, err
	}
	return &runtimeapi.StatusResponse{Status: status}, nil
}

func (d *dockerService) RunPodSandbox(ctx context.Context, r *runtimeapi.RunPodSandboxRequest) (*runtimeapi.RunPodSandboxResponse, error) {
	podSandboxId, err := d.runtimeService.RunPodSandbox(r.GetConfig())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.RunPodSandboxResponse{PodSandboxId: podSandboxId}, nil
}

func (d *dockerService) StopPodSandbox(ctx context.Context, r *runtimeapi.StopPodSandboxRequest) (*runtimeapi.StopPodSandboxResponse, error) {
	err := d.runtimeService.StopPodSandbox(r.PodSandboxId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.StopPodSandboxResponse{}, nil
}

func (d *dockerService) RemovePodSandbox(ctx context.Context, r *runtimeapi.RemovePodSandboxRequest) (*runtimeapi.RemovePodSandboxResponse, error) {
	err := d.runtimeService.RemovePodSandbox(r.PodSandboxId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.RemovePodSandboxResponse{}, nil
}

func (d *dockerService) PodSandboxStatus(ctx context.Context, r *runtimeapi.PodSandboxStatusRequest) (*runtimeapi.PodSandboxStatusResponse, error) {
	podSandboxStatus, err := d.runtimeService.PodSandboxStatus(r.PodSandboxId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.PodSandboxStatusResponse{Status: podSandboxStatus}, nil
}

func (d *dockerService) ListPodSandbox(ctx context.Context, r *runtimeapi.ListPodSandboxRequest) (*runtimeapi.ListPodSandboxResponse, error) {
	items, err := d.runtimeService.ListPodSandbox(r.GetFilter())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.ListPodSandboxResponse{Items: items}, nil
}

func (d *dockerService) CreateContainer(ctx context.Context, r *runtimeapi.CreateContainerRequest) (*runtimeapi.CreateContainerResponse, error) {
	containerId, err := d.runtimeService.CreateContainer(r.PodSandboxId, r.GetConfig(), r.GetSandboxConfig())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.CreateContainerResponse{ContainerId: containerId}, nil
}

func (d *dockerService) StartContainer(ctx context.Context, r *runtimeapi.StartContainerRequest) (*runtimeapi.StartContainerResponse, error) {
	err := d.runtimeService.StartContainer(r.ContainerId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.StartContainerResponse{}, nil
}

func (d *dockerService) StopContainer(ctx context.Context, r *runtimeapi.StopContainerRequest) (*runtimeapi.StopContainerResponse, error) {
	err := d.runtimeService.StopContainer(r.ContainerId, r.Timeout)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.StopContainerResponse{}, nil
}

func (d *dockerService) RemoveContainer(ctx context.Context, r *runtimeapi.RemoveContainerRequest) (*runtimeapi.RemoveContainerResponse, error) {
	err := d.runtimeService.RemoveContainer(r.ContainerId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.RemoveContainerResponse{}, nil
}

func (d *dockerService) ListContainers(ctx context.Context, r *runtimeapi.ListContainersRequest) (*runtimeapi.ListContainersResponse, error) {
	containers, err := d.runtimeService.ListContainers(r.GetFilter())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.ListContainersResponse{Containers: containers}, nil
}

func (d *dockerService) ContainerStatus(ctx context.Context, r *runtimeapi.ContainerStatusRequest) (*runtimeapi.ContainerStatusResponse, error) {
	status, err := d.runtimeService.ContainerStatus(r.ContainerId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.ContainerStatusResponse{Status: status}, nil
}

func (d *dockerService) ExecSync(ctx context.Context, r *runtimeapi.ExecSyncRequest) (*runtimeapi.ExecSyncResponse, error) {
	stdout, stderr, err := d.runtimeService.ExecSync(r.ContainerId, r.Cmd, time.Duration(r.Timeout)*time.Second)
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

func (d *dockerService) Exec(ctx context.Context, r *runtimeapi.ExecRequest) (*runtimeapi.ExecResponse, error) {
	return d.runtimeService.Exec(r)
}

func (d *dockerService) Attach(ctx context.Context, r *runtimeapi.AttachRequest) (*runtimeapi.AttachResponse, error) {
	return d.runtimeService.Attach(r)
}

func (d *dockerService) PortForward(ctx context.Context, r *runtimeapi.PortForwardRequest) (*runtimeapi.PortForwardResponse, error) {
	return d.runtimeService.PortForward(r)
}

func (d *dockerService) UpdateRuntimeConfig(ctx context.Context, r *runtimeapi.UpdateRuntimeConfigRequest) (*runtimeapi.UpdateRuntimeConfigResponse, error) {
	err := d.runtimeService.UpdateRuntimeConfig(r.GetRuntimeConfig())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.UpdateRuntimeConfigResponse{}, nil
}

func (d *dockerService) ListImages(ctx context.Context, r *runtimeapi.ListImagesRequest) (*runtimeapi.ListImagesResponse, error) {
	images, err := d.imageService.ListImages(r.GetFilter())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.ListImagesResponse{Images: images}, nil
}

func (d *dockerService) ImageStatus(ctx context.Context, r *runtimeapi.ImageStatusRequest) (*runtimeapi.ImageStatusResponse, error) {
	image, err := d.imageService.ImageStatus(r.GetImage())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.ImageStatusResponse{Image: image}, nil
}

func (d *dockerService) PullImage(ctx context.Context, r *runtimeapi.PullImageRequest) (*runtimeapi.PullImageResponse, error) {
	image, err := d.imageService.PullImage(r.GetImage(), r.GetAuth())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.PullImageResponse{ImageRef: image}, nil
}

func (d *dockerService) RemoveImage(ctx context.Context, r *runtimeapi.RemoveImageRequest) (*runtimeapi.RemoveImageResponse, error) {
	err := d.imageService.RemoveImage(r.GetImage())
	if err != nil {
		return nil, err
	}
	return &runtimeapi.RemoveImageResponse{}, nil
}
