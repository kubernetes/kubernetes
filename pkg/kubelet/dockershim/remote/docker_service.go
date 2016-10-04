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

	"golang.org/x/net/context"

	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockershim"
)

// DockerService uses dockershim service to implement runtime and image services.
// Notice that the contexts in the functions are not used now.
// TODO(random-liu): Change the dockershim service to support context, and implement
// internal services and remote services with the dockershim service.
type DockerService struct {
	runtimeService internalApi.RuntimeService
	imageService   internalApi.ImageManagerService
}

var _ runtimeApi.RuntimeServiceServer = &DockerService{}

func NewDockerService(s dockershim.DockerService) *DockerService {
	return &DockerService{runtimeService: s, imageService: s}
}

func (d *DockerService) Version(ctx context.Context, r *runtimeApi.VersionRequest) (*runtimeApi.VersionResponse, error) {
	return d.runtimeService.Version(r.GetVersion())
}
func (d *DockerService) UpdateRuntimeConfig(ctx context.Context, r *runtimeApi.UpdateRuntimeConfigRequest) (*runtimeApi.UpdateRuntimeConfigResponse, error) {
	err := d.runtimeService.UpdateRuntimeConfig(r.GetRuntimeConfig())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.UpdateRuntimeConfigResponse{}, nil
}

func (d *DockerService) RunPodSandbox(ctx context.Context, r *runtimeApi.RunPodSandboxRequest) (*runtimeApi.RunPodSandboxResponse, error) {
	podSandboxId, err := d.runtimeService.RunPodSandbox(r.GetConfig())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.RunPodSandboxResponse{PodSandboxId: &podSandboxId}, nil
}

func (d *DockerService) StopPodSandbox(ctx context.Context, r *runtimeApi.StopPodSandboxRequest) (*runtimeApi.StopPodSandboxResponse, error) {
	err := d.runtimeService.StopPodSandbox(r.GetPodSandboxId())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.StopPodSandboxResponse{}, nil
}

func (d *DockerService) RemovePodSandbox(ctx context.Context, r *runtimeApi.RemovePodSandboxRequest) (*runtimeApi.RemovePodSandboxResponse, error) {
	err := d.runtimeService.RemovePodSandbox(r.GetPodSandboxId())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.RemovePodSandboxResponse{}, nil
}

func (d *DockerService) PodSandboxStatus(ctx context.Context, r *runtimeApi.PodSandboxStatusRequest) (*runtimeApi.PodSandboxStatusResponse, error) {
	podSandboxStatus, err := d.runtimeService.PodSandboxStatus(r.GetPodSandboxId())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.PodSandboxStatusResponse{Status: podSandboxStatus}, nil
}

func (d *DockerService) ListPodSandbox(ctx context.Context, r *runtimeApi.ListPodSandboxRequest) (*runtimeApi.ListPodSandboxResponse, error) {
	items, err := d.runtimeService.ListPodSandbox(r.GetFilter())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.ListPodSandboxResponse{Items: items}, nil
}

func (d *DockerService) CreateContainer(ctx context.Context, r *runtimeApi.CreateContainerRequest) (*runtimeApi.CreateContainerResponse, error) {
	containerId, err := d.runtimeService.CreateContainer(r.GetPodSandboxId(), r.GetConfig(), r.GetSandboxConfig())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.CreateContainerResponse{ContainerId: &containerId}, nil
}

func (d *DockerService) StartContainer(ctx context.Context, r *runtimeApi.StartContainerRequest) (*runtimeApi.StartContainerResponse, error) {
	err := d.runtimeService.StartContainer(r.GetContainerId())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.StartContainerResponse{}, nil
}

func (d *DockerService) StopContainer(ctx context.Context, r *runtimeApi.StopContainerRequest) (*runtimeApi.StopContainerResponse, error) {
	err := d.runtimeService.StopContainer(r.GetContainerId(), r.GetTimeout())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.StopContainerResponse{}, nil
}

func (d *DockerService) RemoveContainer(ctx context.Context, r *runtimeApi.RemoveContainerRequest) (*runtimeApi.RemoveContainerResponse, error) {
	err := d.runtimeService.RemoveContainer(r.GetContainerId())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.RemoveContainerResponse{}, nil
}

func (d *DockerService) ListContainers(ctx context.Context, r *runtimeApi.ListContainersRequest) (*runtimeApi.ListContainersResponse, error) {
	containers, err := d.runtimeService.ListContainers(r.GetFilter())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.ListContainersResponse{Containers: containers}, nil
}

func (d *DockerService) ContainerStatus(ctx context.Context, r *runtimeApi.ContainerStatusRequest) (*runtimeApi.ContainerStatusResponse, error) {
	status, err := d.runtimeService.ContainerStatus(r.GetContainerId())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.ContainerStatusResponse{Status: status}, nil
}

func (d *DockerService) Exec(runtimeApi.RuntimeService_ExecServer) error {
	return fmt.Errorf("not implemented")
}

func (d *DockerService) ListImages(ctx context.Context, r *runtimeApi.ListImagesRequest) (*runtimeApi.ListImagesResponse, error) {
	images, err := d.imageService.ListImages(r.GetFilter())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.ListImagesResponse{Images: images}, nil
}

func (d *DockerService) ImageStatus(ctx context.Context, r *runtimeApi.ImageStatusRequest) (*runtimeApi.ImageStatusResponse, error) {
	image, err := d.imageService.ImageStatus(r.GetImage())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.ImageStatusResponse{Image: image}, nil
}

func (d *DockerService) PullImage(ctx context.Context, r *runtimeApi.PullImageRequest) (*runtimeApi.PullImageResponse, error) {
	err := d.imageService.PullImage(r.GetImage(), r.GetAuth())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.PullImageResponse{}, nil
}

func (d *DockerService) RemoveImage(ctx context.Context, r *runtimeApi.RemoveImageRequest) (*runtimeApi.RemoveImageResponse, error) {
	err := d.imageService.RemoveImage(r.GetImage())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.RemoveImageResponse{}, nil
}
