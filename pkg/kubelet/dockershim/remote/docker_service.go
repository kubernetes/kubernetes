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

	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockershim"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
)

// DockerService is the interface implement CRI remote service server.
type DockerService interface {
	runtimeApi.RuntimeServiceServer
	runtimeApi.ImageServiceServer
}

// dockerService uses dockershim service to implement DockerService.
// Notice that the contexts in the functions are not used now.
// TODO(random-liu): Change the dockershim service to support context, and implement
// internal services and remote services with the dockershim service.
type dockerService struct {
	runtimeService internalApi.RuntimeService
	imageService   internalApi.ImageManagerService
}

func NewDockerService(s dockershim.DockerService) DockerService {
	return &dockerService{runtimeService: s, imageService: s}
}

func (d *dockerService) Version(ctx context.Context, r *runtimeApi.VersionRequest) (*runtimeApi.VersionResponse, error) {
	return d.runtimeService.Version(r.GetVersion())
}

func (d *dockerService) Status(ctx context.Context, r *runtimeApi.StatusRequest) (*runtimeApi.StatusResponse, error) {
	status, err := d.runtimeService.Status()
	if err != nil {
		return nil, err
	}
	return &runtimeApi.StatusResponse{Status: status}, nil
}

func (d *dockerService) RunPodSandbox(ctx context.Context, r *runtimeApi.RunPodSandboxRequest) (*runtimeApi.RunPodSandboxResponse, error) {
	podSandboxId, err := d.runtimeService.RunPodSandbox(r.GetConfig())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.RunPodSandboxResponse{PodSandboxId: &podSandboxId}, nil
}

func (d *dockerService) StopPodSandbox(ctx context.Context, r *runtimeApi.StopPodSandboxRequest) (*runtimeApi.StopPodSandboxResponse, error) {
	err := d.runtimeService.StopPodSandbox(r.GetPodSandboxId())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.StopPodSandboxResponse{}, nil
}

func (d *dockerService) RemovePodSandbox(ctx context.Context, r *runtimeApi.RemovePodSandboxRequest) (*runtimeApi.RemovePodSandboxResponse, error) {
	err := d.runtimeService.RemovePodSandbox(r.GetPodSandboxId())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.RemovePodSandboxResponse{}, nil
}

func (d *dockerService) PodSandboxStatus(ctx context.Context, r *runtimeApi.PodSandboxStatusRequest) (*runtimeApi.PodSandboxStatusResponse, error) {
	podSandboxStatus, err := d.runtimeService.PodSandboxStatus(r.GetPodSandboxId())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.PodSandboxStatusResponse{Status: podSandboxStatus}, nil
}

func (d *dockerService) ListPodSandbox(ctx context.Context, r *runtimeApi.ListPodSandboxRequest) (*runtimeApi.ListPodSandboxResponse, error) {
	items, err := d.runtimeService.ListPodSandbox(r.GetFilter())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.ListPodSandboxResponse{Items: items}, nil
}

func (d *dockerService) CreateContainer(ctx context.Context, r *runtimeApi.CreateContainerRequest) (*runtimeApi.CreateContainerResponse, error) {
	containerId, err := d.runtimeService.CreateContainer(r.GetPodSandboxId(), r.GetConfig(), r.GetSandboxConfig())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.CreateContainerResponse{ContainerId: &containerId}, nil
}

func (d *dockerService) StartContainer(ctx context.Context, r *runtimeApi.StartContainerRequest) (*runtimeApi.StartContainerResponse, error) {
	err := d.runtimeService.StartContainer(r.GetContainerId())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.StartContainerResponse{}, nil
}

func (d *dockerService) StopContainer(ctx context.Context, r *runtimeApi.StopContainerRequest) (*runtimeApi.StopContainerResponse, error) {
	err := d.runtimeService.StopContainer(r.GetContainerId(), r.GetTimeout())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.StopContainerResponse{}, nil
}

func (d *dockerService) RemoveContainer(ctx context.Context, r *runtimeApi.RemoveContainerRequest) (*runtimeApi.RemoveContainerResponse, error) {
	err := d.runtimeService.RemoveContainer(r.GetContainerId())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.RemoveContainerResponse{}, nil
}

func (d *dockerService) ListContainers(ctx context.Context, r *runtimeApi.ListContainersRequest) (*runtimeApi.ListContainersResponse, error) {
	containers, err := d.runtimeService.ListContainers(r.GetFilter())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.ListContainersResponse{Containers: containers}, nil
}

func (d *dockerService) ContainerStatus(ctx context.Context, r *runtimeApi.ContainerStatusRequest) (*runtimeApi.ContainerStatusResponse, error) {
	status, err := d.runtimeService.ContainerStatus(r.GetContainerId())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.ContainerStatusResponse{Status: status}, nil
}

func (d *dockerService) ExecSync(ctx context.Context, r *runtimeApi.ExecSyncRequest) (*runtimeApi.ExecSyncResponse, error) {
	stdout, stderr, err := d.runtimeService.ExecSync(r.GetContainerId(), r.GetCmd(), time.Duration(r.GetTimeout())*time.Second)
	var exitCode int32
	if err != nil {
		exitError, ok := err.(utilexec.ExitError)
		if !ok {
			return nil, err
		}
		exitCode = int32(exitError.ExitStatus())
	}
	return &runtimeApi.ExecSyncResponse{
		Stdout:   stdout,
		Stderr:   stderr,
		ExitCode: &exitCode,
	}, nil
}

func (d *dockerService) Exec(ctx context.Context, r *runtimeApi.ExecRequest) (*runtimeApi.ExecResponse, error) {
	return d.runtimeService.Exec(r)
}

func (d *dockerService) Attach(ctx context.Context, r *runtimeApi.AttachRequest) (*runtimeApi.AttachResponse, error) {
	return d.runtimeService.Attach(r)
}

func (d *dockerService) PortForward(ctx context.Context, r *runtimeApi.PortForwardRequest) (*runtimeApi.PortForwardResponse, error) {
	return d.runtimeService.PortForward(r)
}

func (d *dockerService) UpdateRuntimeConfig(ctx context.Context, r *runtimeApi.UpdateRuntimeConfigRequest) (*runtimeApi.UpdateRuntimeConfigResponse, error) {
	err := d.runtimeService.UpdateRuntimeConfig(r.GetRuntimeConfig())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.UpdateRuntimeConfigResponse{}, nil
}

func (d *dockerService) ListImages(ctx context.Context, r *runtimeApi.ListImagesRequest) (*runtimeApi.ListImagesResponse, error) {
	images, err := d.imageService.ListImages(r.GetFilter())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.ListImagesResponse{Images: images}, nil
}

func (d *dockerService) ImageStatus(ctx context.Context, r *runtimeApi.ImageStatusRequest) (*runtimeApi.ImageStatusResponse, error) {
	image, err := d.imageService.ImageStatus(r.GetImage())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.ImageStatusResponse{Image: image}, nil
}

func (d *dockerService) PullImage(ctx context.Context, r *runtimeApi.PullImageRequest) (*runtimeApi.PullImageResponse, error) {
	err := d.imageService.PullImage(r.GetImage(), r.GetAuth())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.PullImageResponse{}, nil
}

func (d *dockerService) RemoveImage(ctx context.Context, r *runtimeApi.RemoveImageRequest) (*runtimeApi.RemoveImageResponse, error) {
	err := d.imageService.RemoveImage(r.GetImage())
	if err != nil {
		return nil, err
	}
	return &runtimeApi.RemoveImageResponse{}, nil
}
