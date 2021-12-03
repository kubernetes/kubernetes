//go:build !dockerless
// +build !dockerless

/*
Copyright 2021 The Kubernetes Authors.

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
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

func TestServer(t *testing.T) {
	file, err := ioutil.TempFile("", "docker-server-")
	assert.Nil(t, err)
	endpoint := "unix://" + file.Name()

	server := NewDockerServer(endpoint, &fakeCRIService{})
	assert.Nil(t, server.Start())

	ctx := context.Background()
	conn, err := grpc.Dial(endpoint, grpc.WithInsecure())
	assert.Nil(t, err)

	runtimeClient := runtimeapi.NewRuntimeServiceClient(conn)
	_, err = runtimeClient.Version(ctx, &runtimeapi.VersionRequest{})
	assert.Nil(t, err)

	imageClient := runtimeapi.NewImageServiceClient(conn)
	_, err = imageClient.ImageFsInfo(ctx, &runtimeapi.ImageFsInfoRequest{})
	assert.Nil(t, err)
}

type fakeCRIService struct{}

func (*fakeCRIService) Start() error {
	return nil
}

func (*fakeCRIService) Version(context.Context, *runtimeapi.VersionRequest) (*runtimeapi.VersionResponse, error) {
	return &runtimeapi.VersionResponse{}, nil
}

func (*fakeCRIService) RunPodSandbox(context.Context, *runtimeapi.RunPodSandboxRequest) (*runtimeapi.RunPodSandboxResponse, error) {
	return nil, nil
}

func (*fakeCRIService) StopPodSandbox(context.Context, *runtimeapi.StopPodSandboxRequest) (*runtimeapi.StopPodSandboxResponse, error) {
	return nil, nil
}

func (*fakeCRIService) RemovePodSandbox(context.Context, *runtimeapi.RemovePodSandboxRequest) (*runtimeapi.RemovePodSandboxResponse, error) {
	return nil, nil
}

func (*fakeCRIService) PodSandboxStatus(context.Context, *runtimeapi.PodSandboxStatusRequest) (*runtimeapi.PodSandboxStatusResponse, error) {
	return nil, nil
}

func (*fakeCRIService) ListPodSandbox(context.Context, *runtimeapi.ListPodSandboxRequest) (*runtimeapi.ListPodSandboxResponse, error) {
	return nil, nil
}

func (*fakeCRIService) CreateContainer(context.Context, *runtimeapi.CreateContainerRequest) (*runtimeapi.CreateContainerResponse, error) {
	return nil, nil
}

func (*fakeCRIService) StartContainer(context.Context, *runtimeapi.StartContainerRequest) (*runtimeapi.StartContainerResponse, error) {
	return nil, nil
}

func (*fakeCRIService) StopContainer(context.Context, *runtimeapi.StopContainerRequest) (*runtimeapi.StopContainerResponse, error) {
	return nil, nil
}

func (*fakeCRIService) RemoveContainer(context.Context, *runtimeapi.RemoveContainerRequest) (*runtimeapi.RemoveContainerResponse, error) {
	return nil, nil
}

func (*fakeCRIService) ListContainers(context.Context, *runtimeapi.ListContainersRequest) (*runtimeapi.ListContainersResponse, error) {
	return nil, nil
}

func (*fakeCRIService) ContainerStatus(context.Context, *runtimeapi.ContainerStatusRequest) (*runtimeapi.ContainerStatusResponse, error) {
	return nil, nil
}

func (*fakeCRIService) UpdateContainerResources(context.Context, *runtimeapi.UpdateContainerResourcesRequest) (*runtimeapi.UpdateContainerResourcesResponse, error) {
	return nil, nil
}

func (*fakeCRIService) ReopenContainerLog(context.Context, *runtimeapi.ReopenContainerLogRequest) (*runtimeapi.ReopenContainerLogResponse, error) {
	return nil, nil
}

func (*fakeCRIService) ExecSync(context.Context, *runtimeapi.ExecSyncRequest) (*runtimeapi.ExecSyncResponse, error) {
	return nil, nil
}

func (*fakeCRIService) Exec(context.Context, *runtimeapi.ExecRequest) (*runtimeapi.ExecResponse, error) {
	return nil, nil
}

func (*fakeCRIService) Attach(context.Context, *runtimeapi.AttachRequest) (*runtimeapi.AttachResponse, error) {
	return nil, nil
}

func (*fakeCRIService) PortForward(context.Context, *runtimeapi.PortForwardRequest) (*runtimeapi.PortForwardResponse, error) {
	return nil, nil
}

func (*fakeCRIService) ContainerStats(context.Context, *runtimeapi.ContainerStatsRequest) (*runtimeapi.ContainerStatsResponse, error) {
	return nil, nil
}

func (*fakeCRIService) ListContainerStats(context.Context, *runtimeapi.ListContainerStatsRequest) (*runtimeapi.ListContainerStatsResponse, error) {
	return nil, nil
}

func (*fakeCRIService) PodSandboxStats(context.Context, *runtimeapi.PodSandboxStatsRequest) (*runtimeapi.PodSandboxStatsResponse, error) {
	return nil, nil
}

func (*fakeCRIService) ListPodSandboxStats(context.Context, *runtimeapi.ListPodSandboxStatsRequest) (*runtimeapi.ListPodSandboxStatsResponse, error) {
	return nil, nil
}

func (*fakeCRIService) UpdateRuntimeConfig(context.Context, *runtimeapi.UpdateRuntimeConfigRequest) (*runtimeapi.UpdateRuntimeConfigResponse, error) {
	return nil, nil
}

func (*fakeCRIService) Status(context.Context, *runtimeapi.StatusRequest) (*runtimeapi.StatusResponse, error) {
	return nil, nil
}

func (*fakeCRIService) ListImages(context.Context, *runtimeapi.ListImagesRequest) (*runtimeapi.ListImagesResponse, error) {
	return nil, nil
}

func (*fakeCRIService) ImageStatus(context.Context, *runtimeapi.ImageStatusRequest) (*runtimeapi.ImageStatusResponse, error) {
	return nil, nil
}

func (*fakeCRIService) PullImage(context.Context, *runtimeapi.PullImageRequest) (*runtimeapi.PullImageResponse, error) {
	return nil, nil
}

func (*fakeCRIService) RemoveImage(context.Context, *runtimeapi.RemoveImageRequest) (*runtimeapi.RemoveImageResponse, error) {
	return nil, nil
}

func (*fakeCRIService) ImageFsInfo(context.Context, *runtimeapi.ImageFsInfoRequest) (*runtimeapi.ImageFsInfoResponse, error) {
	return &runtimeapi.ImageFsInfoResponse{}, nil
}
