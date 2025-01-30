/*
Copyright 2024 The Kubernetes Authors.

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

package runtime

import (
	"context"
	"time"

	criapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	criclient "k8s.io/cri-client/pkg"
)

type defaultImpl struct{}

type impl interface {
	NewRemoteRuntimeService(endpoint string, connectionTimeout time.Duration) (criapi.RuntimeService, error)
	NewRemoteImageService(endpoint string, connectionTimeout time.Duration) (criapi.ImageManagerService, error)
	Status(ctx context.Context, runtimeService criapi.RuntimeService, verbose bool) (*runtimeapi.StatusResponse, error)
	ListPodSandbox(ctx context.Context, runtimeService criapi.RuntimeService, filter *runtimeapi.PodSandboxFilter) ([]*runtimeapi.PodSandbox, error)
	StopPodSandbox(ctx context.Context, runtimeService criapi.RuntimeService, sandboxID string) error
	RemovePodSandbox(ctx context.Context, runtimeService criapi.RuntimeService, podSandboxID string) error
	PullImage(ctx context.Context, imageService criapi.ImageManagerService, image *runtimeapi.ImageSpec, auth *runtimeapi.AuthConfig, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, error)
	ImageStatus(ctx context.Context, imageService criapi.ImageManagerService, image *runtimeapi.ImageSpec, verbose bool) (*runtimeapi.ImageStatusResponse, error)
}

func (*defaultImpl) NewRemoteRuntimeService(endpoint string, connectionTimeout time.Duration) (criapi.RuntimeService, error) {
	return criclient.NewRemoteRuntimeService(endpoint, defaultTimeout, nil, nil)
}

func (*defaultImpl) NewRemoteImageService(endpoint string, connectionTimeout time.Duration) (criapi.ImageManagerService, error) {
	return criclient.NewRemoteImageService(endpoint, connectionTimeout, nil, nil)
}

func (*defaultImpl) Status(ctx context.Context, runtimeService criapi.RuntimeService, verbose bool) (*runtimeapi.StatusResponse, error) {
	return runtimeService.Status(ctx, verbose)
}

func (*defaultImpl) ListPodSandbox(ctx context.Context, runtimeService criapi.RuntimeService, filter *runtimeapi.PodSandboxFilter) ([]*runtimeapi.PodSandbox, error) {
	return runtimeService.ListPodSandbox(ctx, filter)
}

func (*defaultImpl) StopPodSandbox(ctx context.Context, runtimeService criapi.RuntimeService, sandboxID string) error {
	return runtimeService.StopPodSandbox(ctx, sandboxID)
}

func (*defaultImpl) RemovePodSandbox(ctx context.Context, runtimeService criapi.RuntimeService, podSandboxID string) error {
	return runtimeService.RemovePodSandbox(ctx, podSandboxID)
}

func (*defaultImpl) PullImage(ctx context.Context, imageService criapi.ImageManagerService, image *runtimeapi.ImageSpec, auth *runtimeapi.AuthConfig, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	return imageService.PullImage(ctx, image, auth, podSandboxConfig)
}

func (*defaultImpl) ImageStatus(ctx context.Context, imageService criapi.ImageManagerService, image *runtimeapi.ImageSpec, verbose bool) (*runtimeapi.ImageStatusResponse, error) {
	return imageService.ImageStatus(ctx, image, verbose)
}
