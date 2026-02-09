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

package criproxy

import (
	"context"

	kubeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

const (
	ListImages  = "ListImages"
	ImageStatus = "ImageStatus"
	PullImage   = "PullImage"
	RemoveImage = "RemoveImage"
	ImageFsInfo = "ImageFsInfo"
)

// ListImages lists existing images.
func (p *RemoteRuntime) ListImages(ctx context.Context, req *kubeapi.ListImagesRequest) (*kubeapi.ListImagesResponse, error) {
	if err := p.runInjectors(ListImages); err != nil {
		return nil, err
	}

	images, err := p.imageService.ListImages(ctx, req.Filter)
	if err != nil {
		return nil, err
	}
	return &kubeapi.ListImagesResponse{
		Images: images,
	}, nil
}

// ImageStatus returns the status of the image. If the image is not
// present, returns a response with ImageStatusResponse.Image set to
// nil.
func (p *RemoteRuntime) ImageStatus(ctx context.Context, req *kubeapi.ImageStatusRequest) (*kubeapi.ImageStatusResponse, error) {
	if err := p.runInjectors(ImageStatus); err != nil {
		return nil, err
	}

	resp, err := p.imageService.ImageStatus(ctx, req.Image, false)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// PullImage pulls an image with authentication config.
func (p *RemoteRuntime) PullImage(ctx context.Context, req *kubeapi.PullImageRequest) (*kubeapi.PullImageResponse, error) {
	if err := p.runInjectors(PullImage); err != nil {
		return nil, err
	}

	image, err := p.imageService.PullImage(ctx, req.Image, req.Auth, req.SandboxConfig)
	if err != nil {
		return nil, err
	}
	return &kubeapi.PullImageResponse{
		ImageRef: image,
	}, nil
}

// RemoveImage removes the image.
// This call is idempotent, and must not return an error if the image has
// already been removed.
func (p *RemoteRuntime) RemoveImage(ctx context.Context, req *kubeapi.RemoveImageRequest) (*kubeapi.RemoveImageResponse, error) {
	if err := p.runInjectors(RemoveImage); err != nil {
		return nil, err
	}

	err := p.imageService.RemoveImage(ctx, req.Image)
	if err != nil {
		return nil, err
	}
	return &kubeapi.RemoveImageResponse{}, nil
}

// ImageFsInfo returns information of the filesystem that is used to store images.
func (p *RemoteRuntime) ImageFsInfo(ctx context.Context, req *kubeapi.ImageFsInfoRequest) (*kubeapi.ImageFsInfoResponse, error) {
	if err := p.runInjectors(ImageFsInfo); err != nil {
		return nil, err
	}

	resp, err := p.imageService.ImageFsInfo(ctx)
	if err != nil {
		return nil, err
	}
	return resp, nil
}
