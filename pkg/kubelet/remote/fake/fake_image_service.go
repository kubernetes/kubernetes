/*
Copyright 2017 The Kubernetes Authors.

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

package fake

import (
	"context"

	kubeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

// ListImages lists existing images.
func (f *RemoteRuntime) ListImages(ctx context.Context, req *kubeapi.ListImagesRequest) (*kubeapi.ListImagesResponse, error) {
	images, err := f.ImageService.ListImages(req.Filter)
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
func (f *RemoteRuntime) ImageStatus(ctx context.Context, req *kubeapi.ImageStatusRequest) (*kubeapi.ImageStatusResponse, error) {
	status, err := f.ImageService.ImageStatus(req.Image)
	if err != nil {
		return nil, err
	}

	return &kubeapi.ImageStatusResponse{Image: status}, nil
}

// PullImage pulls an image with authentication config.
func (f *RemoteRuntime) PullImage(ctx context.Context, req *kubeapi.PullImageRequest) (*kubeapi.PullImageResponse, error) {
	image, err := f.ImageService.PullImage(req.Image, req.Auth, req.SandboxConfig)
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
func (f *RemoteRuntime) RemoveImage(ctx context.Context, req *kubeapi.RemoveImageRequest) (*kubeapi.RemoveImageResponse, error) {
	err := f.ImageService.RemoveImage(req.Image)
	if err != nil {
		return nil, err
	}

	return &kubeapi.RemoveImageResponse{}, nil
}

// ImageFsInfo returns information of the filesystem that is used to store images.
func (f *RemoteRuntime) ImageFsInfo(ctx context.Context, req *kubeapi.ImageFsInfoRequest) (*kubeapi.ImageFsInfoResponse, error) {
	fsUsage, err := f.ImageService.ImageFsInfo()
	if err != nil {
		return nil, err
	}

	return &kubeapi.ImageFsInfoResponse{ImageFilesystems: fsUsage}, nil
}
