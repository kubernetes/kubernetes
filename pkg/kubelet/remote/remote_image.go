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

	"github.com/golang/glog"
	"google.golang.org/grpc"
	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// RemoteImageService is a gRPC implementation of internalApi.ImageManagerService.
type RemoteImageService struct {
	timeout     time.Duration
	imageClient runtimeApi.ImageServiceClient
}

// NewRemoteImageService creates a new internalApi.ImageManagerService.
func NewRemoteImageService(addr string, connectionTimout time.Duration) (internalApi.ImageManagerService, error) {
	glog.V(3).Infof("Connecting to image service %s", addr)
	conn, err := grpc.Dial(addr, grpc.WithInsecure(), grpc.WithDialer(dial))
	if err != nil {
		glog.Errorf("Connect remote image service %s failed: %v", addr, err)
		return nil, err
	}

	return &RemoteImageService{
		timeout:     connectionTimout,
		imageClient: runtimeApi.NewImageServiceClient(conn),
	}, nil
}

// ListImages lists available images.
func (r *RemoteImageService) ListImages(filter *runtimeApi.ImageFilter) ([]*runtimeApi.Image, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.imageClient.ListImages(ctx, &runtimeApi.ListImagesRequest{
		Filter: filter,
	})
	if err != nil {
		glog.Errorf("ListImages with filter %q from image service failed: %v", filter, err)
		return nil, err
	}

	return resp.Images, nil
}

// ImageStatus returns the status of the image.
func (r *RemoteImageService) ImageStatus(image *runtimeApi.ImageSpec) (*runtimeApi.Image, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	resp, err := r.imageClient.ImageStatus(ctx, &runtimeApi.ImageStatusRequest{
		Image: image,
	})
	if err != nil {
		glog.Errorf("ImageStatus %q from image service failed: %v", image.GetImage(), err)
		return nil, err
	}

	return resp.Image, nil
}

// PullImage pulls a image with authentication config.
func (r *RemoteImageService) PullImage(image *runtimeApi.ImageSpec, auth *runtimeApi.AuthConfig) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.imageClient.PullImage(ctx, &runtimeApi.PullImageRequest{
		Image: image,
		Auth:  auth,
	})
	if err != nil {
		glog.Errorf("PullImage %q from image service failed: %v", image.GetImage(), err)
		return err
	}

	return nil
}

// RemoveImage removes the image.
func (r *RemoteImageService) RemoveImage(image *runtimeApi.ImageSpec) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	_, err := r.imageClient.RemoveImage(ctx, &runtimeApi.RemoveImageRequest{
		Image: image,
	})
	if err != nil {
		glog.Errorf("RemoveImage %q from image service failed: %v", image.GetImage(), err)
		return err
	}

	return nil
}
