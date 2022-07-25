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
	"context"
	"errors"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	runtimeapiV1alpha2 "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/util"
)

// remoteImageService is a gRPC implementation of internalapi.ImageManagerService.
type remoteImageService struct {
	timeout             time.Duration
	imageClient         runtimeapi.ImageServiceClient
	imageClientV1alpha2 runtimeapiV1alpha2.ImageServiceClient
}

// NewRemoteImageService creates a new internalapi.ImageManagerService.
func NewRemoteImageService(endpoint string, connectionTimeout time.Duration) (internalapi.ImageManagerService, error) {
	klog.V(3).InfoS("Connecting to image service", "endpoint", endpoint)
	addr, dialer, err := util.GetAddressAndDialer(endpoint)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), connectionTimeout)
	defer cancel()

	conn, err := grpc.DialContext(ctx, addr, grpc.WithInsecure(), grpc.WithContextDialer(dialer), grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxMsgSize)))
	if err != nil {
		klog.ErrorS(err, "Connect remote image service failed", "address", addr)
		return nil, err
	}

	service := &remoteImageService{timeout: connectionTimeout}

	if err := service.determineAPIVersion(conn); err != nil {
		return nil, err
	}

	return service, nil

}

// useV1API returns true if the v1 CRI API should be used instead of v1alpha2.
func (r *remoteImageService) useV1API() bool {
	return r.imageClientV1alpha2 == nil
}

// determineAPIVersion tries to connect to the remote image service by using the
// highest available API version.
//
// A GRPC redial will always use the initially selected (or automatically
// determined) CRI API version. If the redial was due to the container runtime
// being upgraded, then the container runtime must also support the initially
// selected version or the redial is expected to fail, which requires a restart
// of kubelet.
func (r *remoteImageService) determineAPIVersion(conn *grpc.ClientConn) error {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	klog.V(4).InfoS("Finding the CRI API image version")
	r.imageClient = runtimeapi.NewImageServiceClient(conn)

	if _, err := r.imageClient.ImageFsInfo(ctx, &runtimeapi.ImageFsInfoRequest{}); err == nil {
		klog.V(2).InfoS("Using CRI v1 image API")

	} else if status.Code(err) == codes.Unimplemented {
		klog.V(2).InfoS("Falling back to CRI v1alpha2 image API (deprecated)")
		r.imageClientV1alpha2 = runtimeapiV1alpha2.NewImageServiceClient(conn)

	} else {
		return fmt.Errorf("unable to determine image API version: %w", err)
	}

	return nil
}

// ListImages lists available images.
func (r *remoteImageService) ListImages(filter *runtimeapi.ImageFilter) ([]*runtimeapi.Image, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	if r.useV1API() {
		return r.listImagesV1(ctx, filter)
	}

	return r.listImagesV1alpha2(ctx, filter)
}

func (r *remoteImageService) listImagesV1alpha2(ctx context.Context, filter *runtimeapi.ImageFilter) ([]*runtimeapi.Image, error) {
	resp, err := r.imageClientV1alpha2.ListImages(ctx, &runtimeapiV1alpha2.ListImagesRequest{
		Filter: v1alpha2ImageFilter(filter),
	})
	if err != nil {
		klog.ErrorS(err, "ListImages with filter from image service failed", "filter", filter)
		return nil, err
	}
	return fromV1alpha2ListImagesResponse(resp).Images, nil
}

func (r *remoteImageService) listImagesV1(ctx context.Context, filter *runtimeapi.ImageFilter) ([]*runtimeapi.Image, error) {
	resp, err := r.imageClient.ListImages(ctx, &runtimeapi.ListImagesRequest{
		Filter: filter,
	})
	if err != nil {
		klog.ErrorS(err, "ListImages with filter from image service failed", "filter", filter)
		return nil, err
	}

	return resp.Images, nil
}

// ImageStatus returns the status of the image.
func (r *remoteImageService) ImageStatus(image *runtimeapi.ImageSpec, verbose bool) (*runtimeapi.ImageStatusResponse, error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	// TODO: for refactoring common code blocks between the cri versions into
	// one code block in the internal where possible examples:
	// https://github.com/kubernetes/kubernetes/pull/104575/files#r705600987
	// https://github.com/kubernetes/kubernetes/pull/104575/files#r696793706
	if r.useV1API() {
		return r.imageStatusV1(ctx, image, verbose)
	}

	return r.imageStatusV1alpha2(ctx, image, verbose)
}

func (r *remoteImageService) imageStatusV1alpha2(ctx context.Context, image *runtimeapi.ImageSpec, verbose bool) (*runtimeapi.ImageStatusResponse, error) {
	resp, err := r.imageClientV1alpha2.ImageStatus(ctx, &runtimeapiV1alpha2.ImageStatusRequest{
		Image:   v1alpha2ImageSpec(image),
		Verbose: verbose,
	})
	if err != nil {
		klog.ErrorS(err, "Get ImageStatus from image service failed", "image", image.Image)
		return nil, err
	}

	if resp.Image != nil {
		if resp.Image.Id == "" || resp.Image.Size_ == 0 {
			errorMessage := fmt.Sprintf("Id or size of image %q is not set", image.Image)
			err := errors.New(errorMessage)
			klog.ErrorS(err, "ImageStatus failed", "image", image.Image)
			return nil, err
		}
	}

	return fromV1alpha2ImageStatusResponse(resp), nil
}

func (r *remoteImageService) imageStatusV1(ctx context.Context, image *runtimeapi.ImageSpec, verbose bool) (*runtimeapi.ImageStatusResponse, error) {
	resp, err := r.imageClient.ImageStatus(ctx, &runtimeapi.ImageStatusRequest{
		Image:   image,
		Verbose: verbose,
	})
	if err != nil {
		klog.ErrorS(err, "Get ImageStatus from image service failed", "image", image.Image)
		return nil, err
	}

	if resp.Image != nil {
		if resp.Image.Id == "" || resp.Image.Size_ == 0 {
			errorMessage := fmt.Sprintf("Id or size of image %q is not set", image.Image)
			err := errors.New(errorMessage)
			klog.ErrorS(err, "ImageStatus failed", "image", image.Image)
			return nil, err
		}
	}

	return resp, nil
}

// PullImage pulls an image with authentication config.
func (r *remoteImageService) PullImage(image *runtimeapi.ImageSpec, auth *runtimeapi.AuthConfig, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	if r.useV1API() {
		return r.pullImageV1(ctx, image, auth, podSandboxConfig)
	}

	return r.pullImageV1alpha2(ctx, image, auth, podSandboxConfig)
}

func (r *remoteImageService) pullImageV1alpha2(ctx context.Context, image *runtimeapi.ImageSpec, auth *runtimeapi.AuthConfig, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	resp, err := r.imageClientV1alpha2.PullImage(ctx, &runtimeapiV1alpha2.PullImageRequest{
		Image:         v1alpha2ImageSpec(image),
		Auth:          v1alpha2AuthConfig(auth),
		SandboxConfig: v1alpha2PodSandboxConfig(podSandboxConfig),
	})
	if err != nil {
		klog.ErrorS(err, "PullImage from image service failed", "image", image.Image)
		return "", err
	}

	if resp.ImageRef == "" {
		klog.ErrorS(errors.New("PullImage failed"), "ImageRef of image is not set", "image", image.Image)
		errorMessage := fmt.Sprintf("imageRef of image %q is not set", image.Image)
		return "", errors.New(errorMessage)
	}

	return resp.ImageRef, nil
}

func (r *remoteImageService) pullImageV1(ctx context.Context, image *runtimeapi.ImageSpec, auth *runtimeapi.AuthConfig, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	resp, err := r.imageClient.PullImage(ctx, &runtimeapi.PullImageRequest{
		Image:         image,
		Auth:          auth,
		SandboxConfig: podSandboxConfig,
	})
	if err != nil {
		klog.ErrorS(err, "PullImage from image service failed", "image", image.Image)
		return "", err
	}

	if resp.ImageRef == "" {
		klog.ErrorS(errors.New("PullImage failed"), "ImageRef of image is not set", "image", image.Image)
		errorMessage := fmt.Sprintf("imageRef of image %q is not set", image.Image)
		return "", errors.New(errorMessage)
	}

	return resp.ImageRef, nil
}

// RemoveImage removes the image.
func (r *remoteImageService) RemoveImage(image *runtimeapi.ImageSpec) (err error) {
	ctx, cancel := getContextWithTimeout(r.timeout)
	defer cancel()

	if r.useV1API() {
		_, err = r.imageClient.RemoveImage(ctx, &runtimeapi.RemoveImageRequest{
			Image: image,
		})
	} else {
		_, err = r.imageClientV1alpha2.RemoveImage(ctx, &runtimeapiV1alpha2.RemoveImageRequest{
			Image: v1alpha2ImageSpec(image),
		})
	}
	if err != nil {
		klog.ErrorS(err, "RemoveImage from image service failed", "image", image.Image)
		return err
	}

	return nil
}

// ImageFsInfo returns information of the filesystem that is used to store images.
func (r *remoteImageService) ImageFsInfo() ([]*runtimeapi.FilesystemUsage, error) {
	// Do not set timeout, because `ImageFsInfo` takes time.
	// TODO(random-liu): Should we assume runtime should cache the result, and set timeout here?
	ctx, cancel := getContextWithCancel()
	defer cancel()

	if r.useV1API() {
		return r.imageFsInfoV1(ctx)
	}

	return r.imageFsInfoV1alpha2(ctx)
}

func (r *remoteImageService) imageFsInfoV1alpha2(ctx context.Context) ([]*runtimeapi.FilesystemUsage, error) {
	resp, err := r.imageClientV1alpha2.ImageFsInfo(ctx, &runtimeapiV1alpha2.ImageFsInfoRequest{})
	if err != nil {
		klog.ErrorS(err, "ImageFsInfo from image service failed")
		return nil, err
	}
	return fromV1alpha2ImageFsInfoResponse(resp).GetImageFilesystems(), nil
}

func (r *remoteImageService) imageFsInfoV1(ctx context.Context) ([]*runtimeapi.FilesystemUsage, error) {
	resp, err := r.imageClient.ImageFsInfo(ctx, &runtimeapi.ImageFsInfoRequest{})
	if err != nil {
		klog.ErrorS(err, "ImageFsInfo from image service failed")
		return nil, err
	}
	return resp.GetImageFilesystems(), nil
}
