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

package dockershim

import (
	"fmt"
	"net/http"

	dockertypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/pkg/jsonmessage"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

// This file implements methods in ImageManagerService.

// ListImages lists existing images.
func (ds *dockerService) ListImages(filter *runtimeapi.ImageFilter) ([]*runtimeapi.Image, error) {
	opts := dockertypes.ImageListOptions{}
	if filter != nil {
		if imgSpec := filter.GetImage(); imgSpec != nil {
			opts.Filters.Add("reference", imgSpec.Image)
		}
	}

	images, err := ds.client.ListImages(opts)
	if err != nil {
		return nil, err
	}

	result := make([]*runtimeapi.Image, 0, len(images))
	for _, i := range images {
		apiImage, err := imageToRuntimeAPIImage(&i)
		if err != nil {
			// TODO: log an error message?
			continue
		}
		result = append(result, apiImage)
	}
	return result, nil
}

// ImageStatus returns the status of the image, returns nil if the image doesn't present.
func (ds *dockerService) ImageStatus(image *runtimeapi.ImageSpec) (*runtimeapi.Image, error) {
	imageInspect, err := ds.client.InspectImageByRef(image.Image)
	if err != nil {
		if libdocker.IsImageNotFoundError(err) {
			return nil, nil
		}
		return nil, err
	}
	return imageInspectToRuntimeAPIImage(imageInspect)
}

// PullImage pulls an image with authentication config.
func (ds *dockerService) PullImage(image *runtimeapi.ImageSpec, auth *runtimeapi.AuthConfig) (string, error) {
	authConfig := dockertypes.AuthConfig{}
	if auth != nil {
		authConfig.Username = auth.Username
		authConfig.Password = auth.Password
		authConfig.ServerAddress = auth.ServerAddress
		authConfig.IdentityToken = auth.IdentityToken
		authConfig.RegistryToken = auth.RegistryToken
	}
	err := ds.client.PullImage(image.Image,
		authConfig,
		dockertypes.ImagePullOptions{},
	)
	if err != nil {
		return "", filterHTTPError(err, image.Image)
	}

	return getImageRef(ds.client, image.Image)
}

// RemoveImage removes the image.
func (ds *dockerService) RemoveImage(image *runtimeapi.ImageSpec) error {
	// If the image has multiple tags, we need to remove all the tags
	// TODO: We assume image.Image is image ID here, which is true in the current implementation
	// of kubelet, but we should still clarify this in CRI.
	imageInspect, err := ds.client.InspectImageByID(image.Image)
	if err == nil && imageInspect != nil && len(imageInspect.RepoTags) > 1 {
		for _, tag := range imageInspect.RepoTags {
			if _, err := ds.client.RemoveImage(tag, dockertypes.ImageRemoveOptions{PruneChildren: true}); err != nil && !libdocker.IsImageNotFoundError(err) {
				return err
			}
		}
		return nil
	}
	// dockerclient.InspectImageByID doesn't work with digest and repoTags,
	// it is safe to continue removing it since there is another check below.
	if err != nil && !libdocker.IsImageNotFoundError(err) {
		return err
	}

	_, err = ds.client.RemoveImage(image.Image, dockertypes.ImageRemoveOptions{PruneChildren: true})
	if err != nil && !libdocker.IsImageNotFoundError(err) {
		return err
	}
	return nil
}

// getImageRef returns the image digest if exists, or else returns the image ID.
func getImageRef(client libdocker.Interface, image string) (string, error) {
	img, err := client.InspectImageByRef(image)
	if err != nil {
		return "", err
	}
	if img == nil {
		return "", fmt.Errorf("unable to inspect image %s", image)
	}

	// Returns the digest if it exist.
	if len(img.RepoDigests) > 0 {
		return img.RepoDigests[0], nil
	}

	return img.ID, nil
}

// ImageFsInfo returns information of the filesystem that is used to store images.
func (ds *dockerService) ImageFsInfo() ([]*runtimeapi.FilesystemUsage, error) {
	return nil, fmt.Errorf("not implemented")
}

func filterHTTPError(err error, image string) error {
	// docker/docker/pull/11314 prints detailed error info for docker pull.
	// When it hits 502, it returns a verbose html output including an inline svg,
	// which makes the output of kubectl get pods much harder to parse.
	// Here converts such verbose output to a concise one.
	jerr, ok := err.(*jsonmessage.JSONError)
	if ok && (jerr.Code == http.StatusBadGateway ||
		jerr.Code == http.StatusServiceUnavailable ||
		jerr.Code == http.StatusGatewayTimeout) {
		return fmt.Errorf("RegistryUnavailable: %v", err)
	}
	return err

}
