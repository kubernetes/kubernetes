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
	dockertypes "github.com/docker/engine-api/types"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
)

// This file implements methods in ImageManagerService.

// ListImages lists existing images.
func (ds *dockerService) ListImages(filter *runtimeapi.ImageFilter) ([]*runtimeapi.Image, error) {
	opts := dockertypes.ImageListOptions{}
	if filter != nil {
		if imgSpec := filter.GetImage(); imgSpec != nil {
			opts.MatchName = imgSpec.GetImage()
		}
	}

	images, err := ds.client.ListImages(opts)
	if err != nil {
		return nil, err
	}

	result := []*runtimeapi.Image{}
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
	imageInspect, err := ds.client.InspectImageByRef(image.GetImage())
	if err != nil {
		if dockertools.IsImageNotFoundError(err) {
			return nil, nil
		}
		return nil, err
	}
	return imageInspectToRuntimeAPIImage(imageInspect)
}

// PullImage pulls an image with authentication config.
func (ds *dockerService) PullImage(image *runtimeapi.ImageSpec, auth *runtimeapi.AuthConfig) (string, error) {
	err := ds.client.PullImage(image.GetImage(),
		dockertypes.AuthConfig{
			Username:      auth.GetUsername(),
			Password:      auth.GetPassword(),
			ServerAddress: auth.GetServerAddress(),
			IdentityToken: auth.GetIdentityToken(),
			RegistryToken: auth.GetRegistryToken(),
		},
		dockertypes.ImagePullOptions{},
	)
	if err != nil {
		return "", err
	}

	return dockertools.GetImageRef(ds.client, image.GetImage())
}

// RemoveImage removes the image.
func (ds *dockerService) RemoveImage(image *runtimeapi.ImageSpec) error {
	// If the image has multiple tags, we need to remove all the tags
	// TODO: We assume image.Image is image ID here, which is true in the current implementation
	// of kubelet, but we should still clarify this in CRI.
	imageInspect, err := ds.client.InspectImageByID(image.GetImage())
	if err == nil && imageInspect != nil && len(imageInspect.RepoTags) > 1 {
		for _, tag := range imageInspect.RepoTags {
			if _, err := ds.client.RemoveImage(tag, dockertypes.ImageRemoveOptions{PruneChildren: true}); err != nil {
				return err
			}
		}
		return nil
	}

	_, err = ds.client.RemoveImage(image.GetImage(), dockertypes.ImageRemoveOptions{PruneChildren: true})
	return err
}
