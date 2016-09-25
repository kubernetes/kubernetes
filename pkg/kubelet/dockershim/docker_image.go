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

	dockertypes "github.com/docker/engine-api/types"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// This file implements methods in ImageManagerService.

// ListImages lists existing images.
func (ds *dockerService) ListImages(filter *runtimeApi.ImageFilter) ([]*runtimeApi.Image, error) {
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

	result := []*runtimeApi.Image{}
	for _, i := range images {
		apiImage, err := toRuntimeAPIImage(&i)
		if err != nil {
			// TODO: log an error message?
			continue
		}
		result = append(result, apiImage)
	}
	return result, nil
}

// ImageStatus returns the status of the image.
func (ds *dockerService) ImageStatus(image *runtimeApi.ImageSpec) (*runtimeApi.Image, error) {
	images, err := ds.ListImages(&runtimeApi.ImageFilter{Image: image})
	if err != nil {
		return nil, err
	}
	if len(images) != 1 {
		return nil, fmt.Errorf("ImageStatus returned more than one image: %+v", images)
	}
	return images[0], nil
}

// PullImage pulls an image with authentication config.
func (ds *dockerService) PullImage(image *runtimeApi.ImageSpec, auth *runtimeApi.AuthConfig) error {
	// TODO: add default tags for images or should this be done by kubelet?
	return ds.client.PullImage(image.GetImage(),
		dockertypes.AuthConfig{
			Username:      auth.GetUsername(),
			Password:      auth.GetPassword(),
			ServerAddress: auth.GetServerAddress(),
			IdentityToken: auth.GetIdentityToken(),
			RegistryToken: auth.GetRegistryToken(),
		},
		dockertypes.ImagePullOptions{},
	)
}

// RemoveImage removes the image.
func (ds *dockerService) RemoveImage(image *runtimeApi.ImageSpec) error {
	_, err := ds.client.RemoveImage(image.GetImage(), dockertypes.ImageRemoveOptions{PruneChildren: true})
	return err
}
