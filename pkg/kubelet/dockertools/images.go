/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package dockertools

import (
	"fmt"

	"github.com/golang/glog"

	dockertypes "github.com/docker/engine-api/types"
	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/sets"
)

type runtimeImages struct {
	// Provides image stats
	*imageStatsProvider
	// TODO(yifan): Record the pull failure so we can eliminate the image checking?
	// Lower level docker image puller.
	dockerPuller DockerPuller

	client DockerInterface
}

// Returns an docker implementation of container.RuntimeImages interface.
func NewDockerRuntimeImages(client DockerInterface, qps float32, burst int) kubecontainer.RuntimeImages {
	return &runtimeImages{
		imageStatsProvider: &imageStatsProvider{client},
		dockerPuller:       newDockerPuller(client, qps, burst),
		client:             client,
	}
}

// PullImage pulls an image from network to local storage.
func (ri *runtimeImages) PullImage(image kubecontainer.ImageSpec, secrets []api.Secret) error {
	return ri.dockerPuller.Pull(image.Image, secrets)
}

// IsImagePresent checks whether the container image is already in the local storage.
func (ri *runtimeImages) IsImagePresent(image kubecontainer.ImageSpec) (bool, error) {
	return ri.dockerPuller.IsImagePresent(image.Image)
}

// Removes the specified image.
func (ri *runtimeImages) RemoveImage(image kubecontainer.ImageSpec) error {
	// TODO(harryz) currently Runtime interface does not provide other remove options.
	_, err := ri.client.RemoveImage(image.Image, dockertypes.ImageRemoveOptions{})
	return err
}

// List all images in the local storage.
func (ri *runtimeImages) ListImages() ([]kubecontainer.Image, error) {
	var images []kubecontainer.Image

	dockerImages, err := ri.client.ListImages(dockertypes.ImageListOptions{})
	if err != nil {
		return images, err
	}

	for _, di := range dockerImages {
		image, err := toRuntimeImage(&di)
		if err != nil {
			continue
		}
		images = append(images, *image)
	}
	return images, nil
}

// imageStatsProvider exposes stats about all images currently available.
type imageStatsProvider struct {
	// Docker remote API client
	c DockerInterface
}

func (isp *imageStatsProvider) ImageStats() (*kubecontainer.ImageStats, error) {
	images, err := isp.c.ListImages(dockertypes.ImageListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list docker images - %v", err)
	}
	// A map of all the image layers to its corresponding size.
	imageMap := sets.NewString()
	ret := &kubecontainer.ImageStats{}
	for _, image := range images {
		// Get information about the various layers of each docker image.
		history, err := isp.c.ImageHistory(image.ID)
		if err != nil {
			glog.V(2).Infof("failed to get history of docker image %v - %v", image, err)
			continue
		}
		// Store size information of each layer.
		for _, layer := range history {
			// Skip empty layers.
			if layer.Size == 0 {
				glog.V(10).Infof("skipping image layer %v with size 0", layer)
				continue
			}
			key := layer.ID
			// Some of the layers are empty.
			// We are hoping that these layers are unique to each image.
			// Still keying with the CreatedBy field to be safe.
			if key == "" || key == "<missing>" {
				key = key + layer.CreatedBy
			}
			if !imageMap.Has(key) {
				ret.TotalStorageBytes += uint64(layer.Size)
			}
			imageMap.Insert(key)
		}
	}
	return ret, nil
}
