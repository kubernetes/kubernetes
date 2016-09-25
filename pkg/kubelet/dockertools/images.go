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

package dockertools

import (
	"fmt"
	"sync"

	"github.com/golang/glog"

	dockertypes "github.com/docker/engine-api/types"
	runtime "k8s.io/kubernetes/pkg/kubelet/container"
)

// imageStatsProvider exposes stats about all images currently available.
type imageStatsProvider struct {
	sync.Mutex
	// layers caches the current layers, key is the layer ID.
	layers map[string]*dockertypes.ImageHistory
	// imageToLayerIDs maps image to its layer IDs.
	imageToLayerIDs map[string][]string
	// Docker remote API client
	c DockerInterface
}

func newImageStatsProvider(c DockerInterface) *imageStatsProvider {
	return &imageStatsProvider{
		layers:          make(map[string]*dockertypes.ImageHistory),
		imageToLayerIDs: make(map[string][]string),
		c:               c,
	}
}

func (isp *imageStatsProvider) ImageStats() (*runtime.ImageStats, error) {
	images, err := isp.c.ListImages(dockertypes.ImageListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list docker images - %v", err)
	}
	// Take the lock to protect the cache
	isp.Lock()
	defer isp.Unlock()
	// Create new cache each time, this is a little more memory consuming, but:
	// * ImageStats is only called every 10 seconds
	// * We use pointers and reference to copy cache elements.
	// The memory usage should be acceptable.
	// TODO(random-liu): Add more logic to implement in place cache update.
	newLayers := make(map[string]*dockertypes.ImageHistory)
	newImageToLayerIDs := make(map[string][]string)
	for _, image := range images {
		layerIDs, ok := isp.imageToLayerIDs[image.ID]
		if !ok {
			// Get information about the various layers of the given docker image.
			history, err := isp.c.ImageHistory(image.ID)
			if err != nil {
				// Skip the image and inspect again in next ImageStats if the image is still there
				glog.V(2).Infof("failed to get history of docker image %+v - %v", image, err)
				continue
			}
			// Cache each layer
			for i := range history {
				layer := &history[i]
				key := layer.ID
				// Some of the layers are empty.
				// We are hoping that these layers are unique to each image.
				// Still keying with the CreatedBy field to be safe.
				if key == "" || key == "<missing>" {
					key = key + layer.CreatedBy
				}
				layerIDs = append(layerIDs, key)
				newLayers[key] = layer
			}
		} else {
			for _, layerID := range layerIDs {
				newLayers[layerID] = isp.layers[layerID]
			}
		}
		newImageToLayerIDs[image.ID] = layerIDs
	}
	ret := &runtime.ImageStats{}
	// Calculate the total storage bytes
	for _, layer := range newLayers {
		ret.TotalStorageBytes += uint64(layer.Size)
	}
	// Update current cache
	isp.layers = newLayers
	isp.imageToLayerIDs = newImageToLayerIDs
	return ret, nil
}
