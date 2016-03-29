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
	runtime "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/sets"
)

// imageStatsProvider exposes stats about all images currently available.
type imageStatsProvider struct {
	// Docker remote API client
	c DockerInterface
}

func (isp *imageStatsProvider) ImageStats() (*runtime.ImageStats, error) {
	images, err := isp.c.ListImages(dockertypes.ImageListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list docker images - %v", err)
	}
	// A map of all the image layers to its corresponding size.
	imageMap := sets.NewString()
	ret := &runtime.ImageStats{}
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
