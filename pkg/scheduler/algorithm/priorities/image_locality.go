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

package priorities

import (
	"fmt"

	"k8s.io/api/core/v1"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/pkg/scheduler/schedulercache"
)

// This is a reasonable size range of all container images. 90%ile of images on dockerhub drops into this range.
const (
	mb         int64 = 1024 * 1024
	minImgSize int64 = 23 * mb
	maxImgSize int64 = 1000 * mb
)

// ImageLocalityPriorityMap is a priority function that favors nodes that already have requested pod container's images.
// It will detect whether the requested images are present on a node, and then calculate a score ranging from 0 to 10
// based on the total size of those images.
// - If none of the images are present, this node will be given the lowest priority.
// - If some of the images are present on a node, the larger their sizes' sum, the higher the node's priority.
func ImageLocalityPriorityMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	node := nodeInfo.Node()
	if node == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("node not found")
	}

	sumSize := totalImageSize(nodeInfo, pod.Spec.Containers)

	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: calculateScoreFromSize(sumSize),
	}, nil
}

// calculateScoreFromSize calculates the priority of a node. sumSize is sum size of requested images on this node.
// 1. Split image size range into 10 buckets.
// 2. Decide the priority of a given sumSize based on which bucket it belongs to.
func calculateScoreFromSize(sumSize int64) int {
	switch {
	case sumSize == 0 || sumSize < minImgSize:
		// 0 means none of the images required by this pod are present on this
		// node or the total size of the images present is too small to be taken into further consideration.
		return 0

	case sumSize >= maxImgSize:
		// If existing images' total size is larger than max, just make it highest priority.
		return schedulerapi.MaxPriority
	}

	return int((int64(schedulerapi.MaxPriority) * (sumSize - minImgSize) / (maxImgSize - minImgSize)) + 1)
}

// totalImageSize returns the total image size of all the containers that are already on the node.
func totalImageSize(nodeInfo *schedulercache.NodeInfo, containers []v1.Container) int64 {
	var total int64

	imageSizes := nodeInfo.Images()
	for _, container := range containers {
		if size, ok := imageSizes[container.Image]; ok {
			total += size
		}
	}

	return total
}
