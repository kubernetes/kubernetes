/*
Copyright The Kubernetes Authors.

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

package state

import (
	"encoding/json"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
)

// PodResourceInfoV1 is the legacy checkpoint structure used in v1.37 and earlier.
type PodResourceInfoV1 struct {
	ContainerResources   map[string]v1.ResourceRequirements `json:"ContainerResources,omitempty"`
	PodLevelResources    *v1.ResourceRequirements           `json:"PodLevelResources,omitempty"`
	EmptyDirVolumeLimits map[string]*resource.Quantity      `json:"EmptyDirVolumeLimits,omitempty"`
}

type PodResourceInfoMapV1 map[types.UID]PodResourceInfoV1

type PodResourceCheckpointInfoV1 struct {
	Entries PodResourceInfoMapV1 `json:"entries,omitempty"`
}

// GetPodResourceCheckpointInfoV1 returns legacy Pod Resource Allocation info states from checkpoint
func (cp *Checkpoint) GetPodResourceCheckpointInfoV1() (*PodResourceCheckpointInfoV1, error) {
	var data PodResourceCheckpointInfoV1
	if err := json.Unmarshal([]byte(cp.Data), &data); err != nil {
		return nil, err
	}

	return &data, nil
}

// migrateV1ToV2 converts checkpoints from the v1 format to the v2 format
func migrateV1ToV2(entries PodResourceInfoMapV1) PodResourceInfoMap {
	v2Map := make(PodResourceInfoMap)
	for uid, info := range entries {
		podSpec := &v1.PodSpec{}

		// Migrate container resources
		if len(info.ContainerResources) > 0 {
			podSpec.Containers = make([]v1.Container, 0, len(info.ContainerResources))
			for name, res := range info.ContainerResources {
				podSpec.Containers = append(podSpec.Containers, v1.Container{
					Name:      name,
					Resources: *res.DeepCopy(),
				})
			}
		}

		// Migrate pod-level resources
		if info.PodLevelResources != nil {
			podSpec.Resources = info.PodLevelResources.DeepCopy()
		}

		// Migrate emptyDir volume limits
		if len(info.EmptyDirVolumeLimits) > 0 {
			podSpec.Volumes = make([]v1.Volume, 0, len(info.EmptyDirVolumeLimits))
			for name, limit := range info.EmptyDirVolumeLimits {
				if limit != nil {
					copyLimit := limit.DeepCopy()
					podSpec.Volumes = append(podSpec.Volumes, v1.Volume{
						Name: name,
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{
								SizeLimit: &copyLimit,
							},
						},
					})
				}
			}
		}

		v2Map[uid] = PodResourceInfo{
			PodSpec: podSpec,
		}
	}
	return v2Map
}
