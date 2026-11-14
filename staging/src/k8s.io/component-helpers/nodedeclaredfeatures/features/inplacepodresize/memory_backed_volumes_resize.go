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

package inplacepodresize

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// Ensure the feature struct implements the unified Feature interface.
var _ types.Feature = &podLevelResourcesResizeFeature{}

// IPPRMemoryBackedVolumesFeatureGate is the feature gate for InPlacePodVerticalScalingMemoryBackedVolumes.
const IPPRMemoryBackedVolumesFeatureGate = "InPlacePodVerticalScalingMemoryBackedVolumes"

// MemoryBackedVolumesResizeFeature is the implementation of the IPPRMemoryBackedVolumes feature.
var MemoryBackedVolumesResizeFeature = &memoryBackedVolumesResizeFeature{}

type memoryBackedVolumesResizeFeature struct{}

func (f *memoryBackedVolumesResizeFeature) Name() string {
	return IPPRMemoryBackedVolumesFeatureGate
}

func (f *memoryBackedVolumesResizeFeature) Discover(cfg *types.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(IPPRMemoryBackedVolumesFeatureGate)
}

func (f *memoryBackedVolumesResizeFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{IPPRMemoryBackedVolumesFeatureGate},
	}
}

func (f *memoryBackedVolumesResizeFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	// This feature is only relevant for pod updates.
	return false
}

func (f *memoryBackedVolumesResizeFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	for i := range newPodInfo.Spec.Volumes {
		newVol := &newPodInfo.Spec.Volumes[i]
		if newVol.EmptyDir == nil || newVol.EmptyDir.Medium != v1.StorageMediumMemory {
			// The volume is not memory backed. Standard validation will catch attempts to change volume type.
			continue
		}
		oldVol := findVolume(oldPodInfo.Spec.Volumes, newVol.Name, i)
		if oldVol == nil {
			continue
		}
		if oldVol.EmptyDir == nil || oldVol.EmptyDir.Medium != v1.StorageMediumMemory {
			// The volume is not memory backed. Standard validation will catch attempts to change volume type.
			continue
		}
		hasOldLimit := oldVol.EmptyDir.SizeLimit != nil
		hasNewLimit := newVol.EmptyDir.SizeLimit != nil
		if hasOldLimit != hasNewLimit {
			return true
		}
		if hasOldLimit && hasNewLimit && !newVol.EmptyDir.SizeLimit.Equal(*oldVol.EmptyDir.SizeLimit) {
			return true
		}
	}

	return false
}

func findVolume(volumes []v1.Volume, name string, idxHint int) *v1.Volume {
	if len(volumes) > idxHint && volumes[idxHint].Name == name {
		return &volumes[idxHint]
	}
	for i := range volumes {
		if volumes[i].Name == name {
			return &volumes[i]
		}
	}
	return nil
}

func (f *memoryBackedVolumesResizeFeature) MaxVersion() *version.Version {
	return nil
}
