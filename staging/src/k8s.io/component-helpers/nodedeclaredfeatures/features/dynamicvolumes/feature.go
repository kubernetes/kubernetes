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

package dynamicvolumes

import (
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// Ensure the feature struct implements the unified Feature interface.
var _ types.Feature = &dynamicNodeLocalEphemeralVolumesFeature{}

// DynamicNodeLocalEphemeralVolumesFeatureGate is the feature gate for DynamicNodeLocalEphemeralVolumes.
const DynamicNodeLocalEphemeralVolumesFeatureGate = "DynamicNodeLocalEphemeralVolumes"

// Feature is the implementation of the DynamicNodeLocalEphemeralVolumes feature.
var Feature = &dynamicNodeLocalEphemeralVolumesFeature{}

type dynamicNodeLocalEphemeralVolumesFeature struct{}

func (f *dynamicNodeLocalEphemeralVolumesFeature) Name() string {
	return DynamicNodeLocalEphemeralVolumesFeatureGate
}

func (f *dynamicNodeLocalEphemeralVolumesFeature) Discover(cfg *types.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(DynamicNodeLocalEphemeralVolumesFeatureGate)
}

func (f *dynamicNodeLocalEphemeralVolumesFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{DynamicNodeLocalEphemeralVolumesFeatureGate},
	}
}

func (f *dynamicNodeLocalEphemeralVolumesFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	// Volumes and container volumeMounts can only be dynamically mutated on update.
	return false
}

// InferForUpdate returns true if the update adds or removes a volume, mutates an existing
// main container's volumeMounts, or adds a container mounting a previously unreferenced volume.
func (f *dynamicNodeLocalEphemeralVolumesFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	oldVolumes, newVolumes := oldPodInfo.Spec.Volumes, newPodInfo.Spec.Volumes
	if len(oldVolumes) != len(newVolumes) {
		return true
	}
	// Validation rejects reordering volumes, so at the same length any name mismatch means a
	// volume was removed and another added.
	for i := range oldVolumes {
		if oldVolumes[i].Name != newVolumes[i].Name {
			return true
		}
	}

	return containerVolumeMountsChanged(oldPodInfo.Spec, newPodInfo.Spec)
}

func containerVolumeMountsChanged(oldSpec, newSpec *v1.PodSpec) bool {
	oldContainers, newContainers := oldSpec.Containers, newSpec.Containers
	mountsUnreferencedVolume := func(mounts []v1.VolumeMount) bool {
		for i := range mounts {
			if !isVolumeMountedInSpec(oldSpec, mounts[i].Name) {
				return true
			}
		}
		return false
	}

	i, j := 0, 0
	for i < len(newContainers) && j < len(oldContainers) {
		c := &newContainers[i]
		if c.Name == oldContainers[j].Name {
			if !apiequality.Semantic.DeepEqual(c.VolumeMounts, oldContainers[j].VolumeMounts) {
				return true
			}
			i++
			j++
			continue
		}
		// Containers aren't reordered; check whether an added container newly mounts a
		// previously unreferenced volume so VolumeManager will process and mount it.
		k := j + 1
		for k < len(oldContainers) && oldContainers[k].Name != c.Name {
			k++
		}
		if k == len(oldContainers) {
			if mountsUnreferencedVolume(c.VolumeMounts) {
				return true
			}
			i++
			continue
		}
		j = k
	}
	for ; i < len(newContainers); i++ {
		if mountsUnreferencedVolume(newContainers[i].VolumeMounts) {
			return true
		}
	}
	return false
}

func isVolumeMountedInSpec(spec *v1.PodSpec, volName string) bool {
	for i := range spec.Containers {
		for j := range spec.Containers[i].VolumeMounts {
			if spec.Containers[i].VolumeMounts[j].Name == volName {
				return true
			}
		}
	}
	for i := range spec.InitContainers {
		for j := range spec.InitContainers[i].VolumeMounts {
			if spec.InitContainers[i].VolumeMounts[j].Name == volName {
				return true
			}
		}
	}
	for i := range spec.EphemeralContainers {
		for j := range spec.EphemeralContainers[i].VolumeMounts {
			if spec.EphemeralContainers[i].VolumeMounts[j].Name == volName {
				return true
			}
		}
	}
	return false
}

func (f *dynamicNodeLocalEphemeralVolumesFeature) MaxVersion() *version.Version {
	return nil
}
