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

package volumebindmountoptions

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

var _ types.Feature = &volumeBindMountOptionsFeature{}

const (
	VolumeBindMountOptionsFeatureGate = "VolumeBindMountOptions"
	VolumeBindMountOptionsFeatureName = "VolumeBindMountOptions"
)

var Feature = &volumeBindMountOptionsFeature{}

type volumeBindMountOptionsFeature struct{}

func (f *volumeBindMountOptionsFeature) Name() string {
	return VolumeBindMountOptionsFeatureName
}

func (f *volumeBindMountOptionsFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{VolumeBindMountOptionsFeatureGate},
		RequiredRuntimeFeatures: &types.RuntimeFeatures{
			MountOptions: true,
		},
	}
}

func (f *volumeBindMountOptionsFeature) Discover(cfg *types.NodeConfiguration) bool {
	if !cfg.FeatureGates.Enabled(VolumeBindMountOptionsFeatureGate) {
		return false
	}
	return cfg.RuntimeFeatures.MountOptions
}

func (f *volumeBindMountOptionsFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	for i := range podInfo.Spec.Containers {
		for j := range podInfo.Spec.Containers[i].VolumeMounts {
			if len(podInfo.Spec.Containers[i].VolumeMounts[j].BindMountOptions) > 0 {
				return true
			}
		}
	}
	for i := range podInfo.Spec.InitContainers {
		for j := range podInfo.Spec.InitContainers[i].VolumeMounts {
			if len(podInfo.Spec.InitContainers[i].VolumeMounts[j].BindMountOptions) > 0 {
				return true
			}
		}
	}
	for i := range podInfo.Spec.EphemeralContainers {
		for j := range podInfo.Spec.EphemeralContainers[i].VolumeMounts {
			if len(podInfo.Spec.EphemeralContainers[i].VolumeMounts[j].BindMountOptions) > 0 {
				return true
			}
		}
	}
	return false
}

func (f *volumeBindMountOptionsFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	return false
}

func (f *volumeBindMountOptionsFeature) MaxVersion() *version.Version {
	return nil
}
