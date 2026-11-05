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
	"reflect"

	v1 "k8s.io/api/core/v1"
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
	// Dynamic volume additions and removals only occur on pod updates.
	return false
}

func (f *dynamicNodeLocalEphemeralVolumesFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	if len(oldPodInfo.Spec.Volumes) != len(newPodInfo.Spec.Volumes) {
		return true
	}

	oldVolNames := make(map[string]struct{}, len(oldPodInfo.Spec.Volumes))
	for i := range oldPodInfo.Spec.Volumes {
		oldVolNames[oldPodInfo.Spec.Volumes[i].Name] = struct{}{}
	}
	for i := range newPodInfo.Spec.Volumes {
		if _, exists := oldVolNames[newPodInfo.Spec.Volumes[i].Name]; !exists {
			return true
		}
	}

	return containerVolumeMountsChanged(oldPodInfo.Spec.Containers, newPodInfo.Spec.Containers)
}

func containerVolumeMountsChanged(oldContainers, newContainers []v1.Container) bool {
	oldByName := make(map[string]*v1.Container, len(oldContainers))
	for i := range oldContainers {
		oldByName[oldContainers[i].Name] = &oldContainers[i]
	}
	for i := range newContainers {
		oldC, exists := oldByName[newContainers[i].Name]
		if !exists {
			continue
		}
		if !reflect.DeepEqual(oldC.VolumeMounts, newContainers[i].VolumeMounts) {
			return true
		}
	}
	return false
}

func (f *dynamicNodeLocalEphemeralVolumesFeature) MaxVersion() *version.Version {
	return nil
}
