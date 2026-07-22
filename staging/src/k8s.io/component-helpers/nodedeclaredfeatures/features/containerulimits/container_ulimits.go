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

package containerulimits

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// Ensure the feature struct implements the unified Feature interface.
var _ types.Feature = &containerUlimitsFeature{}

const (
	// ContainerUlimitsFeatureGate is the feature gate name backing this declared feature.
	ContainerUlimitsFeatureGate = "ContainerUlimits"
	// ContainerUlimits is a declared feature that indicates the runtime supports per-container ulimits.
	ContainerUlimits = "ContainerUlimits"
)

// Feature is the implementation of the `ContainerUlimits` feature.
var Feature = &containerUlimitsFeature{}

type containerUlimitsFeature struct{}

func (f *containerUlimitsFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{ContainerUlimitsFeatureGate},
		RequiredRuntimeFeatures: &types.RuntimeFeatures{
			ContainerUlimits: true,
		},
	}
}

func (f *containerUlimitsFeature) Name() string {
	return ContainerUlimits
}

func (f *containerUlimitsFeature) Discover(cfg *types.NodeConfiguration) bool {
	if cfg.FeatureGates == nil || !cfg.FeatureGates.Enabled(ContainerUlimitsFeatureGate) {
		return false
	}
	return cfg.RuntimeFeatures.ContainerUlimits
}

func (f *containerUlimitsFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	if podInfo == nil || podInfo.Spec == nil {
		return false
	}
	for _, c := range podInfo.Spec.Containers {
		if c.SecurityContext != nil && c.SecurityContext.Ulimits != nil {
			return true
		}
	}
	for _, c := range podInfo.Spec.InitContainers {
		if c.SecurityContext != nil && c.SecurityContext.Ulimits != nil {
			return true
		}
	}
	for _, c := range podInfo.Spec.EphemeralContainers {
		if c.SecurityContext != nil && c.SecurityContext.Ulimits != nil {
			return true
		}
	}
	return false
}

func (f *containerUlimitsFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	return !f.InferForScheduling(oldPodInfo) && f.InferForScheduling(newPodInfo)
}

func (f *containerUlimitsFeature) MaxVersion() *version.Version {
	return nil
}
