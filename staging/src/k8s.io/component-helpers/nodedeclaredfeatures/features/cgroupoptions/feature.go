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

package cgroupoptions

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// Ensure the feature struct implements the unified Feature interface.
var _ types.Feature = &cgroupOptionsFeature{}

const (
	// CgroupOptionsFeatureGate is the feature gate name.
	CgroupOptionsFeatureGate = "CgroupOptions"
	// CgroupOptionsFeatureName is the declared feature name.
	CgroupOptionsFeatureName = "CgroupOptions"
)

// Feature is the implementation of the `CgroupOptions` feature.
var Feature = &cgroupOptionsFeature{}

type cgroupOptionsFeature struct{}

func (f *cgroupOptionsFeature) Name() string {
	return CgroupOptionsFeatureName
}

func (f *cgroupOptionsFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{CgroupOptionsFeatureGate},
		RequiredRuntimeFeatures: &types.RuntimeFeatures{
			CgroupMountMode: true,
		},
		RequiredStaticConfig: &types.StaticConfiguration{
			Cgroup2UnifiedMode: true,
			CgroupsPerQOS:      true,
			CgroupNsdelegate:   true,
		},
	}
}

func (f *cgroupOptionsFeature) Discover(cfg *types.NodeConfiguration) bool {
	if !cfg.FeatureGates.Enabled(CgroupOptionsFeatureGate) {
		return false
	}
	// One feature covers both mount modes. Its requirements are those of
	// Writable: cgroup v2 mounted with nsdelegate, and a cgroup per pod
	// (CgroupsPerQOS) for the descendant and depth limits.
	return cfg.StaticConfig.Cgroup2UnifiedMode &&
		cfg.StaticConfig.CgroupNsdelegate &&
		cfg.StaticConfig.CgroupsPerQOS &&
		cfg.RuntimeFeatures.CgroupMountMode
}

func (f *cgroupOptionsFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	for i := range podInfo.Spec.Containers {
		if requestsCgroupMountMode(podInfo.Spec.Containers[i].SecurityContext) {
			return true
		}
	}
	for i := range podInfo.Spec.InitContainers {
		if requestsCgroupMountMode(podInfo.Spec.InitContainers[i].SecurityContext) {
			return true
		}
	}
	return false
}

func (f *cgroupOptionsFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	// Static Pod updates can change the security context without changing the UID.
	return !f.InferForScheduling(oldPodInfo) && f.InferForScheduling(newPodInfo)
}

func (f *cgroupOptionsFeature) MaxVersion() *version.Version {
	return nil
}

func requestsCgroupMountMode(sc *v1.SecurityContext) bool {
	return sc != nil && sc.CgroupOptions != nil && sc.CgroupOptions.MountMode != nil
}
