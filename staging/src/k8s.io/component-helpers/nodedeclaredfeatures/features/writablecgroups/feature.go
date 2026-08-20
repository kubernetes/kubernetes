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

package writablecgroups

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

var _ types.Feature = &writableCgroupsFeature{}

const (
	WritableCgroupsFeatureGate = "CgroupOptions"
	WritableCgroupsFeatureName = "CgroupOptions"
)

var Feature = &writableCgroupsFeature{}

type writableCgroupsFeature struct{}

func (f *writableCgroupsFeature) Name() string {
	return WritableCgroupsFeatureName
}

func (f *writableCgroupsFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{WritableCgroupsFeatureGate},
		RequiredRuntimeFeatures: &types.RuntimeFeatures{
			SupportsCgroupMountMode: true,
		},
	}
}

func (f *writableCgroupsFeature) Discover(cfg *types.NodeConfiguration) bool {
	if !cfg.FeatureGates.Enabled(WritableCgroupsFeatureGate) {
		return false
	}
	return cfg.RuntimeFeatures.SupportsCgroupMountMode
}

func (f *writableCgroupsFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	requiresWritableCgroups := func(sc *v1.SecurityContext) bool {
		return sc != nil && sc.CgroupOptions != nil && sc.CgroupOptions.MountMode != nil &&
			*sc.CgroupOptions.MountMode == v1.CgroupMountModeWritable
	}
	for i := range podInfo.Spec.Containers {
		if requiresWritableCgroups(podInfo.Spec.Containers[i].SecurityContext) {
			return true
		}
	}
	for i := range podInfo.Spec.InitContainers {
		if requiresWritableCgroups(podInfo.Spec.InitContainers[i].SecurityContext) {
			return true
		}
	}
	for i := range podInfo.Spec.EphemeralContainers {
		if requiresWritableCgroups(podInfo.Spec.EphemeralContainers[i].SecurityContext) {
			return true
		}
	}
	return false
}

func (f *writableCgroupsFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	return false
}

func (f *writableCgroupsFeature) MaxVersion() *version.Version {
	return nil
}
