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

package dranodeallocatableresources

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

var _ types.Feature = &draNodeAllocatableResourcesFeature{}

const (
	DRANodeAllocatableResourcesFeatureGate = "DRANodeAllocatableResources"
	// The declared name changed from "DRANodeAllocatableResources" to "DRAExtraNodeAllocatableResources" in 1.38, because the pod status API changed between
	// alpha and beta. 1.38+ kubelets use the updated API for cgroup adjustments so pods that use status.extraNodeAllocatableResources must only run on nodes that declare the new name.
	DRANodeAllocatableResourcesFeature = "DRAExtraNodeAllocatableResources"
)

var Feature = &draNodeAllocatableResourcesFeature{}

type draNodeAllocatableResourcesFeature struct{}

func (f *draNodeAllocatableResourcesFeature) Name() string {
	return DRANodeAllocatableResourcesFeature
}

func (f *draNodeAllocatableResourcesFeature) Discover(cfg *types.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(DRANodeAllocatableResourcesFeatureGate)
}

func (f *draNodeAllocatableResourcesFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{DRANodeAllocatableResourcesFeatureGate},
	}
}

func (f *draNodeAllocatableResourcesFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	// Whether a pod needs this feature depends on the devices allocated to its claims, which is not known
	// before Filter. Instead, node.status.declaredFeatures is checked in
	// 1. ResourceSlice admission, which rejects node allocatable devices on nodes without this feature.
	// 2. The DRA scheduler plugin Filter.
	return false
}

func (f *draNodeAllocatableResourcesFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	// This feature does not affect pod updates.
	return false
}

func (f *draNodeAllocatableResourcesFeature) MaxVersion() *version.Version {
	return nil
}
