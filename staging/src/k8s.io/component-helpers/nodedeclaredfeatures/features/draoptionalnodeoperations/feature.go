/*
Copyright 2025 The Kubernetes Authors.

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

package draoptionalnodeoperations

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

var _ types.Feature = &draOptionalNodeOperationsFeature{}

const (
	DRAOptionalNodeOperationsFeatureGate = "DRAOptionalNodeOperations"
)

var Feature = &draOptionalNodeOperationsFeature{}

type draOptionalNodeOperationsFeature struct{}

func (f *draOptionalNodeOperationsFeature) Name() string {
	return DRAOptionalNodeOperationsFeatureGate
}

func (f *draOptionalNodeOperationsFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{DRAOptionalNodeOperationsFeatureGate},
	}
}

func (f *draOptionalNodeOperationsFeature) Discover(cfg *types.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(DRAOptionalNodeOperationsFeatureGate)
}

func (f *draOptionalNodeOperationsFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	// ResourceClaim status allocation is not set until prebind, so the NDF
	// plugin cannot infer this feature requirement at the PreFilter stage.
	// Filtering for claims requiring optional node operations is handled
	// directly by the dynamicresources scheduler plugin.
	return false
}

func (f *draOptionalNodeOperationsFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	return false
}

func (f *draOptionalNodeOperationsFeature) MaxVersion() *version.Version {
	return nil
}
