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
	DraOptionalNodeOperations            = "DraOptionalNodeOperations"
)

var Feature = &draOptionalNodeOperationsFeature{}

type draOptionalNodeOperationsFeature struct{}

func (f *draOptionalNodeOperationsFeature) Name() string {
	return DraOptionalNodeOperations
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
	if podInfo == nil {
		return false
	}
	for _, claim := range podInfo.ResourceClaims {
		if claim == nil || claim.Status.Allocation == nil {
			continue
		}
		for _, result := range claim.Status.Allocation.Devices.Results {
			if result.SkipNodeOperations != nil && *result.SkipNodeOperations {
				return true
			}
		}
	}
	return false
}

func (f *draOptionalNodeOperationsFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	return f.InferForScheduling(newPodInfo)
}

func (f *draOptionalNodeOperationsFeature) MaxVersion() *version.Version {
	return nil
}
