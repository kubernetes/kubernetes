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

package perpodpidlimit

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

var _ types.Feature = &perPodPIDLimitFeature{}

const (
	PerPodPIDLimitFeatureGate = "PerPodPIDLimit"
)

// Feature is the implementation of the PerPodPIDLimit declared feature.
var Feature = &perPodPIDLimitFeature{}

type perPodPIDLimitFeature struct{}

func (f *perPodPIDLimitFeature) Name() string {
	return PerPodPIDLimitFeatureGate
}

func (f *perPodPIDLimitFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{PerPodPIDLimitFeatureGate},
	}
}

func (f *perPodPIDLimitFeature) Discover(cfg *types.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(PerPodPIDLimitFeatureGate)
}

func (f *perPodPIDLimitFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	if podInfo.Spec.Resources == nil {
		return false
	}
	_, hasPID := podInfo.Spec.Resources.Limits[v1.ResourcePID]
	return hasPID
}

func (f *perPodPIDLimitFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	// PID limits are immutable after pod creation.
	return false
}

func (f *perPodPIDLimitFeature) MaxVersion() *version.Version {
	return nil
}
