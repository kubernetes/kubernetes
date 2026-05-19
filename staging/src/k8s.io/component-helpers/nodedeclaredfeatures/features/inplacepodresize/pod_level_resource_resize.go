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

package inplacepodresize

import (
	"reflect"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures"
)

// Ensure the feature struct implements the unified Feature interface.
var _ nodedeclaredfeatures.Feature = &podLevelResourcesResizeFeature{}

// IPPRPodLevelResourcesFeatureGate is the feature gate for IPPRPodLevelResourcesVerticalScaling.
const IPPRPodLevelResourcesFeatureGate = "InPlacePodLevelResourcesVerticalScaling"

// Feature is the implementation of the `PodLevelResourcesResize` feature.
var PodLevelResourcesResizeFeature = &podLevelResourcesResizeFeature{}

type podLevelResourcesResizeFeature struct{}

func (f *podLevelResourcesResizeFeature) Name() string {
	return IPPRPodLevelResourcesFeatureGate
}

func (f *podLevelResourcesResizeFeature) Discover(cfg *nodedeclaredfeatures.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(IPPRPodLevelResourcesFeatureGate)
}

func (f *podLevelResourcesResizeFeature) InferForScheduling(podInfo *nodedeclaredfeatures.PodInfo) bool {
	// This feature is only relevant for pod updates.
	return false
}

func (f *podLevelResourcesResizeFeature) InferForUpdate(oldPodInfo, newPodInfo *nodedeclaredfeatures.PodInfo) bool {
	if oldPodInfo.Spec.Resources == nil && newPodInfo.Spec.Resources == nil {
		return false
	}
	return !reflect.DeepEqual(oldPodInfo.Spec.Resources, newPodInfo.Spec.Resources)
}

func (f *podLevelResourcesResizeFeature) MaxVersion() *version.Version {
	return nil
}
