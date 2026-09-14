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

package extendwebsocketstokubelet

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// Ensure the feature struct implements the unified Feature interface.
var _ types.Feature = &extendWebSocketsToKubeletFeature{}

const (
	ExtendWebSocketsToKubeletFeatureGate = "ExtendWebSocketsToKubelet"
)

// Feature is the implementation of the `ExtendWebSocketsToKubelet` feature.
var Feature = &extendWebSocketsToKubeletFeature{}

type extendWebSocketsToKubeletFeature struct{}

func (f *extendWebSocketsToKubeletFeature) Name() string {
	return ExtendWebSocketsToKubeletFeatureGate
}

func (f *extendWebSocketsToKubeletFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{ExtendWebSocketsToKubeletFeatureGate},
	}
}

func (f *extendWebSocketsToKubeletFeature) Discover(cfg *types.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(ExtendWebSocketsToKubeletFeatureGate)
}

func (f *extendWebSocketsToKubeletFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	// This feature does not affect scheduling.
	return false
}

func (f *extendWebSocketsToKubeletFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	// This feature does not affect updates.
	return false
}

func (f *extendWebSocketsToKubeletFeature) MaxVersion() *version.Version {
	return nil
}
