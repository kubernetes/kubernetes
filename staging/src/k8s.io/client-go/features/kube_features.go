/*
Copyright 2023 The Kubernetes Authors.
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

package features

import (
	"k8s.io/internal/feature"
)

const (
	// Every feature gate should add method here following this template:
	//
	// // owner: @username
	// // alpha: v1.4
	// MyFeature featuregate.Feature = "MyFeature"
	//
	// Feature gates should be listed in alphabetical, case-sensitive
	// (upper before any lower case character) order. This reduces the risk
	// of code conflicts because changes are more likely to be scattered
	// across the file.

	// owner: @p0lyn0mial
	// alpha: v1.27
	// beta: v1.29
	//
	// Allow the API server to stream individual items instead of chunking
	WatchList feature.Feature = "WatchListClient"
)

// AddFeaturesToExistingFeatureGates adds the default feature gates to the provided set.
// Usually this function is combined with SetFeatureGates to take control of the
// features exposed by this library.
func AddFeaturesToExistingFeatureGates(mutableFeatureGates feature.MutableFeatureGate) error {
	return mutableFeatureGates.Add(defaultKubernetesFeatureGates)
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
//
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[feature.Feature]feature.FeatureSpec{
	WatchList: {Default: false, PreRelease: feature.Beta},
}
