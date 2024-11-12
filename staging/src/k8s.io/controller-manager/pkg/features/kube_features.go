/*
Copyright 2020 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
)

const (
	// Every feature gate should add method here following this template:
	//
	// // owner: @username
	// MyFeature featuregate.Feature = "MyFeature"
	//
	// Feature gates should be listed in alphabetical, case-sensitive
	// (upper before any lower case character) order. This reduces the risk
	// of code conflicts because changes are more likely to be scattered
	// across the file.

	// owner: @nckturner
	// kep:  http://kep.k8s.io/2699
	// Enable webhook in cloud controller manager
	CloudControllerManagerWebhook featuregate.Feature = "CloudControllerManagerWebhook"
)

func SetupCurrentKubernetesSpecificFeatureGates(featuregates featuregate.MutableVersionedFeatureGate) error {
	return featuregates.AddVersioned(versionedCloudPublicFeatureGates)
}

// versionedCloudPublicFeatureGates consists of versioned cloud-specific feature keys.
// To add a new feature, define a key for it above and add it here.
var versionedCloudPublicFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	CloudControllerManagerWebhook: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
	},
}
