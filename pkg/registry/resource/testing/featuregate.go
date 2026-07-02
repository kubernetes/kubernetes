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

package testing

import (
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

var (
	// emulationVersionForFeature defines the version we need to emulate in order run tests
	// for features that have been locked at current version.
	emulationVersionForFeature = map[featuregate.Feature]string{
		features.DRAAdminAccess:            "1.35",
		features.DRAPrioritizedList:        "1.36",
		features.DynamicResourceAllocation: "1.34",
	}
)

// SetMinimumRequiredFeatureGateVersion sets the minimum required feature gate version based on the features that are disabled.
func SetMinimumRequiredFeatureGateVersion(tCtx ktesting.TContext, featuresToConfigure map[featuregate.Feature]bool) {
	var minRequiredVersion *version.Version
	for featureGate, versionStr := range emulationVersionForFeature {
		if enabled, exists := featuresToConfigure[featureGate]; exists && !enabled {
			v := version.MustParse(versionStr)
			if minRequiredVersion == nil || v.LessThan(minRequiredVersion) {
				minRequiredVersion = v
			}
		}
	}
	if minRequiredVersion != nil {
		featuregatetesting.SetFeatureGateEmulationVersionDuringTest(tCtx, utilfeature.DefaultFeatureGate, minRequiredVersion)
	}
}
