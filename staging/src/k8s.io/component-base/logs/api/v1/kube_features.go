/*
Copyright 2022 The Kubernetes Authors.

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

package v1

import (
	"k8s.io/component-base/featuregate"
)

const (
	// owner: @pohly
	// kep: http://kep.k8s.io/3077
	// alpha: v1.24
	//
	// Enables looking up a logger from a context.Context instead of using
	// the global fallback logger and manipulating the logger that is
	// used by a call chain.
	ContextualLogging featuregate.Feature = "ContextualLogging"

	// contextualLoggingDefault must remain false while in alpha. It can
	// become true in beta.
	contextualLoggingDefault = false

	// Allow fine-tuning of experimental, alpha-quality logging options.
	//
	// Per https://groups.google.com/g/kubernetes-sig-architecture/c/Nxsc7pfe5rw/m/vF2djJh0BAAJ
	// we want to avoid a proliferation of feature gates. This feature gate:
	// - will guard *a group* of logging options whose quality level is alpha.
	// - will never graduate to beta or stable.
	LoggingAlphaOptions featuregate.Feature = "LoggingAlphaOptions"

	// Allow fine-tuning of experimental, beta-quality logging options.
	//
	// Per https://groups.google.com/g/kubernetes-sig-architecture/c/Nxsc7pfe5rw/m/vF2djJh0BAAJ
	// we want to avoid a proliferation of feature gates. This feature gate:
	// - will guard *a group* of logging options whose quality level is beta.
	// - is thus *introduced* as beta
	// - will never graduate to stable.
	LoggingBetaOptions featuregate.Feature = "LoggingBetaOptions"

	// Stable logging options. Always enabled.
	LoggingStableOptions featuregate.Feature = "LoggingStableOptions"
)

func featureGates() map[featuregate.Feature]featuregate.FeatureSpec {
	return map[featuregate.Feature]featuregate.FeatureSpec{
		ContextualLogging: {Default: contextualLoggingDefault, PreRelease: featuregate.Alpha},

		LoggingAlphaOptions: {Default: false, PreRelease: featuregate.Alpha},
		LoggingBetaOptions:  {Default: true, PreRelease: featuregate.Beta},
	}
}

// AddFeatureGates adds all feature gates used by this package.
func AddFeatureGates(mutableFeatureGate featuregate.MutableFeatureGate) error {
	return mutableFeatureGate.Add(featureGates())
}
