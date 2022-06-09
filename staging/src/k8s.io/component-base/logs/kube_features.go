/*
Copyright 2021 The Kubernetes Authors.

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

package logs

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
)

func featureGates() map[featuregate.Feature]featuregate.FeatureSpec {
	return map[featuregate.Feature]featuregate.FeatureSpec{
		ContextualLogging: {Default: contextualLoggingDefault, PreRelease: featuregate.Alpha},
	}
}

// AddFeatureGates adds all feature gates used by this package.
func AddFeatureGates(mutableFeatureGate featuregate.MutableFeatureGate) error {
	return mutableFeatureGate.Add(featureGates())
}
