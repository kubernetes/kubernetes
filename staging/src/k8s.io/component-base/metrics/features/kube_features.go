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

package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
)

const (
	// owner: @logicalhan
	// kep: https://kep.k8s.io/3466
	ComponentSLIs featuregate.Feature = "ComponentSLIs"
)

func featureGates() map[featuregate.Feature]featuregate.VersionedSpecs {
	return map[featuregate.Feature]featuregate.VersionedSpecs{
		ComponentSLIs: {
			{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
			{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
			// ComponentSLIs officially graduated to GA in v1.29 but the gate was not updated until v1.32.
			// To support emulated versions, keep the gate until v1.35.
			{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
		},
	}
}

// AddFeatureGates adds all feature gates used by this package.
func AddFeatureGates(mutableFeatureGate featuregate.MutableVersionedFeatureGate) error {
	return mutableFeatureGate.AddVersioned(featureGates())
}
