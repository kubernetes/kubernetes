/*
Copyright 2024 The Kubernetes Authors.

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
	"fmt"

	clientfeatures "k8s.io/client-go/features"
	"k8s.io/component-base/featuregate"
)

// clientAdapter adapts a k8s.io/component-base/featuregate.MutableFeatureGate to client-go's
// feature Gate and Registry interfaces. The component-base types Feature, FeatureSpec, and
// prerelease, and the component-base prerelease constants, are duplicated by parallel types and
// constants in client-go. The parallel types exist to allow the feature gate mechanism to be used
// for client-go features without introducing a circular dependency between component-base and
// client-go.
type clientAdapter struct {
	mfg featuregate.MutableFeatureGate
}

var _ clientfeatures.Gates = &clientAdapter{}

func (a *clientAdapter) Enabled(name clientfeatures.Feature) bool {
	return a.mfg.Enabled(featuregate.Feature(name))
}

var _ clientfeatures.Registry = &clientAdapter{}

func (a *clientAdapter) Add(in map[clientfeatures.Feature]clientfeatures.FeatureSpec) error {
	out := map[featuregate.Feature]featuregate.FeatureSpec{}
	for name, spec := range in {
		converted := featuregate.FeatureSpec{
			Default:       spec.Default,
			LockToDefault: spec.LockToDefault,
		}
		switch spec.PreRelease {
		case clientfeatures.Alpha:
			converted.PreRelease = featuregate.Alpha
		case clientfeatures.Beta:
			converted.PreRelease = featuregate.Beta
		case clientfeatures.GA:
			converted.PreRelease = featuregate.GA
		case clientfeatures.Deprecated:
			converted.PreRelease = featuregate.Deprecated
		default:
			// The default case implies programmer error.  The same set of prerelease
			// constants must exist in both component-base and client-go, and each one
			// must have a case here.
			panic(fmt.Sprintf("unrecognized prerelease %q of feature %q", spec.PreRelease, name))
		}
		out[featuregate.Feature(name)] = converted
	}
	return a.mfg.Add(out)
}
