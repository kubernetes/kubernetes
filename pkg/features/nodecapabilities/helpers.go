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

package nodecapabilities

import (
	"fmt"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	nodecapabilitieslib "k8s.io/component-helpers/nodecapabilities"
)

// BuildFeatureDependency builds a FeatureDependency from a feature gate.
func BuildFeatureDependency(fg featuregate.Feature) nodecapabilitieslib.FeatureDependency {
	versionedSpecs, ok := utilfeature.DefaultMutableFeatureGate.GetAllVersioned()[fg]
	if !ok || len(versionedSpecs) == 0 {
		panic(fmt.Sprintf("feature %s is not registered or has no versioned specs", fg))
	}
	fd := nodecapabilitieslib.FeatureDependency{
		FeatureGate:  string(fg),
		IsGA:         false,
		IsDeprecated: false,
	}
	for _, spec := range versionedSpecs {
		if spec.PreRelease == featuregate.GA {
			fd.IsGA = true
			fd.GAVersion = spec.Version
		}
		if spec.PreRelease == featuregate.Deprecated {
			fd.IsDeprecated = true
		}
	}
	return fd
}
