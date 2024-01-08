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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"testing"
)

func TestKubeFeatures(t *testing.T) {
	features := utilfeature.DefaultFeatureGate.DeepCopy().GetAll()

	for i := range features {
		featureName := string(i)

		if featureName == "AllAlpha" || featureName == "AllBeta" {
			continue
		}

		if _, ok := defaultKubernetesFeatureGates[i]; !ok {
			t.Errorf("The feature gate %q is not registered", featureName)
		}
	}
}
