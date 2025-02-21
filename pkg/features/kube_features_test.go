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
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientfeatures "k8s.io/client-go/features"
	"k8s.io/component-base/featuregate"
)

// TestKubeFeaturesRegistered tests that all kube features are registered.
func TestKubeFeaturesRegistered(t *testing.T) {
	registeredFeatures := utilfeature.DefaultFeatureGate.DeepCopy().GetAll()

	for featureName := range defaultKubernetesFeatureGates {
		if _, ok := registeredFeatures[featureName]; !ok {
			t.Errorf("The feature gate %q is not registered in the DefaultFeatureGate", featureName)
		}
	}
}

// TestClientFeaturesRegistered tests that all client features are registered.
func TestClientFeaturesRegistered(t *testing.T) {
	onlyClientFg := featuregate.NewFeatureGate()
	if err := clientfeatures.AddFeaturesToExistingFeatureGates(&clientAdapter{onlyClientFg}); err != nil {
		t.Fatal(err)
	}
	registeredFeatures := utilfeature.DefaultFeatureGate.DeepCopy().GetAll()

	for featureName := range onlyClientFg.GetAll() {
		if _, ok := registeredFeatures[featureName]; !ok {
			t.Errorf("The client-go's feature gate %q is not registered in the DefaultFeatureGate", featureName)
		}
	}
}

// TestAllRegisteredFeaturesExpected tests that the set of features actually registered does not
// include any features other than those on the list in this package or in client-go's feature
// package.
func TestAllRegisteredFeaturesExpected(t *testing.T) {
	registeredFeatures := utilfeature.DefaultFeatureGate.DeepCopy().GetAll()
	knownFeatureGates := featuregate.NewFeatureGate()
	if err := clientfeatures.AddFeaturesToExistingFeatureGates(&clientAdapter{knownFeatureGates}); err != nil {
		t.Fatal(err)
	}
	if err := knownFeatureGates.Add(defaultKubernetesFeatureGates); err != nil {
		t.Fatal(err)
	}
	if err := knownFeatureGates.AddVersioned(defaultVersionedKubernetesFeatureGates); err != nil {
		t.Fatal(err)
	}
	knownFeatures := knownFeatureGates.GetAll()

	for registeredFeature := range registeredFeatures {
		if _, ok := knownFeatures[registeredFeature]; !ok {
			t.Errorf("The feature gate %q is not from known feature gates", registeredFeature)
		}
	}
}
func TestEnsureAlphaGatesAreNotSwitchedOnByDefault(t *testing.T) {
	checkAlphaGates := func(feature featuregate.Feature, spec featuregate.FeatureSpec) {
		// FIXME(dims): remove this check when WindowsHostNetwork is fixed up or removed
		// entirely. Please do NOT add more entries here.
		if feature == "WindowsHostNetwork" {
			return
		}
		if spec.PreRelease == featuregate.Alpha && spec.Default {
			t.Errorf("The alpha feature gate %q is switched on by default", feature)
		}
		if spec.PreRelease == featuregate.Alpha && spec.LockToDefault {
			t.Errorf("The alpha feature gate %q is locked to default", feature)
		}
	}

	for feature, spec := range defaultKubernetesFeatureGates {
		checkAlphaGates(feature, spec)
	}
	for feature, specs := range defaultVersionedKubernetesFeatureGates {
		for _, spec := range specs {
			checkAlphaGates(feature, spec)
		}
	}
}
