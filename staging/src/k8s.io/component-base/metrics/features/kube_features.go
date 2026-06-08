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
	internalmetrics "k8s.io/component-base/metrics/internal"
)

const (
	// owner: @richabanker
	// kep: https://kep.k8s.io/5808
	// alpha: v1.36
	//
	// Enables native histogram support for Kubernetes metrics.
	// When enabled, histogram metrics will be exposed in both classic
	// and native histogram formats.
	NativeHistograms featuregate.Feature = "NativeHistograms"
)

func featureGates() map[featuregate.Feature]featuregate.VersionedSpecs {
	return map[featuregate.Feature]featuregate.VersionedSpecs{
		NativeHistograms: {
			{Version: version.MustParse("1.36"), Default: false, PreRelease: featuregate.Alpha},
		},
	}
}

// AddFeatureGates adds all feature gates used by this package.
func AddFeatureGates(mutableFeatureGate featuregate.MutableVersionedFeatureGate) error {
	return mutableFeatureGate.AddVersioned(featureGates())
}

// ApplyFeatureGates propagates the current feature gate state to the metrics
// subsystem. It must be called after feature gates are finalised and before
// any histogram metrics are registered.
func ApplyFeatureGates(featureGate featuregate.FeatureGate) {
	internalmetrics.SetNativeHistogramsEnabled(featureGate.Enabled(NativeHistograms))
}
