/*
Copyright 2017 The Kubernetes Authors.

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
	"k8s.io/component-base/featuregate"
)

const (
	// Every feature gate should add method here following this template:
	//
	// // owner: @username
	// // alpha: v1.4
	// MyFeature() bool

	// owner: @alexzielenski
	// alpha: v1.28
	//
	// Ignores errors raised on unchanged fields of Custom Resources
	// across UPDATE/PATCH requests.
	CRDValidationRatcheting featuregate.Feature = "CRDValidationRatcheting"

	// owner: @jpbetz
	// alpha: v1.30
	//
	// CustomResourceDefinitions may include SelectableFields to declare which fields
	// may be used as field selectors.
	CustomResourceFieldSelectors featuregate.Feature = "CustomResourceFieldSelectors"
)

func init() {
	utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates)
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	CRDValidationRatcheting:      {Default: true, PreRelease: featuregate.Beta},
	CustomResourceFieldSelectors: {Default: false, PreRelease: featuregate.Alpha},
}
