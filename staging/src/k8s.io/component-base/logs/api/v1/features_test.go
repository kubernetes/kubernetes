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

var (
	// pre-defined feature gates with the features from this package in a
	// certain state (default, all enabled, all disabled).
	defaultFeatureGate, enabledFeatureGate, disabledFeatureGate featuregate.FeatureGate
)

func init() {
	mutable := featuregate.NewFeatureGate()
	if err := AddFeatureGates(mutable); err != nil {
		panic(err)
	}
	defaultFeatureGate = mutable
	enabled := mutable.DeepCopy()
	disabled := mutable.DeepCopy()
	for feature := range mutable.GetAll() {
		if err := enabled.SetFromMap(map[string]bool{string(feature): true}); err != nil {
			panic(err)
		}
		if err := disabled.SetFromMap(map[string]bool{string(feature): false}); err != nil {
			panic(err)
		}
	}
	enabledFeatureGate = enabled
	disabledFeatureGate = disabled
}
