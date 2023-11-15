/*
Copyright 2023 The Kubernetes Authors.

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
	"k8s.io/component-base/featuregate"
)

// Provider defines methods for interacting with feature gates.
type Provider interface {
	// Enabled returns true if the key is enabled.
	Enabled(key featuregate.Feature) bool
}

// DefaultFeatureGates returns the feature gates exposed by this library.
//
// By default, only the default features gate will be returned.
// The default implementation allows controlling the features
// via environmental variables.
// For example, if you have a feature named "MyFeature,"
// setting an environmental variable "KUBE_FEATURE_MyFeature"
// will allow you to configure the state of that feature.
func DefaultFeatureGates() Provider {
	// this would have to be impl as a singleton
	return &dummyProvider{}
}

type dummyProvider struct{}

func (p *dummyProvider) Enabled(key featuregate.Feature) bool {
	return true
}
