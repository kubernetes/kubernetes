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
	"sync/atomic"

	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/component-base/featuregate"
)

// DefaultFeatureGates returns the feature gates exposed by this library.
//
// By default, only the default features gate will be returned.
// The default implementation allows controlling the features
// via environmental variables.
// For example, if you have a feature named "MyFeature,"
// setting an environmental variable "KUBE_FEATURE_MyFeature"
// will allow you to configure the state of that feature.
//
// Please note that the actual set of the features gates
// might be overwritten by calling SetFeatureGates method.
func DefaultFeatureGates() featuregate.Reader {
	return featureGates.Load().(featuregate.Reader)
}

// AddFeaturesToExistingFeatureGates adds the default feature gates to the provided set.
// Usually this function is combined with SetFeatureGates to take control of the
// features exposed by this library.
func AddFeaturesToExistingFeatureGates(mutableFeatureGates featuregate.MutableFeatureGate) error {
	return mutableFeatureGates.Add(defaultKubernetesFeatureGates)
}

// SetFeatureGates overwrites the default implementation of the feature gate
// used by this library.
//
// Useful for binaries that would like to have full control of the features
// exposed by this library. For example to allow consumers of a binary
// to interact with the features via a command line flag.
func SetFeatureGates(newFeatureGates featuregate.FeatureGate) {
	wrappedFeatureGate := &featureGateWrapper{newFeatureGates}
	featureGates.Store(wrappedFeatureGate)
}

func init() {
	envVarGates := featuregate.NewEnvVarFeatureGate()
	runtime.Must(envVarGates.Add(defaultKubernetesFeatureGates))

	wrappedFeatureGate := &featureGateWrapper{envVarGates}
	featureGates.Store(wrappedFeatureGate)
}

// featureGateWrapper a thin wrapper to satisfy featureGates variable (atomic.Value)
type featureGateWrapper struct {
	featuregate.Reader
}

var (
	// featureGates is a shared global FeatureGate.
	//
	// Top-level commands/options setup that needs to modify this feature gate
	// should use AddFeaturesToExistingFeatureGates followed by SetFeatureGates.
	featureGates = &atomic.Value{}
)

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
//
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{}
