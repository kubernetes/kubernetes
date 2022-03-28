/*
Copyright 2020 The Kubernetes Authors.

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

const (
	// Every feature gate should add method here following this template:
	//
	// // owner: @username
	// // alpha: v1.4
	// MyFeature() bool

	// owner: @khenidak
	// alpha: v1.15
	//
	// Enables ipv6 dual stack
	// Original copy from k8s.io/kubernetes/pkg/features/kube_features.go
	IPv6DualStack featuregate.Feature = "IPv6DualStack"

	// owner: @jiahuif
	// alpha: v1.21
	// beta:  v1.22
	// GA:    v1.24
	//
	// Enables Leader Migration for kube-controller-manager and cloud-controller-manager
	// copied and sync'ed from k8s.io/kubernetes/pkg/features/kube_features.go
	ControllerManagerLeaderMigration featuregate.Feature = "ControllerManagerLeaderMigration"
)

func SetupCurrentKubernetesSpecificFeatureGates(featuregates featuregate.MutableFeatureGate) error {
	return featuregates.Add(cloudPublicFeatureGates)
}

// cloudPublicFeatureGates consists of cloud-specific feature keys.
// To add a new feature, define a key for it at k8s.io/api/pkg/features and add it here.
var cloudPublicFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	IPv6DualStack:                    {Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	ControllerManagerLeaderMigration: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26
}
