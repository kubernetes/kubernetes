/*
Copyright The Kubernetes Authors.

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

package usernamespaceshostnetwork

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures"
)

// Ensure the feature struct implements the unified Feature interface.
var _ nodedeclaredfeatures.Feature = &userNamespacesHostNetworkFeature{}

const (
	// UserNamespacesHostNetworkSupportFeatureGate is the feature gate name.
	UserNamespacesHostNetworkSupportFeatureGate = "UserNamespacesHostNetworkSupport"
	// UserNamespacesHostNetworkSupport is the declared feature name.
	UserNamespacesHostNetworkSupport = "UserNamespacesHostNetworkSupport"
)

// Feature is the implementation of the `UserNamespacesHostNetworkSupport` feature.
var Feature = &userNamespacesHostNetworkFeature{}

type userNamespacesHostNetworkFeature struct{}

func (f *userNamespacesHostNetworkFeature) Name() string {
	return UserNamespacesHostNetworkSupport
}

func (f *userNamespacesHostNetworkFeature) Discover(cfg *nodedeclaredfeatures.NodeConfiguration) bool {
	// This feature requires both the feature gate to be enabled AND
	// runtime-level support for user namespaces with host network.
	if !cfg.FeatureGates.Enabled(UserNamespacesHostNetworkSupportFeatureGate) {
		return false
	}

	return cfg.RuntimeFeatures.UserNamespacesHostNetwork
}

func (f *userNamespacesHostNetworkFeature) InferForScheduling(podInfo *nodedeclaredfeatures.PodInfo) bool {
	// A pod needs this feature if it uses both host network AND user namespaces.
	if podInfo.Spec.HostNetwork && podInfo.Spec.HostUsers != nil && !*podInfo.Spec.HostUsers {
		return true
	}
	return false
}

func (f *userNamespacesHostNetworkFeature) InferForUpdate(oldPodInfo, newPodInfo *nodedeclaredfeatures.PodInfo) bool {
	// HostNetwork and HostUsers fields are immutable, so no update inference is needed.
	return false
}

func (f *userNamespacesHostNetworkFeature) MaxVersion() *version.Version {
	return nil
}
