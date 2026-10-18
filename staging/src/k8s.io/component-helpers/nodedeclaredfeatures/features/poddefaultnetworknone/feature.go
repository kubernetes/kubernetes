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

package poddefaultnetworknone

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

var _ types.Feature = &podDefaultNetworkNoneFeature{}

const (
	// PodDefaultNetworkFeatureGate is the feature gate that enables spec.defaultNetwork.
	PodDefaultNetworkFeatureGate = "PodDefaultNetwork"
	// PodDefaultNetworkNone is the declared feature name.
	PodDefaultNetworkNone = "PodDefaultNetworkNone"
)

// Feature is declared by nodes whose kubelet and container runtime can run
// pods with spec.defaultNetwork "None" (KEP-6313). Pods that request it are
// kept away from other nodes by the scheduler and rejected by the kubelet
// admission handler, so an isolated pod is never attached to the default pod
// network by a node that does not understand the request.
var Feature = &podDefaultNetworkNoneFeature{}

type podDefaultNetworkNoneFeature struct{}

func (f *podDefaultNetworkNoneFeature) Name() string {
	return PodDefaultNetworkNone
}

func (f *podDefaultNetworkNoneFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{PodDefaultNetworkFeatureGate},
		RequiredRuntimeFeatures: &types.RuntimeFeatures{
			DefaultNetworkNone: true,
		},
	}
}

func (f *podDefaultNetworkNoneFeature) Discover(cfg *types.NodeConfiguration) bool {
	if !cfg.FeatureGates.Enabled(PodDefaultNetworkFeatureGate) {
		return false
	}
	return cfg.RuntimeFeatures.DefaultNetworkNone
}

func (f *podDefaultNetworkNoneFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	return podInfo.Spec.DefaultNetwork != nil && *podInfo.Spec.DefaultNetwork == v1.PodDefaultNetworkNone
}

func (f *podDefaultNetworkNoneFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	// defaultNetwork is immutable.
	return false
}

func (f *podDefaultNetworkNoneFeature) MaxVersion() *version.Version {
	return nil
}
