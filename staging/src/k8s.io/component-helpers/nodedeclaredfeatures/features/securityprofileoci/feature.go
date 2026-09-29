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

package securityprofileoci

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// Ensure the feature struct implements the unified Feature interface.
var _ types.Feature = &securityProfileOCIFeature{}

const (
	// SecurityProfileOCIFeatureGate is the feature gate name.
	SecurityProfileOCIFeatureGate = "SecurityProfileOCI"
	// SecurityProfileOCI is the declared feature name.
	SecurityProfileOCI = "SecurityProfileOCI"
)

// Feature is the implementation of the `SecurityProfileOCI` feature.
var Feature = &securityProfileOCIFeature{}

type securityProfileOCIFeature struct{}

func (f *securityProfileOCIFeature) Name() string {
	return SecurityProfileOCI
}

// Requirements depends on the feature gate only. Declared features must be
// derivable from static node configuration, so container runtime support is
// checked separately by the kubelet.
func (f *securityProfileOCIFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{SecurityProfileOCIFeatureGate},
	}
}

func (f *securityProfileOCIFeature) Discover(cfg *types.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(SecurityProfileOCIFeatureGate)
}

func (f *securityProfileOCIFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	return podSpecUsesSeccompProfileOCI(podInfo.Spec)
}

func (f *securityProfileOCIFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	// Seccomp profiles of existing containers are immutable, so only an update
	// that adds the first OCI profile introduces the requirement: adding an
	// ephemeral container with an OCI profile to a pod without one. The
	// kubelet checks such updates; the API server does not validate the
	// ephemeralcontainers subresource against declared features.
	return !podSpecUsesSeccompProfileOCI(oldPodInfo.Spec) && podSpecUsesSeccompProfileOCI(newPodInfo.Spec)
}

func (f *securityProfileOCIFeature) MaxVersion() *version.Version {
	return nil
}

func podSpecUsesSeccompProfileOCI(spec *v1.PodSpec) bool {
	if spec == nil {
		return false
	}
	if spec.SecurityContext != nil && isOCI(spec.SecurityContext.SeccompProfile) {
		return true
	}
	for i := range spec.InitContainers {
		if containerUsesOCI(spec.InitContainers[i].SecurityContext) {
			return true
		}
	}
	for i := range spec.Containers {
		if containerUsesOCI(spec.Containers[i].SecurityContext) {
			return true
		}
	}
	for i := range spec.EphemeralContainers {
		if containerUsesOCI(spec.EphemeralContainers[i].SecurityContext) {
			return true
		}
	}
	return false
}

func containerUsesOCI(sc *v1.SecurityContext) bool {
	return sc != nil && isOCI(sc.SeccompProfile)
}

func isOCI(profile *v1.SeccompProfile) bool {
	return profile != nil && profile.Type == v1.SeccompProfileTypeOCI
}
