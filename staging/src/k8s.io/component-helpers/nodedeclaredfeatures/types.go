/*
Copyright 2025 The Kubernetes Authors.

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

package nodedeclaredfeatures

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
)

// PodInfo is an extensible data structure that wraps the pod object
// and can be expanded in the future to include ancillary resources
// like ResourceClaims or PVCs.
type PodInfo struct {
	Pod *v1.Pod
	// Add other ancillary resources here in the future as needed.
	// Example: ResourceClaims []*v1.ResourceClaim
}

// Feature encapsulates all logic for a given declared feature.
type Feature interface {
	// Name returns the feature's well-known name.
	Name() string

	// Discover checks if a node provides the feature based on its configuration.
	Discover(cfg *NodeConfiguration) (bool, error)

	// InferFromCreate checks if a new pod requires the feature.
	InferFromCreate(podInfo *PodInfo) bool

	// InferFromUpdate checks if a pod update requires the feature.
	InferFromUpdate(oldPodInfo, newPodInfo *PodInfo) bool

	// MinVersion is the minimum Kubernetes version where this feature is a scheduling constraint.
	MinVersion() *version.Version

	// MaxVersion is the maximum Kubernetes version where this feature is a scheduling constraint.
	MaxVersion() *version.Version
}

// NodeConfiguration provides a generic view of a node's static configuration.
type NodeConfiguration struct {
	// FeatureGates is a map of enabled feature gates.
	FeatureGates map[string]bool
	// KubeletConfig is a map of other Kubelet configuration values.
	KubeletConfig map[string]string
}

// IncompatibleFeatureError is returned when a pod requires a feature
// that is not available in the target Kubernetes version.
type IncompatibleFeatureError struct {
	FeatureName        string
	TargetVersion      *version.Version
	RequiredMinVersion *version.Version
}

func (e *IncompatibleFeatureError) Error() string {
	return fmt.Sprintf("feature %q is not available in version %s, requires at least %s", e.FeatureName, e.TargetVersion, e.RequiredMinVersion)
}
