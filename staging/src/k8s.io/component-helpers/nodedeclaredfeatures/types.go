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

//go:generate mockery
package nodedeclaredfeatures

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
)

// PodInfo is an extensible data structure that wraps the pod object
// and can be expanded in the future to include ancillary resources
// like ResourceClaims or PVCs.
type PodInfo struct {
	// Spec is the Pod's specification.
	Spec *v1.PodSpec
	// Status is the Pod's current status. This field can be nil, for example,
	// when this struct represents a pod not yet created.
	// (e.g., during scheduling).
	Status *v1.PodStatus
	// Add other ancillary resources here in the future as needed.
	// Example: ResourceClaims []*v1.ResourceClaim
}

// Feature encapsulates all logic for a given declared feature.
type Feature interface {
	// Name returns the feature's well-known name.
	Name() string

	// Discover checks if a node provides the feature based on its configuration.
	Discover(cfg *NodeConfiguration) bool

	// InferForScheduling checks if pod scheduling requires the feature.
	InferForScheduling(podInfo *PodInfo) bool

	// InferForUpdate checks if a pod update requires the feature.
	InferForUpdate(oldPodInfo, newPodInfo *PodInfo) bool

	// MaxVersion specifies the upper bound Kubernetes version (inclusive) for this feature's relevance
	// as a scheduling factor. Should be set based on the feature's GA version
	// and the cluster's version skew policy. Nil means no upper version bound.
	// Comparisons use the full semantic versioning scheme.
	MaxVersion() *version.Version
}

// FeatureGate is an interface that abstracts feature gate checking.
type FeatureGate interface {
	// Enabled returns true if the named feature gate is enabled.
	Enabled(key string) bool
}

// StaticConfiguration provides a view of a node's static configuration.
type StaticConfiguration struct {
	// Kubelet's CPU Manager policy
	CPUManagerPolicy string
}

// NodeConfiguration provides a generic view of a node's static configuration.
type NodeConfiguration struct {
	// FeatureGates holds an implementation of the FeatureGate interface.
	FeatureGates FeatureGate
	// StaticConfig holds node static configuration.
	StaticConfig StaticConfiguration
	// Version holds the current node version. This is used for full semantic version comparisons
	// with Feature.MaxVersion() to determine if a feature needs to be reported.
	Version *version.Version
}
