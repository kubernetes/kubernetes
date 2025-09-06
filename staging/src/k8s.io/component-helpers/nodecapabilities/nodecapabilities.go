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

package nodecapabilities

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog/v2"
)

const (
	// SupportedVersionSkew is the supported version skew between control plane components and the Kubelet.
	// Set based on https://kubernetes.io/releases/version-skew-policy/
	SupportedVersionSkew = 3
)

// CapabilityRequirement is a key-value pair representing a single capability a pod requires from a node.
type CapabilityRequirement struct {
	Key   string
	Value string
}

// CreateInferrer is an interface for inferring capability requirements from a new pod.
type CreateInferrer interface {
	Infer(ctx context.Context, pod *v1.Pod) *CapabilityRequirement
}

// UpdateInferrer is an interface for inferring capability requirements from a pod update.
type UpdateInferrer interface {
	Infer(ctx context.Context, oldPod, newPod *v1.Pod) *CapabilityRequirement
}

// FeatureDependency defines the metadata for a single feature that a capability depends on.
type FeatureDependency struct {
	// FeatureGate is the name of the feature gate that controls this feature.
	FeatureGate string
	// IsGA indicates if the feature is generally available.
	IsGA bool
	// GAVersion indicates the version at which the feature became generally available. Nil if the feature is not GA yet.
	GAVersion *version.Version
	// IsDeprecated indicates if the feature is deprecated.
	IsDeprecated bool
}

// Capability defines the metadata for a single node capability.
type Capability struct {
	// Name is the unique identifier for the capability.
	Name string
	// FeatureDependency lists the features that this capability depends on.
	FeatureDependency []FeatureDependency
	CreateInferrer    CreateInferrer
	UpdateInferrer    UpdateInferrer
}

// Registry is an interface for a collection of capabilities.
type Registry interface {
	Get(name string) (Capability, bool)
	ForEach(func(name string, cap Capability))
}

// FeatureGate is an interface that allows checking if a feature is enabled.
type FeatureGate interface {
	Enabled(feature string) bool
}

// NodeCapabilityHelper provides functions for inferring and matching pod capability requirements.
type NodeCapabilityHelper struct {
	registry         Registry
	componentVersion *version.Version
}

// NewNodeCapabilityHelper creates a new instance of the capability helper.
func NewNodeCapabilityHelper(registry Registry, componentVersion *version.Version) (*NodeCapabilityHelper, error) {
	if registry == nil {
		return nil, fmt.Errorf("registry must not be nil")
	}
	if componentVersion == nil {
		return nil, fmt.Errorf("componentVersion must not be nil")
	}
	return &NodeCapabilityHelper{
		registry:         registry,
		componentVersion: componentVersion,
	}, nil
}

// InferPodCreateRequirements inspects a new pod and returns the set of capabilities
// required for its initial scheduling.
func (h *NodeCapabilityHelper) InferPodCreateRequirements(ctx context.Context, pod *v1.Pod) ([]CapabilityRequirement, error) {
	logger := klog.FromContext(ctx)
	var reqs []CapabilityRequirement
	h.registry.ForEach(func(name string, cap Capability) {
		if cap.CreateInferrer != nil {
			if req := cap.CreateInferrer.Infer(ctx, pod); req != nil {
				reqs = append(reqs, *req)
				logger.V(4).Info("Inferred capability requirement for pod create", "capability", req.Key, "pod", klog.KObj(pod))
			}
		}
	})
	return reqs, nil
}

// InferPodUpdateRequirements inspects the change between an old and new pod spec
// and returns the set of capabilities required to validate the update operation.
func (h *NodeCapabilityHelper) InferPodUpdateRequirements(ctx context.Context, oldPod, newPod *v1.Pod) ([]CapabilityRequirement, error) {
	logger := klog.FromContext(ctx)
	var reqs []CapabilityRequirement
	h.registry.ForEach(func(name string, cap Capability) {
		if cap.UpdateInferrer != nil {
			if req := cap.UpdateInferrer.Infer(ctx, oldPod, newPod); req != nil {
				reqs = append(reqs, *req)
				logger.V(4).Info("Inferred capability requirement for pod update", "capability", req.Key, "pod", klog.KObj(newPod))
			}
		}
	})
	return reqs, nil
}

// MatchResult encapsulates the result of a capability match check.
type MatchResult struct {
	// IsMatch is true if the node satisfies all capability requirements.
	IsMatch bool
	// UnsatisfiedRequirements lists the specific capabilities that were not met.
	// This field is only populated if IsMatch is false.
	UnsatisfiedRequirements []CapabilityRequirement
}

// MatchNode checks if a node's advertised capabilities satisfy the pre-computed requirements.
// It returns a MatchResult object summarizing the outcome.
// It only returns a non-nil error for unexpected problems (e.g., a nil node).
func (h *NodeCapabilityHelper) MatchNode(ctx context.Context, reqs []CapabilityRequirement, node *v1.Node) (*MatchResult, error) {
	if node == nil {
		return nil, fmt.Errorf("node cannot be nil")
	}
	return h.match(ctx, reqs, node.Status.Capabilities)
}

// MatchCurrentNode checks if the current node's capabilities satisfy the pre-computed requirements for a pod.
// It returns a MatchResult object summarizing the outcome.
// It only returns a non-nil error for unexpected problems.
func (h *NodeCapabilityHelper) MatchCurrentNode(ctx context.Context, reqs []CapabilityRequirement, nodeCapabilities map[string]string) (*MatchResult, error) {
	return h.match(ctx, reqs, nodeCapabilities)
}

func (h *NodeCapabilityHelper) match(ctx context.Context, reqs []CapabilityRequirement, nodeCapabilities map[string]string) (*MatchResult, error) {
	logger := klog.FromContext(ctx)
	if reqs == nil {
		return &MatchResult{IsMatch: true}, nil // No requirements to match.
	}

	var mismatched []CapabilityRequirement

	for _, req := range reqs {

		val, ok := nodeCapabilities[req.Key]
		if !ok {
			// A required capability is missing.
			logger.V(4).Info("Node does not have required capability", "capability", req.Key, "nodeCapabilities", nodeCapabilities)
			mismatched = append(mismatched, req)
			continue
		}
		if val != req.Value {
			// A required capability does not match.
			logger.V(4).Info("Node has mismatched capability", "capability", req.Key, "expected", req.Value, "actual", val)
			mismatched = append(mismatched, req)
		}

	}

	if len(mismatched) > 0 {
		return &MatchResult{IsMatch: false, UnsatisfiedRequirements: mismatched}, nil
	}

	return &MatchResult{IsMatch: true}, nil
}
