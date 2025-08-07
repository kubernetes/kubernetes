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

// PodRequirements represents the set of capabilities a pod requires.
type PodRequirements struct {
	Capabilities []PodRequirement
}

// PodRequirement is a key-value pair representing a single capability requirement.
type PodRequirement struct {
	Key   string
	Value string
}

// CreateInferrer is an interface for inferring capability requirements from a new pod.
type CreateInferrer interface {
	Infer(ctx context.Context, pod *v1.Pod) *PodRequirement
}

// UpdateInferrer is an interface for inferring capability requirements from a pod update.
type UpdateInferrer interface {
	Infer(ctx context.Context, oldPod, newPod *v1.Pod) *PodRequirement
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
func (h *NodeCapabilityHelper) InferPodCreateRequirements(ctx context.Context, pod *v1.Pod) (*PodRequirements, error) {
	logger := klog.FromContext(ctx)
	reqs := &PodRequirements{}
	h.registry.ForEach(func(name string, cap Capability) {
		if cap.CreateInferrer != nil {
			if req := cap.CreateInferrer.Infer(ctx, pod); req != nil {
				reqs.Capabilities = append(reqs.Capabilities, *req)
				logger.V(4).Info("Inferred capability requirement for pod create", "capability", req.Key, "pod", klog.KObj(pod))
			}
		}
	})
	return reqs, nil
}

// InferPodUpdateRequirements inspects the change between an old and new pod spec
// and returns the set of capabilities required to validate the update operation.
func (h *NodeCapabilityHelper) InferPodUpdateRequirements(ctx context.Context, oldPod, newPod *v1.Pod) (*PodRequirements, error) {
	logger := klog.FromContext(ctx)
	reqs := &PodRequirements{}
	h.registry.ForEach(func(name string, cap Capability) {
		if cap.UpdateInferrer != nil {
			if req := cap.UpdateInferrer.Infer(ctx, oldPod, newPod); req != nil {
				reqs.Capabilities = append(reqs.Capabilities, *req)
				logger.V(4).Info("Inferred capability requirement for pod update", "capability", req.Key, "pod", klog.KObj(newPod))
			}
		}
	})
	return reqs, nil
}

// MatchNode checks if a node's advertised capabilities satisfy the pre-computed requirements for a pod.
func (h *NodeCapabilityHelper) MatchNode(ctx context.Context, reqs *PodRequirements, node *v1.Node) (bool, error) {
	if node == nil {
		return false, fmt.Errorf("node cannot be nil")
	}
	return h.match(ctx, reqs, node.Status.Capabilities)
}

// MatchCurrentNode checks if the current node's capabilities satisfy the pre-computed requirements for a pod.
func (h *NodeCapabilityHelper) MatchCurrentNode(ctx context.Context, reqs *PodRequirements, nodeCapabilities map[string]string) (bool, error) {
	return h.match(ctx, reqs, nodeCapabilities)
}

func (h *NodeCapabilityHelper) match(ctx context.Context, reqs *PodRequirements, nodeCapabilities map[string]string) (bool, error) {
	logger := klog.FromContext(ctx)
	if reqs == nil {
		return true, nil // No requirements to match.
	}

	for _, req := range reqs.Capabilities {
		if h.shouldCheckCapability(ctx, req.Key) {
			val, ok := nodeCapabilities[req.Key]
			if !ok {
				// A required capability is missing.
				logger.V(4).Info("Node does not have required capability", "capability", req.Key, "nodeCapabilities", nodeCapabilities)
				return false, nil
			}
			if val != req.Value {
				// A required capability does not match.
				logger.V(4).Info("Node has mismatched capability", "capability", req.Key, "expected", req.Value, "actual", val)
				return false, nil
			}
		}
	}

	return true, nil
}

func (h *NodeCapabilityHelper) shouldCheckCapability(ctx context.Context, capabilityName string) bool {
	logger := klog.FromContext(ctx)
	capability, ok := h.registry.Get(capabilityName)
	if !ok {
		logger.Info("Skipping capability reporting", "capabilityName", capabilityName, "reason", "capability not registered")
		return false
	}

	for _, fd := range capability.FeatureDependency {
		if h.IsFeatureGateRelevant(ctx, fd, h.componentVersion) {
			return true
		}
	}

	logger.V(4).Info("Skipping capability checking", "capabilityName", capabilityName, "reason", "all feature gates are past GA + version skew")
	return false
}

func (h *NodeCapabilityHelper) IsFeatureGateRelevant(ctx context.Context, fd FeatureDependency, componentVersion *version.Version) bool {
	logger := klog.FromContext(ctx)
	if !fd.IsGA {
		logger.V(4).Info("Featuregate is relevant", "feature gate", fd.FeatureGate, "reason", "feature is not GA")
		return true
	}
	if fd.IsDeprecated {
		logger.V(4).Info("Featuregate is not relevant", "feature gate", fd.FeatureGate, "reason", "feature is deprecated")
		return false
	}
	if componentVersion.LessThan(fd.GAVersion.AddMinor(SupportedVersionSkew)) {
		logger.V(4).Info("Featuregate is relevant", "feature gate", fd.FeatureGate, "reason", "component version is within supported skew")
		return true
	}
	logger.V(4).Info("Featuregate is not relevant", "feature gate", fd.FeatureGate, "reason", "component version is past supported skew")
	return false
}
