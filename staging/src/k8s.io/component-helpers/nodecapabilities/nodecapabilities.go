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
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	featuregateutil "k8s.io/component-base/featuregate"
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
	Infer(*v1.Pod) *PodRequirement
}

// UpdateInferrer is an interface for inferring capability requirements from a pod update.
type UpdateInferrer interface {
	Infer(oldPod, newPod *v1.Pod) *PodRequirement
}

// Capability defines the metadata for a single node capability.
type Capability struct {
	FeatureGates   []featuregateutil.Feature
	CreateInferrer CreateInferrer
	UpdateInferrer UpdateInferrer
}

// Registry is an interface for a collection of capabilities.
type Registry interface {
	Get(name string) (Capability, bool)
	ForEach(func(name string, cap Capability))
}

// NodeCapabilityHelper provides functions for inferring and matching pod capability requirements.
type NodeCapabilityHelper struct {
	registry         Registry
	componentVersion *version.Version
	featureGates     featuregateutil.FeatureGate
	versionSpecs     map[featuregateutil.Feature]featuregateutil.VersionedSpecs
}

// NewNodeCapabilityHelper creates a new instance of the capability helper.
func NewNodeCapabilityHelper(registry Registry, componentVersion *version.Version, featureGates featuregateutil.FeatureGate) (*NodeCapabilityHelper, error) {
	if registry == nil {
		return nil, fmt.Errorf("registry must not be nil")
	}
	if componentVersion == nil {
		return nil, fmt.Errorf("componentVersion must not be nil")
	}
	if featureGates == nil {
		return nil, fmt.Errorf("featureGates must not be nil")
	}
	return &NodeCapabilityHelper{
		registry:         registry,
		componentVersion: componentVersion,
		featureGates:     featureGates,
		versionSpecs:     featureGates.DeepCopy().GetAllVersioned(),
	}, nil
}

// InferPodCreateRequirements inspects a new pod and returns the set of capabilities
// required for its initial scheduling.
func (h *NodeCapabilityHelper) InferPodCreateRequirements(pod *v1.Pod) (*PodRequirements, error) {
	reqs := &PodRequirements{}
	h.registry.ForEach(func(name string, cap Capability) {
		if cap.CreateInferrer != nil {
			if req := cap.CreateInferrer.Infer(pod); req != nil {
				reqs.Capabilities = append(reqs.Capabilities, *req)
				klog.V(4).InfoS("Inferred capability requirement for pod create", "capability", req.Key, "pod", klog.KObj(pod))
			}
		}
	})
	return reqs, nil
}

// InferPodUpdateRequirements inspects the change between an old and new pod spec
// and returns the set of capabilities required to validate the update operation.
func (h *NodeCapabilityHelper) InferPodUpdateRequirements(oldPod, newPod *v1.Pod) (*PodRequirements, error) {
	reqs := &PodRequirements{}
	h.registry.ForEach(func(name string, cap Capability) {
		if cap.UpdateInferrer != nil {
			if req := cap.UpdateInferrer.Infer(oldPod, newPod); req != nil {
				reqs.Capabilities = append(reqs.Capabilities, *req)
				klog.V(4).InfoS("Inferred capability requirement for pod update", "capability", req.Key, "pod", klog.KObj(newPod))
			}
		}
	})
	return reqs, nil
}

// MatchNode checks if a node's advertised capabilities satisfy the pre-computed requirements for a pod.
func (h *NodeCapabilityHelper) MatchNode(reqs *PodRequirements, node *v1.Node) (bool, error) {
	if node == nil {
		return false, fmt.Errorf("node cannot be nil")
	}
	return h.match(reqs, node.Status.Capabilities)
}

// MatchCurrentNode checks if the current node's capabilities satisfy the pre-computed requirements for a pod.
func (h *NodeCapabilityHelper) MatchCurrentNode(reqs *PodRequirements, nodeCapabilities map[string]string) (bool, error) {
	return h.match(reqs, nodeCapabilities)
}

func (h *NodeCapabilityHelper) match(reqs *PodRequirements, nodeCapabilities map[string]string) (bool, error) {
	if reqs == nil {
		return true, nil // No requirements to match.
	}

	for _, req := range reqs.Capabilities {
		if h.shouldCheckCapability(req.Key) {
			val, ok := nodeCapabilities[req.Key]
			if !ok {
				// A required capability is missing.
				klog.V(4).InfoS("Node does not have required capability", "capability", req.Key, "nodeCapabilities", nodeCapabilities)
				return false, nil
			}
			if val != req.Value {
				// A required capability does not match.
				klog.V(4).InfoS("Node has mismatched capability", "capability", req.Key, "expected", req.Value, "actual", val)
				return false, nil
			}
		}
	}

	return true, nil
}

func (h *NodeCapabilityHelper) shouldCheckCapability(capabilityName string) bool {
	capability, ok := h.registry.Get(capabilityName)
	if !ok {
		klog.InfoS("Skipping capability reporting", "capabilityName", capabilityName, "reason", "capability not registered")
		return false
	}

	for _, fg := range capability.FeatureGates {
		if h.IsFeatureGateRelevant(fg, h.componentVersion) {
			return true
		}
	}

	klog.V(4).InfoS("Skipping capability checking", "capabilityName", capabilityName, "reason", "all feature gates are past GA + version skew")
	return false
}

// ShouldReportCapability checks if a capability should be reported based on the feature gates and component version.
func (h *NodeCapabilityHelper) ShouldReportCapability(capabilityName string) bool {
	capability, ok := h.registry.Get(capabilityName)
	if !ok {
		klog.InfoS("Skipping capability reporting", "capabilityName", capabilityName, "reason", "capability not registered")
		return false
	}

	allGatesEnabled := true
	for _, fg := range capability.FeatureGates {
		if !h.featureGates.Enabled(fg) {
			allGatesEnabled = false
			klog.V(4).InfoS("Skipping capability reporting", "capabilityName", capabilityName, "reason", "feature gates not enabled", "featureGate", fg)
		}
	}
	if !allGatesEnabled {
		return false
	}

	for _, fg := range capability.FeatureGates {
		if h.IsFeatureGateRelevant(fg, h.componentVersion) {
			return true
		}
	}
	klog.V(4).InfoS("Skipping capability reporting", "capabilityName", capabilityName, "reason", "all feature gates are past GA + version skew")
	return false
}

func (h *NodeCapabilityHelper) IsFeatureGateRelevant(featuregate featuregateutil.Feature, componentVersion *version.Version) bool {
	featureGateSpecs, ok := h.versionSpecs[featuregate]
	if !ok {
		klog.InfoS("Cannot fetch version information for featuregate", "featuregate", featuregate)
		return false
	}
	for _, spec := range featureGateSpecs {
		if spec.Default && spec.PreRelease == featuregateutil.GA {
			if componentVersion.GreaterThan(spec.Version.AddMinor(SupportedVersionSkew)) {
				return false
			}
		}
	}
	return true
}
