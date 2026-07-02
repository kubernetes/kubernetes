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

package workloadbuilder

import (
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
)

// SchedulingConfig is the hierarchy-agnostic intermediate representation of a
// group's scheduling configuration, decoupled from the leaf/composite API
// split.
//
// +k8s:deepcopy-gen=true
type SchedulingConfig struct {
	// Policy selects the scheduling mode; nil resolves to Basic.
	Policy *SchedulingPolicy

	// Constraints holds placement constraints such as topology; nil means the
	// group has no placement constraints.
	Constraints *SchedulingConstraints

	// DisruptionMode selects how the group's pods are disrupted; nil leaves the
	// mode unset so the scheduler default applies.
	DisruptionMode *DisruptionMode

	// ResourceClaims lists dynamic resource claims shared across the group.
	ResourceClaims []ResourceClaim

	// PriorityClassName is copied onto the compiled PodGroupTemplate so a
	// PodGroup materialized from it inherits the group's priority.
	PriorityClassName string
}

// SchedulingPolicy selects the scheduling mode. Exactly one field must be set.
//
// +k8s:deepcopy-gen=true
type SchedulingPolicy struct {
	// Basic selects standard Kubernetes pod-by-pod scheduling.
	Basic *BasicSchedulingPolicy

	// Gang selects all-or-nothing scheduling.
	Gang *GangSchedulingPolicy
}

// BasicSchedulingPolicy indicates standard Kubernetes pod-by-pod scheduling.
//
// +k8s:deepcopy-gen=true
type BasicSchedulingPolicy struct{}

// GangSchedulingPolicy defines all-or-nothing scheduling parameters.
//
// +k8s:deepcopy-gen=true
type GangSchedulingPolicy struct {
	// MinCount may be nil for a callback to default (e.g. to a Job's
	// parallelism); it must be positive before Build compiles the template.
	MinCount *int32
}

// SchedulingConstraints holds placement constraints such as topology.
//
// +k8s:deepcopy-gen=true
type SchedulingConstraints struct {
	// Topology lists the topology constraints applied to the group's pods.
	Topology []schedulingv1beta1.TopologyConstraint
}

// DisruptionMode selects how pods are disrupted. Exactly one field must be set.
//
// +k8s:deepcopy-gen=true
type DisruptionMode struct {
	// Single indicates pods can be disrupted independently.
	Single *SingleDisruptionMode

	// All indicates all pods must be disrupted together.
	All *AllDisruptionMode
}

// SingleDisruptionMode indicates pods can be disrupted independently.
//
// +k8s:deepcopy-gen=true
type SingleDisruptionMode struct{}

// AllDisruptionMode indicates all pods must be disrupted together.
//
// +k8s:deepcopy-gen=true
type AllDisruptionMode struct{}

// ResourceClaim references a shared dynamic resource claim.
// Exactly one of ResourceClaimName or ResourceClaimTemplateName must be set.
//
// +k8s:deepcopy-gen=true
type ResourceClaim struct {
	// Name uniquely identifies this claim within the group.
	Name string

	// ResourceClaimName references an existing ResourceClaim by name.
	ResourceClaimName *string

	// ResourceClaimTemplateName references a ResourceClaimTemplate by name from
	// which a per-group ResourceClaim is generated.
	ResourceClaimTemplateName *string
}

// SchedulingConfigFunc post-processes the merged SchedulingConfig after the
// default/user merge.
type SchedulingConfigFunc func(*SchedulingConfig)

// WorkloadItem is a node in a controller's logical workload.
//
// When a WorkloadItem is passed as ValidationInput.OldRoot, only Name and Input
// are consulted. DefaultConfig, Callbacks, and Input.*.PathElements are ignored
// because the resolved config and error paths come from the new root.
type WorkloadItem struct {
	// Name identifies this component. It becomes the PodGroupTemplate name and
	// must be non-empty.
	Name string

	// DefaultConfig is the controller's default config for this node, used for
	// any field the user left unset.
	DefaultConfig *SchedulingConfig

	// Input holds the user's intent for this node as the original versioned API
	// objects and their associated field paths for precise declarative
	// validation and error reporting. The nil fields fall back to DefaultConfig.
	Input WorkloadInput

	// Callbacks run in order against the resolved config after the
	// default/user merge.
	Callbacks []SchedulingConfigFunc
}

// WorkloadInput bundles the leaf-level building blocks a controller embeds
// in its own API. The zero value means "nothing set", so callers only
// populate the blocks they care about.
type WorkloadInput struct {
	// Policy is the scheduling policy building block and its field path.
	Policy PolicyInput

	// Constraints is the scheduling constraints building block and its field path.
	Constraints ConstraintsInput

	// DisruptionMode is the disruption mode building block and its field path.
	DisruptionMode DisruptionModeInput

	// ResourceClaims is the resource claims building block and its field path.
	ResourceClaims ResourceClaimsInput
}

// PolicyInput wraps the scheduling policy building block with its field path.
type PolicyInput struct {
	// PodGroupData is the Workload scheduling policy info for the PodGroup.
	PodGroupData *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy

	// PathElements is the path, relative to the WorkloadItem's rootPath, at
	// which this building block is embedded in the controller's API. For a
	// rootPath of `spec.scheduling` and PathElements of []string{"schedulingPolicy"},
	// validation errors are reported at `spec.scheduling.schedulingPolicy`. When
	// empty, errors are reported directly at the rootPath.
	PathElements []string
}

// ConstraintsInput wraps the scheduling constraints building block with its field path.
type ConstraintsInput struct {
	// PodGroupData is the Workload scheduling constraints info for the PodGroup.
	PodGroupData *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints

	// PathElements is the path, relative to the WorkloadItem's rootPath, at which
	// this building block is embedded in the controller's API; see PolicyInput.PathElements.
	PathElements []string
}

// DisruptionModeInput wraps the disruption mode building block with its field path.
type DisruptionModeInput struct {
	// PodGroupData is the Workload disruption mode info for the PodGroup.
	PodGroupData *schedulingv1alpha3.WorkloadPodGroupDisruptionMode

	// PathElements is the path, relative to the WorkloadItem's rootPath, at which
	// this building block is embedded in the controller's API; see PolicyInput.PathElements.
	PathElements []string
}

// ResourceClaimsInput wraps the resource claims building block with its field path.
type ResourceClaimsInput struct {
	// PodGroupData is the list of Workload resource claims for the PodGroup.
	PodGroupData []schedulingv1alpha3.WorkloadPodGroupResourceClaim

	// PathElements is the path, relative to the WorkloadItem's rootPath, at which
	// this building block is embedded in the controller's API; see PolicyInput.PathElements.
	PathElements []string
}
