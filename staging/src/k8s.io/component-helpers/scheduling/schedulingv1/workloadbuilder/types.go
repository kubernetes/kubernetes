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
)

// SchedulingConfig is the hierarchy-agnostic intermediate representation of a
// group's scheduling configuration, decoupled from the leaf/composite API
// split. Controllers usually populate it via MapPodGroupConfig.
type SchedulingConfig struct {
	// Policy selects the scheduling mode; nil resolves to Basic.
	Policy         *SchedulingPolicy
	Constraints    *SchedulingConstraints
	DisruptionMode *DisruptionMode

	// ResourceClaims lists dynamic resource claims shared across the group.
	ResourceClaims []ResourceClaim

	// PriorityClassName is copied onto the compiled PodGroupTemplate so a
	// PodGroup materialized from it inherits the group's priority.
	PriorityClassName string
}

// SchedulingPolicy selects the scheduling mode. Exactly one field must be set.
type SchedulingPolicy struct {
	Basic *BasicSchedulingPolicy
	Gang  *GangSchedulingPolicy
}

// BasicSchedulingPolicy indicates standard Kubernetes pod-by-pod scheduling.
type BasicSchedulingPolicy struct{}

// GangSchedulingPolicy defines all-or-nothing scheduling parameters.
type GangSchedulingPolicy struct {
	// MinCount may be nil for a callback to default (e.g. to a Job's
	// parallelism); it must be positive before Build compiles the template.
	MinCount *int32
}

// SchedulingConstraints holds placement constraints such as topology.
type SchedulingConstraints struct {
	Topology []schedulingv1alpha3.TopologyConstraint
}

// DisruptionMode selects how pods are disrupted. Exactly one field must be set.
type DisruptionMode struct {
	Single *SingleDisruptionMode
	All    *AllDisruptionMode
}

// SingleDisruptionMode indicates pods can be disrupted independently.
type SingleDisruptionMode struct{}

// AllDisruptionMode indicates all pods must be disrupted together.
type AllDisruptionMode struct{}

// ResourceClaim references a shared dynamic resource claim.
// Exactly one of ResourceClaimName or ResourceClaimTemplateName must be set.
type ResourceClaim struct {
	Name                      string
	ResourceClaimName         *string
	ResourceClaimTemplateName *string
}

// SchedulingConfigFunc post-processes the merged SchedulingConfig after the
// default/user merge.
type SchedulingConfigFunc func(*SchedulingConfig)

// WorkloadItem is a node in a controller's logical workload.
type WorkloadItem struct {
	// Name identifies this component. It becomes the PodGroupTemplate name and
	// must be non-empty.
	Name string

	// DefaultConfig is the controller's default config for this node, used for
	// any field the user left unset.
	DefaultConfig *SchedulingConfig

	// Input holds the user's intent for this node, including original API objects
	// and their associated field paths for precise declarative validation and error reporting.
	// Nil fields fall back to DefaultConfig.
	Input WorkloadInput

	// Callbacks run in order against the resolved config after the
	// default/user merge.
	Callbacks []SchedulingConfigFunc
}

// WorkloadInput bundles the leaf-level building blocks a controller embeds
// in its own API, along with their field paths.
type WorkloadInput struct {
	Policy         PolicyInput
	Constraints    ConstraintsInput
	DisruptionMode DisruptionModeInput
	ResourceClaims ResourceClaimsInput
}

// PolicyInput wraps the scheduling policy with its field path.
type PolicyInput struct {
	PodGroupData *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy
	
	// PathElements specifies the relative path from the WorkloadItem's rootPath
	// to where this building block is embedded in the controller's API.
	// For example, if rootPath is `job.spec.scheduling` and PathElements is
	// `[]string{"policy"}`, validation errors will be reported at
	// `job.spec.scheduling.policy`.
	// If left empty, errors are reported directly at the rootPath.
	PathElements []string
}

// ConstraintsInput wraps the topology constraints with its field path.
type ConstraintsInput struct {
	PodGroupData *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints
	PathElements []string
}

// DisruptionModeInput wraps the disruption mode with its field path.
type DisruptionModeInput struct {
	PodGroupData *schedulingv1alpha3.WorkloadPodGroupDisruptionMode
	PathElements []string
}

// ResourceClaimsInput wraps the resource claims with their field path.
type ResourceClaimsInput struct {
	PodGroupData []schedulingv1alpha3.WorkloadPodGroupResourceClaim
	PathElements []string
}

// DeepCopy returns a deep copy of c, or nil if c is nil. Build copies configs
// before merging so callbacks mutating the resolved config cannot leak back into
// a controller's shared DefaultConfig or the user's input.
func (c *SchedulingConfig) DeepCopy() *SchedulingConfig {
	if c == nil {
		return nil
	}
	out := &SchedulingConfig{PriorityClassName: c.PriorityClassName}
	if c.Policy != nil {
		p := &SchedulingPolicy{}
		if c.Policy.Basic != nil {
			p.Basic = &BasicSchedulingPolicy{}
		}
		if c.Policy.Gang != nil {
			p.Gang = &GangSchedulingPolicy{}
			if c.Policy.Gang.MinCount != nil {
				mc := *c.Policy.Gang.MinCount
				p.Gang.MinCount = &mc
			}
		}
		out.Policy = p
	}
	if c.Constraints != nil {
		cc := &SchedulingConstraints{}
		if len(c.Constraints.Topology) > 0 {
			cc.Topology = make([]schedulingv1alpha3.TopologyConstraint, len(c.Constraints.Topology))
			copy(cc.Topology, c.Constraints.Topology)
		}
		out.Constraints = cc
	}
	if c.DisruptionMode != nil {
		dm := &DisruptionMode{}
		if c.DisruptionMode.Single != nil {
			dm.Single = &SingleDisruptionMode{}
		}
		if c.DisruptionMode.All != nil {
			dm.All = &AllDisruptionMode{}
		}
		out.DisruptionMode = dm
	}
	if len(c.ResourceClaims) > 0 {
		out.ResourceClaims = make([]ResourceClaim, len(c.ResourceClaims))
		for i := range c.ResourceClaims {
			rc := ResourceClaim{Name: c.ResourceClaims[i].Name}
			if c.ResourceClaims[i].ResourceClaimName != nil {
				v := *c.ResourceClaims[i].ResourceClaimName
				rc.ResourceClaimName = &v
			}
			if c.ResourceClaims[i].ResourceClaimTemplateName != nil {
				v := *c.ResourceClaims[i].ResourceClaimTemplateName
				rc.ResourceClaimTemplateName = &v
			}
			out.ResourceClaims[i] = rc
		}
	}
	return out
}
