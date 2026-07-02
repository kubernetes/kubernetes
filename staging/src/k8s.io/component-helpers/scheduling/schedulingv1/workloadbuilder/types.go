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
	"k8s.io/apimachinery/pkg/util/validation/field"
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

// FieldPaths declares where this item's scheduling fields live in the
// controller's API, so validation errors point at real, controller-relative
// paths instead of the library's internal IR structure. Each path is the root
// of its field; the builder derives deeper paths (e.g. per-claim indices) from
// it. Any nil path falls back to a relative path, so the mapping is optional.
// Controllers set this during tree construction, and hierarchical controllers
// can give each WorkloadItem its own paths.
type FieldPaths struct {
	Name           *field.Path
	Policy         *field.Path
	GangMinCount   *field.Path
	Constraints    *field.Path
	DisruptionMode *field.Path
	ResourceClaims *field.Path
}

// WorkloadItemFunc post-processes an item's ResolvedConfig after the
// default/user merge, typically for controller defaulting (e.g. gang MinCount).
type WorkloadItemFunc func(*WorkloadItem)

// WorkloadItem is a node in a controller's logical workload.
type WorkloadItem struct {
	// Name identifies this component. It becomes the PodGroupTemplate name and
	// must be non-empty.
	Name string

	// DefaultConfig is the controller's default config for this node, used for
	// any field the user left unset.
	DefaultConfig *SchedulingConfig

	// UserConfig is the user's intent for this node; nil fields fall back to
	// DefaultConfig field-by-field.
	UserConfig *SchedulingConfig

	// Callbacks run in order against ResolvedConfig after the default/user merge.
	Callbacks []WorkloadItemFunc

	// ResolvedConfig is the merged config, populated by Build before Callbacks
	// run. Callers do not set it.
	ResolvedConfig *SchedulingConfig

	// FieldPaths optionally declares where this item's fields live in the
	// controller's API, so Build and Validate can report errors against the
	// controller's own field paths. When nil, errors use relative paths.
	FieldPaths *FieldPaths

	// Children, when non-empty, mark this node as a structural group.
	Children []*WorkloadItem
}

// DeepCopy returns a deep copy of c, or nil if c is nil. Build copies configs
// before merging so callbacks mutating ResolvedConfig cannot leak back into a
// controller's shared DefaultConfig or the user's input.
func (c *SchedulingConfig) DeepCopy() *SchedulingConfig {
	if c == nil {
		return nil
	}
	out := &SchedulingConfig{}
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
