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

// mapWorkloadInput translates the leaf-level scheduling.k8s.io building blocks
// into the IR for a WorkloadItem's config. It returns a non-nil config, leaving
// a field nil when its input is nil so resolveSchedulingConfig can fall back to
// defaults field-by-field.
func mapWorkloadInput(input WorkloadInput) *SchedulingConfig {
	cfg := &SchedulingConfig{}

	if input.Policy.PodGroupData != nil {
		if p := mapSchedulingPolicy(input.Policy.PodGroupData); p != nil {
			cfg.Policy = p
		}
	}
	if input.Constraints.PodGroupData != nil {
		if c := mapTopologyConstraints(input.Constraints.PodGroupData); c != nil {
			cfg.Constraints = c
		}
	}
	if input.DisruptionMode.PodGroupData != nil {
		if d := mapDisruptionMode(input.DisruptionMode.PodGroupData); d != nil {
			cfg.DisruptionMode = d
		}
	}
	if len(input.ResourceClaims.PodGroupData) > 0 {
		cfg.ResourceClaims = mapResourceClaims(input.ResourceClaims.PodGroupData)
	}

	return cfg
}

// mapCompositeGroupInput translates the composite-level scheduling.k8s.io
// building blocks into the IR for a composite WorkloadItem's config. It includes
// the group-of-groups scheduling policy, its topology constraints, disruption
//
//	mode, and preemption policy. The remaining CompositePodGroupTemplate fields
//
// (PriorityClassName and Priority) are populated from DefaultConfig/callbacks
// during resolution, not from WorkloadInput. It returns a non-nil config,
// leaving each field nil when its input is nil so resolveSchedulingConfig can
// fall back to the controller default field-by-field.
func mapCompositeGroupInput(input WorkloadInput) *SchedulingConfig {
	cfg := &SchedulingConfig{}

	if input.Policy.CompositePodGroupData != nil {
		if p := mapCompositeSchedulingPolicy(input.Policy.CompositePodGroupData); p != nil {
			cfg.Policy = p
		}
	}
	if input.Constraints.CompositePodGroupData != nil {
		if c := mapCompositeTopologyConstraints(input.Constraints.CompositePodGroupData); c != nil {
			cfg.Constraints = c
		}
	}
	if input.DisruptionMode.CompositePodGroupData != nil {
		if d := mapCompositeDisruptionMode(input.DisruptionMode.CompositePodGroupData); d != nil {
			cfg.DisruptionMode = d
		}
	}

	return cfg
}

// mapSchedulingPolicy returns nil for an empty policy so resolveSchedulingConfig
// falls back to the controller default instead of treating an unset user block
// as an explicit override.
func mapSchedulingPolicy(p *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy) *SchedulingPolicy {
	switch {
	case p.Basic != nil:
		return &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}
	case p.Gang != nil:
		gang := &GangSchedulingPolicy{}
		// Copy the pointer so a callback mutating the resolved config through it
		// cannot leak back into the caller's input building block.
		if p.Gang.MinCount != nil {
			mc := *p.Gang.MinCount
			gang.MinCount = &mc
		}
		return &SchedulingPolicy{Gang: gang}
	default:
		return nil
	}
}

// mapCompositeSchedulingPolicy maps the composite building-block policy onto the
// shared IR policy. The composite gang's minGroupCount is carried in the IR's
// GangSchedulingPolicy.MinCount and compiled back into
// CompositePodGroupSchedulingPolicy.MinGroupCount for composite nodes. An empty
// policy returns nil so resolveSchedulingConfig falls back to the controller
// default instead of treating an unset user block as an explicit override.
func mapCompositeSchedulingPolicy(p *schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy) *SchedulingPolicy {
	switch {
	case p.Basic != nil:
		return &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}
	case p.Gang != nil:
		gang := &GangSchedulingPolicy{}
		// Copy the pointer so a callback mutating the resolved config through it
		// cannot leak back into the caller's input building block.
		if p.Gang.MinGroupCount != nil {
			mgc := *p.Gang.MinGroupCount
			gang.MinCount = &mgc
		}
		return &SchedulingPolicy{Gang: gang}
	default:
		return nil
	}
}

func mapTopologyConstraints(c *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints) *SchedulingConstraints {
	if len(c.Topology) == 0 {
		return nil
	}
	topology := make([]schedulingv1beta1.TopologyConstraint, len(c.Topology))
	for i, t := range c.Topology {
		topology[i] = schedulingv1beta1.TopologyConstraint{Key: t.Key}
	}
	return &SchedulingConstraints{Topology: topology}
}

func mapCompositeTopologyConstraints(c *schedulingv1alpha3.WorkloadCompositePodGroupSchedulingConstraints) *SchedulingConstraints {
	if len(c.Topology) == 0 {
		return nil
	}
	topology := make([]schedulingv1beta1.TopologyConstraint, len(c.Topology))
	for i, t := range c.Topology {
		topology[i] = schedulingv1beta1.TopologyConstraint{Key: t.Key}
	}
	return &SchedulingConstraints{Topology: topology}
}

func mapDisruptionMode(d *schedulingv1alpha3.WorkloadPodGroupDisruptionMode) *DisruptionMode {
	switch {
	case d.Single != nil:
		return &DisruptionMode{Single: &SingleDisruptionMode{}}
	case d.All != nil:
		return &DisruptionMode{All: &AllDisruptionMode{}}
	default:
		return nil
	}
}

func mapCompositeDisruptionMode(d *schedulingv1alpha3.WorkloadCompositePodGroupDisruptionMode) *DisruptionMode {
	switch {
	case d.Single != nil:
		return &DisruptionMode{Single: &SingleDisruptionMode{}}
	case d.All != nil:
		return &DisruptionMode{All: &AllDisruptionMode{}}
	default:
		return nil
	}
}

func mapResourceClaims(claims []schedulingv1alpha3.WorkloadPodGroupResourceClaim) []ResourceClaim {
	result := make([]ResourceClaim, len(claims))
	for i := range claims {
		rc := ResourceClaim{Name: claims[i].Name}
		if claims[i].ResourceClaimName != nil {
			v := *claims[i].ResourceClaimName
			rc.ResourceClaimName = &v
		}
		if claims[i].ResourceClaimTemplateName != nil {
			v := *claims[i].ResourceClaimTemplateName
			rc.ResourceClaimTemplateName = &v
		}
		result[i] = rc
	}
	return result
}
