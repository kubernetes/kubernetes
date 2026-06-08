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

// mapWorkloadInput translates the leaf-level scheduling.k8s.io building blocks
// into the IR for a WorkloadItem's config. It returns a non-nil config, leaving
// a field nil when its input is nil so resolveSchedulingConfig can fall back to
// defaults field-by-field.
//
// TODO: Add mapCompositeGroupInput once the WorkloadCompositePodGroup* types
// and CompositePodGroup resource are added to scheduling.k8s.io/v1alpha3.
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

func mapTopologyConstraints(c *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints) *SchedulingConstraints {
	if len(c.Topology) == 0 {
		return nil
	}
	topology := make([]schedulingv1alpha3.TopologyConstraint, len(c.Topology))
	copy(topology, c.Topology)
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
