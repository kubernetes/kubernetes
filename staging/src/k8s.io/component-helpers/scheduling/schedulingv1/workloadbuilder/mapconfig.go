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

// MapPodGroupConfig translates the leaf-level scheduling.k8s.io building blocks
// into the IR for a WorkloadItem's UserConfig or DefaultConfig. It returns a
// non-nil config, leaving a field nil when its input is nil so resolveConfig can
// fall back to defaults field-by-field.
//
// TODO: Add MapCompositeGroupConfig once the WorkloadCompositePodGroup* types
// and CompositePodGroup resource are added to scheduling.k8s.io/v1alpha3.
func MapPodGroupConfig(
	policy *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy,
	constraints *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints,
	disruption *schedulingv1alpha3.WorkloadPodGroupDisruptionMode,
	claims []schedulingv1alpha3.WorkloadPodGroupResourceClaim,
) *SchedulingConfig {
	cfg := &SchedulingConfig{}

	if policy != nil {
		if p := mapSchedulingPolicy(policy); p != nil {
			cfg.Policy = p
		}
	}
	if constraints != nil {
		if c := mapTopologyConstraints(constraints); c != nil {
			cfg.Constraints = c
		}
	}
	if disruption != nil {
		if d := mapDisruptionMode(disruption); d != nil {
			cfg.DisruptionMode = d
		}
	}
	if len(claims) > 0 {
		cfg.ResourceClaims = mapResourceClaims(claims)
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
		return &SchedulingPolicy{Gang: &GangSchedulingPolicy{MinCount: p.Gang.MinCount}}
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
		result[i] = ResourceClaim{
			Name:                      claims[i].Name,
			ResourceClaimName:         claims[i].ResourceClaimName,
			ResourceClaimTemplateName: claims[i].ResourceClaimTemplateName,
		}
	}
	return result
}
