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
// TODO: Implement MapCompositeGroupConfig once the WorkloadCompositePodGroup*
// types and CompositePodGroup resource are added to scheduling.k8s.io/v1alpha3.
//
// func MapCompositeGroupConfig(
//	policy *schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy,
//	constraints *schedulingv1alpha3.WorkloadCompositePodGroupSchedulingConstraints,
//	disruption *schedulingv1alpha3.WorkloadCompositePodGroupDisruptionMode,
//	) *SchedulingConfig {
//		cfg := &SchedulingConfig{}
//	}

func MapPodGroupConfig(
	policy *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy,
	constraints *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints,
	disruption *schedulingv1alpha3.WorkloadPodGroupDisruptionMode,
	claims []schedulingv1alpha3.WorkloadPodGroupResourceClaim,
) *SchedulingConfig {
	cfg := &SchedulingConfig{}

	if policy != nil {
		cfg.Policy = mapSchedulingPolicy(policy)
	}
	if constraints != nil {
		cfg.Constraints = mapTopologyConstraints(constraints)
	}
	if disruption != nil {
		cfg.DisruptionMode = mapDisruptionMode(disruption)
	}
	if len(claims) > 0 {
		cfg.ResourceClaims = mapResourceClaims(claims)
	}

	return cfg
}

func mapSchedulingPolicy(p *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy) *SchedulingPolicy {
	sp := &SchedulingPolicy{}
	switch {
	case p.Basic != nil:
		sp.Basic = &BasicSchedulingPolicy{}
	case p.Gang != nil:
		sp.Gang = &GangSchedulingPolicy{MinCount: p.Gang.MinCount}
	}
	return sp
}

func mapTopologyConstraints(c *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints) *SchedulingConstraints {
	if len(c.Topology) == 0 {
		return &SchedulingConstraints{}
	}
	topology := make([]schedulingv1alpha3.TopologyConstraint, len(c.Topology))
	copy(topology, c.Topology)
	return &SchedulingConstraints{Topology: topology}
}

func mapDisruptionMode(d *schedulingv1alpha3.WorkloadPodGroupDisruptionMode) *DisruptionMode {
	dm := &DisruptionMode{}
	switch {
	case d.Single != nil:
		dm.Single = &SingleDisruptionMode{}
	case d.All != nil:
		dm.All = &AllDisruptionMode{}
	}
	return dm
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
