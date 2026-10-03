/*
Copyright 2020 The Kubernetes Authors.

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

package topology

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
)

// GetZoneKey is a helper function that builds a string identifier that is unique per failure-zone;
// it returns empty-string for no zone.
// Since there are currently two separate zone keys:
//   - "failure-domain.beta.kubernetes.io/zone"
//   - "topology.kubernetes.io/zone"
//
// GetZoneKey will first check failure-domain.beta.kubernetes.io/zone and if not exists, will then check
// topology.kubernetes.io/zone
func GetZoneKey(node *v1.Node) string {
	labels := node.Labels
	if labels == nil {
		return ""
	}

	// TODO: "failure-domain.beta..." names are deprecated, but will
	// stick around a long time due to existing on old extant objects like PVs.
	// Maybe one day we can stop considering them (see #88493).
	zone, ok := labels[v1.LabelFailureDomainBetaZone]
	if !ok {
		zone, _ = labels[v1.LabelTopologyZone]
	}

	region, ok := labels[v1.LabelFailureDomainBetaRegion]
	if !ok {
		region, _ = labels[v1.LabelTopologyRegion]
	}

	if region == "" && zone == "" {
		return ""
	}

	// We include the null character just in case region or failureDomain has a colon
	// (We do assume there's no null characters in a region or failureDomain)
	// As a nice side-benefit, the null character is not printed by fmt.Print or glog
	return region + ":\x00:" + zone
}

// MatchTopologySelectorTerms checks whether the given labels match any of the
// topology selector terms. An empty term list matches all objects, while an
// empty term matches no objects.
func MatchTopologySelectorTerms(topologySelectorTerms []v1.TopologySelectorTerm, lbls labels.Set) bool {
	if len(topologySelectorTerms) == 0 {
		return true
	}

	for _, term := range topologySelectorTerms {
		if len(term.MatchLabelExpressions) == 0 {
			continue
		}

		selector, err := topologySelectorRequirementsAsSelector(term.MatchLabelExpressions)
		if err == nil && selector.Matches(lbls) {
			return true
		}
	}

	return false
}

func topologySelectorRequirementsAsSelector(requirements []v1.TopologySelectorLabelRequirement) (labels.Selector, error) {
	if len(requirements) == 0 {
		return labels.Nothing(), nil
	}

	selector := labels.NewSelector()
	for _, expression := range requirements {
		requirement, err := labels.NewRequirement(expression.Key, selection.In, expression.Values)
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*requirement)
	}

	return selector, nil
}
