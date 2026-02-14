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

package affinities

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// DedupAffinityFields de-duplicates sub-fields in an affinity and keeps any strict supersets.
func DedupAffinityFields(affinity *api.Affinity) *api.Affinity {
	if affinity == nil {
		return nil
	}
	dedupNodeAffinity(affinity)
	dedupPodAffinity(affinity)
	dedupPodAntiAffinity(affinity)
	return affinity
}

func dedupNodeAffinity(affinity *api.Affinity) {
	if affinity.NodeAffinity == nil {
		return
	}

	if affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
		affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms = dedupNodeSelectorTerms(affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms)
	}

	if affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution != nil {
		affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution = dedupPreferredSchedulingTerms(affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution)
	}
}

func dedupPodAffinity(affinity *api.Affinity) {
	if affinity.PodAffinity == nil {
		return
	}

	affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution = dedupPodAffinityTerms(affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution)
	affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution = dedupWeightedPodAffinityTerms(affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution)
}

func dedupPodAntiAffinity(affinity *api.Affinity) {
	if affinity.PodAntiAffinity == nil {
		return
	}

	affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution = dedupPodAffinityTerms(affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution)
	affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution = dedupWeightedPodAffinityTerms(affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution)
}

func dedupNodeSelectorTerms(terms []api.NodeSelectorTerm) []api.NodeSelectorTerm {
	if len(terms) <= 1 {
		return terms
	}

	var deduped []api.NodeSelectorTerm
	for i, term := range terms {
		isSubset := false
		for j, otherTerm := range terms {
			if i == j {
				continue
			}
			if isNodeSelectorTermSubset(term, otherTerm) {
				if !isNodeSelectorTermSubset(otherTerm, term) { // strict subset
					isSubset = true
					break
				} else { // equal
					if i > j {
						isSubset = true
						break
					}
				}
			}
		}
		if !isSubset {
			deduped = append(deduped, term)
		}
	}
	return deduped
}

func dedupPreferredSchedulingTerms(terms []api.PreferredSchedulingTerm) []api.PreferredSchedulingTerm {
	if len(terms) <= 1 {
		return terms
	}

	var deduped []api.PreferredSchedulingTerm
	for i, term := range terms {
		isSubset := false
		for j, otherTerm := range terms {
			if i == j {
				continue
			}
			if term.Weight == otherTerm.Weight && isNodeSelectorTermSubset(term.Preference, otherTerm.Preference) {
				if !isNodeSelectorTermSubset(otherTerm.Preference, term.Preference) { // strict subset
					isSubset = true
					break
				} else { // equal
					if i > j {
						isSubset = true
						break
					}
				}
			}
		}
		if !isSubset {
			deduped = append(deduped, term)
		}
	}
	return deduped
}

func dedupPodAffinityTerms(terms []api.PodAffinityTerm) []api.PodAffinityTerm {
	if len(terms) <= 1 {
		return terms
	}

	var deduped []api.PodAffinityTerm
	for i, term := range terms {
		isSubset := false
		for j, otherTerm := range terms {
			if i == j {
				continue
			}
			if isPodAffinityTermSubset(term, otherTerm) {
				if !isPodAffinityTermSubset(otherTerm, term) { // strict subset
					isSubset = true
					break
				} else { // equal
					if i > j {
						isSubset = true
						break
					}
				}
			}
		}
		if !isSubset {
			deduped = append(deduped, term)
		}
	}
	return deduped
}

func dedupWeightedPodAffinityTerms(terms []api.WeightedPodAffinityTerm) []api.WeightedPodAffinityTerm {
	if len(terms) <= 1 {
		return terms
	}

	var deduped []api.WeightedPodAffinityTerm
	for i, term := range terms {
		isSubset := false
		for j, otherTerm := range terms {
			if i == j {
				continue
			}
			if term.Weight == otherTerm.Weight && isPodAffinityTermSubset(term.PodAffinityTerm, otherTerm.PodAffinityTerm) {
				if !isPodAffinityTermSubset(otherTerm.PodAffinityTerm, term.PodAffinityTerm) { // strict subset
					isSubset = true
					break
				} else { // equal
					if i > j {
						isSubset = true
						break
					}
				}
			}
		}
		if !isSubset {
			deduped = append(deduped, term)
		}
	}
	return deduped
}

func isNodeSelectorTermSubset(a, b api.NodeSelectorTerm) bool {
	return isNodeSelectorRequirementSubset(a.MatchExpressions, b.MatchExpressions) && isNodeSelectorRequirementSubset(a.MatchFields, b.MatchFields)
}

func isPodAffinityTermSubset(a, b api.PodAffinityTerm) bool {
	if a.TopologyKey != b.TopologyKey {
		return false
	}

	if !isLabelSelectorSubset(a.LabelSelector, b.LabelSelector) {
		return false
	}

	if !isStringSliceSubset(a.Namespaces, b.Namespaces) {
		return false
	}

	return true
}

func isNodeSelectorRequirementSubset(a, b []api.NodeSelectorRequirement) bool {
	for _, reqA := range a {
		found := false
		for _, reqB := range b {
			if reqA.Key == reqB.Key && reqA.Operator == reqB.Operator && isStringSliceSubset(reqA.Values, reqB.Values) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func isLabelSelectorSubset(a, b *metav1.LabelSelector) bool {
	if a == nil || b == nil {
		return a == b
	}

	for k, v := range a.MatchLabels {
		if w, ok := b.MatchLabels[k]; !ok || v != w {
			return false
		}
	}

	return isLabelSelectorRequirementSubset(a.MatchExpressions, b.MatchExpressions)
}

func isLabelSelectorRequirementSubset(a, b []metav1.LabelSelectorRequirement) bool {
	for _, reqA := range a {
		found := false
		for _, reqB := range b {
			if reqA.Key == reqB.Key && reqA.Operator == reqB.Operator && isStringSliceSubset(reqA.Values, reqB.Values) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func isStringSliceSubset(a, b []string) bool {
	for _, valA := range a {
		found := false
		for _, valB := range b {
			if valA == valB {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}
