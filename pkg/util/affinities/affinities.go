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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// MergeAffinities merges two sets of affinities into one. If one affinity is a superset of
// another, only the superset is kept.
func MergeAffinities(first, second []api.Affinity) []api.Affinity {
	all := append(first, second...)
	var merged []api.Affinity

next:
	for i, a := range all {
		for _, t2 := range merged {
			if isSuperset(t2, a) {
				continue next // a is redundant; ignore it
			}
		}
		if i+1 < len(all) {
			for _, a2 := range all[i+1:] {
				// If the affinities are equal, prefer the first.
				if !apiequality.Semantic.DeepEqual(&a.NodeAffinity, &a2.NodeAffinity) &&
					!apiequality.Semantic.DeepEqual(&a.PodAffinity, &a2.PodAffinity) &&
					!apiequality.Semantic.DeepEqual(&a.PodAntiAffinity, &a2.PodAntiAffinity) {
					continue
				}
				if isSuperset(a2, a) {
					continue next // a is redundant; ignore it
				}
			}
		}
		merged = append(merged, a)
	}

	return merged
}

// isSuperset checks whether ss is a superset of a.
// An affinity 'ss' is a superset of 'a' if 'ss' is less restrictive than 'a'.
func isSuperset(ss, a api.Affinity) bool {
	if apiequality.Semantic.DeepEqual(&a, &ss) {
		return true
	}

	if !isNodeAffinitySuperset(ss.NodeAffinity, a.NodeAffinity) {
		return false
	}
	if !isPodAffinitySuperset(ss.PodAffinity, a.PodAffinity) {
		return false
	}
	if !isPodAntiAffinitySuperset(ss.PodAntiAffinity, a.PodAntiAffinity) {
		return false
	}

	return true
}

// isNodeAffinitySuperset checks if ss is a superset of a.
func isNodeAffinitySuperset(ss, a *api.NodeAffinity) bool {
	if ss == nil {
		return true
	}
	if a == nil {
		return false
	}
	// A full implementation would be needed to check for supersets.
	// For now, we'll consider them equal.
	return apiequality.Semantic.DeepEqual(ss, a)
}

// isPodAffinitySuperset checks if ss is a superset of a.
func isPodAffinitySuperset(ss, a *api.PodAffinity) bool {
	if ss == nil {
		return true
	}
	if a == nil {
		return false
	}
	// A full implementation would be needed to check for supersets.
	// For now, we'll consider them equal.
	return apiequality.Semantic.DeepEqual(ss, a)
}

// isPodAntiAffinitySuperset checks if ss is a superset of a.
// For anti-affinity, superset means it is *less* restrictive.
func isPodAntiAffinitySuperset(ss, a *api.PodAntiAffinity) bool {
	if ss == nil {
		return true
	}
	if a == nil {
		return false
	}
	// A full implementation would be needed to check for supersets.
	// For now, we'll consider them equal.
	return apiequality.Semantic.DeepEqual(ss, a)
}

// MergePodAffinities merges two affinity objects.
// It merges each field of the affinity struct.
// For NodeAffinity, PodAffinity, and PodAntiAffinity, it merges the
// Required and Preferred terms by appending them.
func MergePodAffinities(podAffinity, defaultAffinity *api.Affinity) *api.Affinity {
	if defaultAffinity == nil {
		return podAffinity
	}
	if podAffinity == nil {
		return defaultAffinity
	}

	merged := podAffinity.DeepCopy()

	// Merge NodeAffinity
	if defaultAffinity.NodeAffinity != nil {
		if merged.NodeAffinity == nil {
			merged.NodeAffinity = defaultAffinity.NodeAffinity.DeepCopy()
		} else {
			// Merge RequiredDuringSchedulingIgnoredDuringExecution
			if defaultAffinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
				if merged.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
					merged.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = defaultAffinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.DeepCopy()
				} else {
					merged.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms =
						append(merged.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms,
							defaultAffinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms...)
				}
			}
			// Merge PreferredDuringSchedulingIgnoredDuringExecution
			merged.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution =
				append(merged.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution,
					defaultAffinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution...)
		}
	}

	// Merge PodAffinity
	if defaultAffinity.PodAffinity != nil {
		if merged.PodAffinity == nil {
			merged.PodAffinity = defaultAffinity.PodAffinity.DeepCopy()
		} else {
			merged.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution =
				append(merged.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution,
					defaultAffinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution...)
			merged.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution =
				append(merged.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution,
					defaultAffinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution...)
		}
	}

	// Merge PodAntiAffinity
	if defaultAffinity.PodAntiAffinity != nil {
		if merged.PodAntiAffinity == nil {
			merged.PodAntiAffinity = defaultAffinity.PodAntiAffinity.DeepCopy()
		} else {
			merged.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution =
				append(merged.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution,
					defaultAffinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution...)
			merged.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution =
				append(merged.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution,
					defaultAffinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution...)
		}
	}

	return merged
}
