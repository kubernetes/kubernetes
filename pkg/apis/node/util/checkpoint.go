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

// Package util holds helpers shared between the PodCheckpoint controller (which
// captures the checkpointed pod template) and the kubelet (which validates a
// restoring pod against it). Keeping the sanitization in one place ensures the
// capture and the compare apply identical rules.
package util

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SanitizePodTemplate builds the portable PodTemplateSpec recorded in
// PodCheckpoint.status.checkpointedPodTemplate from a pod, and is also used to
// normalize a restoring pod before comparing it to that record. It captures the
// user-meaningful metadata and spec while dropping fields that are node-local,
// cluster-specific, or server-assigned identity so that the capture and the
// equality check are symmetric.
//
// The pod is not mutated; a deep copy is taken first.
//
// Because the same function is applied both when capturing the template and
// when normalizing a restoring pod for the equality check, any field dropped
// here is dropped symmetrically on both sides, so stripping cannot cause a
// false mismatch.
func SanitizePodTemplate(pod *v1.Pod) *v1.PodTemplateSpec {
	if pod == nil {
		return nil
	}
	src := pod.DeepCopy()

	tmpl := &v1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			// Keep only the user-meaningful metadata.
			Labels:          src.Labels,
			Annotations:     src.Annotations,
			OwnerReferences: src.OwnerReferences,
		},
		Spec: src.Spec,
	}

	// Drop node-local scheduling state; the restore is pinned to the
	// checkpoint's node via status.nodeName, not via the recorded template.
	tmpl.Spec.NodeName = ""
	// Drop restoreFrom: the source pod has none, while a restoring pod has a
	// checkpoint reference; clearing it keeps capture and compare symmetric.
	tmpl.Spec.RestoreFrom = nil
	// Drop scheduling constraints that pin to a specific node by identity
	// (kubernetes.io/hostname or metadata.name). These reference the source
	// node and are not portable; the restore node is selected out of band.
	stripNodeIdentityConstraints(&tmpl.Spec)

	return tmpl
}

// nodeIdentityKeys are the well-known label/field keys that pin scheduling to a
// specific node's identity.
var nodeIdentityKeys = map[string]bool{
	v1.LabelHostname: true, // kubernetes.io/hostname
	"metadata.name":  true,
}

// stripNodeIdentityConstraints removes nodeSelector entries and node-affinity
// match expressions/fields that reference a node's identity, so the recorded
// template does not pin to the source node.
func stripNodeIdentityConstraints(spec *v1.PodSpec) {
	for k := range spec.NodeSelector {
		if nodeIdentityKeys[k] {
			delete(spec.NodeSelector, k)
		}
	}
	if len(spec.NodeSelector) == 0 {
		spec.NodeSelector = nil
	}

	if spec.Affinity == nil {
		return
	}
	if spec.Affinity.NodeAffinity == nil {
		if *spec.Affinity == (v1.Affinity{}) {
			spec.Affinity = nil
		}
		return
	}
	na := spec.Affinity.NodeAffinity

	if req := na.RequiredDuringSchedulingIgnoredDuringExecution; req != nil {
		terms := req.NodeSelectorTerms[:0]
		for _, term := range req.NodeSelectorTerms {
			if pruneSelectorTerm(&term); !selectorTermEmpty(term) {
				terms = append(terms, term)
			}
		}
		if len(terms) == 0 {
			na.RequiredDuringSchedulingIgnoredDuringExecution = nil
		} else {
			req.NodeSelectorTerms = terms
		}
	}

	if pref := na.PreferredDuringSchedulingIgnoredDuringExecution; pref != nil {
		kept := pref[:0]
		for _, p := range pref {
			if pruneSelectorTerm(&p.Preference); !selectorTermEmpty(p.Preference) {
				kept = append(kept, p)
			}
		}
		if len(kept) == 0 {
			na.PreferredDuringSchedulingIgnoredDuringExecution = nil
		} else {
			na.PreferredDuringSchedulingIgnoredDuringExecution = kept
		}
	}

	if na.RequiredDuringSchedulingIgnoredDuringExecution == nil &&
		na.PreferredDuringSchedulingIgnoredDuringExecution == nil {
		spec.Affinity.NodeAffinity = nil
	}
	if *spec.Affinity == (v1.Affinity{}) {
		// Admission may have created these wrappers solely for the restore-node
		// requirement. Drop them after pruning so equality is not sensitive to
		// nil versus an empty struct.
		spec.Affinity = nil
	}
}

func pruneSelectorTerm(term *v1.NodeSelectorTerm) {
	term.MatchExpressions = filterRequirements(term.MatchExpressions)
	term.MatchFields = filterRequirements(term.MatchFields)
}

func filterRequirements(reqs []v1.NodeSelectorRequirement) []v1.NodeSelectorRequirement {
	kept := reqs[:0]
	for _, r := range reqs {
		if !nodeIdentityKeys[r.Key] {
			kept = append(kept, r)
		}
	}
	if len(kept) == 0 {
		return nil
	}
	return kept
}

func selectorTermEmpty(term v1.NodeSelectorTerm) bool {
	return len(term.MatchExpressions) == 0 && len(term.MatchFields) == 0
}
