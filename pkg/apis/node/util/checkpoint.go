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

// Package util provides pod-template sanitization shared by checkpoint capture
// in the kubelet and restore validation in admission and the kubelet. Keeping
// sanitization in one place ensures capture and comparison apply identical rules.
package util

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SanitizePodTemplate builds the portable PodTemplateSpec recorded in
// PodCheckpoint.status.checkpointedPodTemplate from a pod, and is also used to
// normalize a restoring pod before comparing it to that record. It captures the
// user-meaningful metadata and spec while dropping the node binding, restore
// invocation, ephemeral containers, scheduling gates, and server-assigned
// metadata. Source scheduling constraints are preserved in full.
//
// The pod is not mutated; a deep copy is taken first.
//
// Restore comparisons add the checkpoint-node affinity pin to a copy of the
// captured spec; they do not remove any user-supplied affinity requirements.
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
	// Drop the restore invocation. The source pod has none, while a restoring pod
	// supplies a checkpoint reference with options specific to that restore
	// attempt. It does not describe the checkpointed workload.
	tmpl.Spec.RestoreFrom = nil
	// Ephemeral containers are excluded from the CRI checkpoint container set
	// and cannot be specified when creating a restore Pod.
	tmpl.Spec.EphemeralContainers = nil
	// Gates delay scheduling and may be removed before the kubelet observes the
	// restore Pod. They do not describe the checkpointed workload.
	tmpl.Spec.SchedulingGates = nil
	return tmpl
}

// AddRestoreNodeAffinity restricts placement to the checkpoint node without
// removing any source constraints. Callers must own spec; this mutates it.
// Reapplying the pin is safe when admission is reinvoked or the source Pod was
// itself restored from a checkpoint on the same node.
func AddRestoreNodeAffinity(spec *v1.PodSpec, nodeName string) {
	if spec.Affinity == nil {
		spec.Affinity = &v1.Affinity{}
	}
	if spec.Affinity.NodeAffinity == nil {
		spec.Affinity.NodeAffinity = &v1.NodeAffinity{}
	}
	na := spec.Affinity.NodeAffinity
	pin := v1.NodeSelectorRequirement{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{nodeName}}
	if na.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		na.RequiredDuringSchedulingIgnoredDuringExecution = &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{{MatchFields: []v1.NodeSelectorRequirement{pin}}}}
		return
	}
	for i := range na.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms {
		term := &na.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms[i]
		// Empty terms match no nodes; adding the pin would make them satisfiable.
		if len(term.MatchExpressions) == 0 && len(term.MatchFields) == 0 {
			continue
		}
		found := false
		for _, requirement := range term.MatchFields {
			if requirement.Key == pin.Key && requirement.Operator == pin.Operator && len(requirement.Values) == 1 && requirement.Values[0] == nodeName {
				found = true
				break
			}
		}
		if !found {
			term.MatchFields = append(term.MatchFields, pin)
		}
	}
}
