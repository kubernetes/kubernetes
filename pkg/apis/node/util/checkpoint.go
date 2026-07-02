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

// Package util provides pod-template sanitization for checkpoint capture in the
// kubelet.
package util

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SanitizePodTemplate builds the portable PodTemplateSpec recorded in
// PodCheckpoint.status.checkpointedPodTemplate from a pod. It captures the
// user-meaningful metadata and spec while dropping the node binding, ephemeral
// containers, scheduling gates, and server-assigned metadata. Source
// scheduling constraints are preserved in full.
//
// The pod is not mutated; a deep copy is taken first.
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
	// Ephemeral containers are excluded from the CRI checkpoint container set
	// and cannot be specified when creating a restore Pod.
	tmpl.Spec.EphemeralContainers = nil
	// Gates delay scheduling and may be removed before the kubelet observes the
	// restore Pod. They do not describe the checkpointed workload.
	tmpl.Spec.SchedulingGates = nil
	return tmpl
}
