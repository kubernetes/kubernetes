/*
Copyright 2026 The Kubernetes Authors.

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

package preemption

import (
	v1 "k8s.io/api/core/v1"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	nodeaffinity "k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
)

// CanRescheduleElsewhere reports whether a replacement of pod, rebuilt from the
// same spec without a node assignment, could satisfy pod's own hard scheduling
// constraints (nodeSelector, required nodeAffinity and taints) on at least one
// node other than current.
//
// Only the pod's own constraints are read; owner, priority and PDB state are
// ignored. Resources, cordon state and soft constraints (preferred nodeAffinity,
// topology spread) are deliberately not considered: a resource shortfall may
// clear up by queueing, while a constraint mismatch cannot.
func CanRescheduleElsewhere(logger klog.Logger, pod *v1.Pod, current string, nodes []*v1.Node, enableComparisonOperators bool) bool {
	for _, node := range nodes {
		if node == nil || node.Name == current {
			continue
		}
		if podMatchesNodeConstraints(logger, pod, node, enableComparisonOperators) {
			return true
		}
	}
	return false
}

func podMatchesNodeConstraints(logger klog.Logger, pod *v1.Pod, node *v1.Node, enableComparisonOperators bool) bool {
	// Reuse the same required node affinity evaluation as the nodeaffinity
	// Filter plugin: pod.Spec.NodeSelector and required nodeAffinity together
	// form the scheduler's node constraints for a pod.
	if ok, err := nodeaffinity.NewRequiredNodeAffinity(pod.Spec.NodeSelector, pod.Spec.Affinity).Match(node); err != nil || !ok {
		return false
	}
	if _, untolerated := corev1helpers.FindMatchingUntoleratedTaint(logger, node.Spec.Taints, pod.Spec.Tolerations, doNotScheduleTaintsFilterFunc, enableComparisonOperators); untolerated {
		return false
	}
	return true
}

// doNotScheduleTaintsFilterFunc mirrors the taint-toleration Filter plugin: only
// NoSchedule and NoExecute taints prevent a pod from being scheduled on a node.
func doNotScheduleTaintsFilterFunc(t *v1.Taint) bool {
	return t.Effect == v1.TaintEffectNoSchedule || t.Effect == v1.TaintEffectNoExecute
}
