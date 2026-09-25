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

package preemption

import (
	v1 "k8s.io/api/core/v1"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	nodeaffinity "k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
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
	// Evaluate pod.Spec.NodeSelector and required nodeAffinity the same way as
	// the nodeaffinity Filter plugin: build the matcher once per pod and reuse
	// it for every node.
	requiredNodeAffinity := nodeaffinity.GetRequiredNodeAffinity(pod)
	for _, node := range nodes {
		if node == nil || node.Name == current {
			continue
		}
		if ok, err := requiredNodeAffinity.Match(node); err != nil || !ok {
			continue
		}
		// Only NoSchedule and NoExecute taints prevent scheduling on a node;
		// filter them the same way as the taint-toleration Filter plugin.
		if _, untolerated := corev1helpers.FindMatchingUntoleratedTaint(logger, node.Spec.Taints, pod.Spec.Tolerations, helper.DoNotScheduleTaintsFilterFunc(), enableComparisonOperators); untolerated {
			continue
		}
		return true
	}
	return false
}
