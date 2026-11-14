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

package daemonset

import (
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	v1helper "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
)

// NodeShouldRunDaemonPodWithPrecomputed checks a set of preconditions against a (node, daemonset)
// pair. It is meant for callers that evaluate many nodes against the same DaemonSet: tolerations and
// requiredNodeAffinity depend only on ds, so per-node loops build them once instead of once per node.
// Parsing the required node affinity dominates the cost of this check.
//
// The function cannot check its inputs, so callers must pass:
//   - tolerations: the tolerations of ds's pod template after the default daemon pod tolerations are
//     added, as the DaemonSet controller does for the pods it creates. Without the defaults, nodes with
//     taints such as node.kubernetes.io/not-ready or node.kubernetes.io/unschedulable are wrongly
//     reported as not running the DaemonSet.
//   - requiredNodeAffinity:
//     nodeaffinity.NewRequiredNodeAffinity(ds.Spec.Template.Spec.NodeSelector, ds.Spec.Template.Spec.Affinity)
//     for the same ds. If the affinity has terms that fail to parse and no other term matches, the
//     node is treated as not matching; the parse error is not returned.
//   - enableTaintTolerationComparisonOperators: the value of the TaintTolerationComparisonOperators
//     feature gate in the kube-controller-manager that runs the DaemonSet controller. It decides
//     whether tolerations with the Gt and Lt operators can tolerate taints. Callers outside
//     kube-controller-manager that pass a different value get a wrong answer for such tolerations.
//
// Returned booleans are:
//   - shouldRun:
//     Returns true when a daemonset should run on the node if a daemonset pod is not already
//     running on that node.
//   - shouldContinueRunning:
//     Returns true when a daemonset pod that is already running on the node is not evicted, that is
//     when every untolerated taint of the node has the NoSchedule effect and none has NoExecute.
func NodeShouldRunDaemonPodWithPrecomputed(logger klog.Logger, node *v1.Node, ds *appsv1.DaemonSet, tolerations []v1.Toleration, requiredNodeAffinity nodeaffinity.RequiredNodeAffinity, enableTaintTolerationComparisonOperators bool) (shouldRun, shouldContinueRunning bool) {
	// If the daemon set specifies a node name, check that it matches with node.Name.
	if ds.Spec.Template.Spec.NodeName != "" && ds.Spec.Template.Spec.NodeName != node.Name {
		return false, false
	}

	taints := node.Spec.Taints
	fitsNodeAffinity, fitsTaints := predicates(logger, node, taints, tolerations, requiredNodeAffinity, enableTaintTolerationComparisonOperators)
	if !fitsNodeAffinity {
		return false, false
	}

	if !fitsTaints {
		// Scheduled daemon pods should continue running if they tolerate NoExecute taint.
		_, hasUntoleratedTaint := v1helper.FindMatchingUntoleratedTaint(logger, taints, tolerations, func(t *v1.Taint) bool {
			return t.Effect == v1.TaintEffectNoExecute
		}, enableTaintTolerationComparisonOperators)
		return false, !hasUntoleratedTaint
	}

	return true, true
}

// predicates checks if a DaemonSet's pod can run on a node. tolerations and
// requiredNodeAffinity must be derived from the DaemonSet's pod template
// (see NodeShouldRunDaemonPodWithPrecomputed).
func predicates(logger klog.Logger, node *v1.Node, taints []v1.Taint, tolerations []v1.Toleration, requiredNodeAffinity nodeaffinity.RequiredNodeAffinity, enableTaintTolerationComparisonOperators bool) (fitsNodeAffinity, fitsTaints bool) {
	// Ignore parsing errors for backwards compatibility.
	fitsNodeAffinity, _ = requiredNodeAffinity.Match(node)
	_, hasUntoleratedTaint := v1helper.FindMatchingUntoleratedTaint(logger, taints, tolerations, func(t *v1.Taint) bool {
		return t.Effect == v1.TaintEffectNoExecute || t.Effect == v1.TaintEffectNoSchedule
	}, enableTaintTolerationComparisonOperators)
	fitsTaints = !hasUntoleratedTaint
	return
}
