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

package corev1

import (
	"encoding/json"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
)

// PodPriority returns priority of the given pod.
func PodPriority(pod *v1.Pod) int32 {
	if pod.Spec.Priority != nil {
		return *pod.Spec.Priority
	}
	// When priority of a running pod is nil, it means it was created at a time
	// that there was no global default priority class and the priority class
	// name of the pod was empty. So, we resolve to the static default priority.
	return 0
}

// MatchNodeSelectorTerms checks whether the node labels and fields match node selector terms in ORed;
// nil or empty term matches no objects.
func MatchNodeSelectorTerms(
	node *v1.Node,
	nodeSelector *v1.NodeSelector,
) (bool, error) {
	if node == nil {
		return false, nil
	}
	return nodeaffinity.NewLazyErrorNodeSelector(nodeSelector).Match(node)
}

// GetAvoidPodsFromNodeAnnotations scans the list of annotations and
// returns the pods that needs to be avoided for this node from scheduling
func GetAvoidPodsFromNodeAnnotations(annotations map[string]string) (v1.AvoidPods, error) {
	var avoidPods v1.AvoidPods
	if len(annotations) > 0 && annotations[v1.PreferAvoidPodsAnnotationKey] != "" {
		err := json.Unmarshal([]byte(annotations[v1.PreferAvoidPodsAnnotationKey]), &avoidPods)
		if err != nil {
			return avoidPods, err
		}
	}
	return avoidPods, nil
}

// TolerationsTolerateTaint checks if taint is tolerated by any of the tolerations.
func TolerationsTolerateTaint(logger klog.Logger, tolerations []v1.Toleration, taint *v1.Taint, enableComparisonOperators bool) (bool, error) {
	var firstErr error
	for i := range tolerations {
		tolerated, err := tolerations[i].ToleratesTaint(taint, enableComparisonOperators)
		if tolerated {
			return true, nil
		}
		if firstErr == nil && err != nil {
			firstErr = err
		}
	}
	return false, firstErr
}

type taintsFilterFunc func(*v1.Taint) bool

// FindMatchingUntoleratedTaint checks if the given tolerations tolerate
// all the filtered taints, and returns a taint without a matching toleration.
// Prefer a cleanly untolerated taint over a comparison parse failure so callers
// get a stable result regardless of taint order.
// Returns true if there is an untolerated taint (or only comparison errors).
// Returns false if all taints are tolerated.
func FindMatchingUntoleratedTaint(logger klog.Logger, taints []v1.Taint, tolerations []v1.Toleration, inclusionFilter taintsFilterFunc, enableComparisonOperators bool) (v1.Taint, bool, error) {
	filteredTaints := getFilteredTaints(taints, inclusionFilter)
	var firstErrTaint v1.Taint
	var firstErr error
	for _, taint := range filteredTaints {
		tolerated, err := TolerationsTolerateTaint(logger, tolerations, &taint, enableComparisonOperators)
		if err != nil {
			if firstErr == nil {
				firstErr = err
				firstErrTaint = taint
			}
			continue
		}
		if !tolerated {
			return taint, true, nil
		}
	}
	if firstErr != nil {
		return firstErrTaint, true, firstErr
	}
	return v1.Taint{}, false, nil
}

// getFilteredTaints returns a list of taints satisfying the filter predicate
func getFilteredTaints(taints []v1.Taint, inclusionFilter taintsFilterFunc) []v1.Taint {
	if inclusionFilter == nil {
		return taints
	}
	filteredTaints := []v1.Taint{}
	for _, taint := range taints {
		if !inclusionFilter(&taint) {
			continue
		}
		filteredTaints = append(filteredTaints, taint)
	}
	return filteredTaints
}
