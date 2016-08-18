/*
Copyright 2016 The Kubernetes Authors.

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

package replicaset

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"

	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/sets"
)

type PodAnalysisResult struct {
	// Total number of pods created.
	Total int
	// Number of pods that are running and ready.
	RunningAndReady int
	// Number of pods that have been in unschedulable state for a while.
	Unschedulable int

	// TODO: Handle other scenarios like pod waiting too long for scheduler etc.
}

const (
	UnschedulableThreshold = 60 * time.Second
)

// A function that calculates how many pods from the list are in one of
// the meaningful (from the replica set perspective) states. This function is
// a temporary workaround against the current lack of ownerRef in pods.
func AnalysePods(replicaSet *v1beta1.ReplicaSet, allPods []util.FederatedObject, currentTime time.Time) (map[string]PodAnalysisResult, error) {
	selector, err := labelSelectorAsSelector(replicaSet.Spec.Selector)
	if err != nil {
		return nil, fmt.Errorf("invalid selector: %v", err)

	}
	result := make(map[string]PodAnalysisResult)

	for _, fedObject := range allPods {
		pod, isPod := fedObject.Object.(*api_v1.Pod)
		if !isPod {
			return nil, fmt.Errorf("invalid arg content - not a *pod")
		}

		if !selector.Empty() && selector.Matches(labels.Set(pod.Labels)) {
			status := result[fedObject.ClusterName]
			status.Total++
			for _, condition := range pod.Status.Conditions {
				if pod.Status.Phase == api_v1.PodRunning {
					if condition.Type == api_v1.PodReady {
						status.RunningAndReady++
					}
				} else {
					if condition.Type == api_v1.PodScheduled &&
						condition.Status == api_v1.ConditionFalse &&
						condition.Reason == "Unschedulable" &&
						condition.LastTransitionTime.Add(UnschedulableThreshold).Before(currentTime) {

						status.Unschedulable++
					}
				}
			}
			result[fedObject.ClusterName] = status
		}
	}
	return result, nil
}

// This is borrowed from pkg/api/unversioned/helper.go. There is no equivalent that works with
// estensions/v1beta1.
// LabelSelectorAsSelector converts the LabelSelector api type into a struct that implements
// labels.Selector.
// Note: This function should be kept in sync with the selector methods in pkg/labels/selector.go
// This function should be removed once we move to more efficient pod status accounting.
func labelSelectorAsSelector(ps *v1beta1.LabelSelector) (labels.Selector, error) {
	if ps == nil {
		return labels.Nothing(), nil
	}
	if len(ps.MatchLabels)+len(ps.MatchExpressions) == 0 {
		return labels.Everything(), nil
	}
	selector := labels.NewSelector()
	for k, v := range ps.MatchLabels {
		r, err := labels.NewRequirement(k, labels.EqualsOperator, sets.NewString(v))
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*r)
	}
	for _, expr := range ps.MatchExpressions {
		var op labels.Operator
		switch expr.Operator {
		case v1beta1.LabelSelectorOpIn:
			op = labels.InOperator
		case v1beta1.LabelSelectorOpNotIn:
			op = labels.NotInOperator
		case v1beta1.LabelSelectorOpExists:
			op = labels.ExistsOperator
		case v1beta1.LabelSelectorOpDoesNotExist:
			op = labels.DoesNotExistOperator
		default:
			return nil, fmt.Errorf("%q is not a valid pod selector operator", expr.Operator)
		}
		r, err := labels.NewRequirement(expr.Key, op, sets.NewString(expr.Values...))
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*r)
	}
	return selector, nil
}
