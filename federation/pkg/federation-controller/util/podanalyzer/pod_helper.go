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

package podanalyzer

import (
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
)

type PodAnalysisResult struct {
	// Total number of pods created.
	Total int
	// Number of pods that are running and ready.
	RunningAndReady int
	// Number of pods that have been in unschedulable state for UnshedulableThreshold seconds.
	Unschedulable int

	// TODO: Handle other scenarios like pod waiting too long for scheduler etc.
}

const (
	// TODO: make it configurable
	UnschedulableThreshold = 60 * time.Second
)

// A function that calculates how many pods from the list are in one of
// the meaningful (from the replica set perspective) states. This function is
// a temporary workaround against the current lack of ownerRef in pods.
func AnalysePods(selectorv1 *metav1.LabelSelector, allPods []util.FederatedObject, currentTime time.Time) (map[string]PodAnalysisResult, error) {
	selector, err := metav1.LabelSelectorAsSelector(selectorv1)
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
