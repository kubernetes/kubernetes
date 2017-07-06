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
	"time"

	api_v1 "k8s.io/api/core/v1"
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

// AnalyzePods calculates how many pods from the list are in one of
// the meaningful (from the replica set perspective) states. This function is
// a temporary workaround against the current lack of ownerRef in pods.
func AnalyzePods(pods *api_v1.PodList, currentTime time.Time) PodAnalysisResult {
	result := PodAnalysisResult{}
	for _, pod := range pods.Items {
		result.Total++
		for _, condition := range pod.Status.Conditions {
			if pod.Status.Phase == api_v1.PodRunning {
				if condition.Type == api_v1.PodReady {
					result.RunningAndReady++
				}
			} else if condition.Type == api_v1.PodScheduled &&
				condition.Status == api_v1.ConditionFalse &&
				condition.Reason == api_v1.PodReasonUnschedulable &&
				condition.LastTransitionTime.Add(UnschedulableThreshold).Before(currentTime) {

				result.Unschedulable++
			}
		}
	}
	return result
}
