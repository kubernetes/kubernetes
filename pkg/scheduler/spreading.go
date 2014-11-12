/*
Copyright 2014 Google Inc. All rights reserved.

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

package scheduler

import (
	"math/rand"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// CalculateSpreadPriority spreads pods by minimizing the number of pods on the same machine with the same labels.
// Importantly, if there are services in the system that span multiple heterogenous sets of pods, this spreading priority
// may not provide optimal spreading for the members of that Service.
// TODO: consider if we want to include Service label sets in the scheduling priority.
func CalculateSpreadPriority(pod api.Pod, podLister PodLister, minionLister MinionLister) (HostPriorityList, error) {
	pods, err := getMatchingPods(pod.Labels, podLister)
	if err != nil {
		return nil, err
	}
	minions, err := minionLister.List()
	if err != nil {
		return nil, err
	}

	counts := map[string]int{}
	for _, otherPod := range pods {
		counts[otherPod.CurrentState.Host] += commonLabelsCount(pod.Labels, otherPod.Labels)
	}

	result := []HostPriority{}
	for _, minion := range minions.Items {
		result = append(result, HostPriority{host: minion.Name, score: counts[minion.Name]})
	}
	return result, nil
}

func NewSpreadingScheduler(podLister PodLister, minionLister MinionLister, predicates []FitPredicate, random *rand.Rand) Scheduler {
	return NewGenericScheduler(predicates, CalculateSpreadPriority, podLister, random)
}

// Returns a map of pod name to pod with an entry for all pods that have at least one label in the label set.
// This could possibly be replaced with an orTerm selector but it doesn't exist yet.
// TODO: look at replacing podSet key with pod.ObjectMeta.UID when it's populated
func getMatchingPods(labelSet map[string]string, podLister PodLister) (map[string]api.Pod, error) {
	podSet := map[string]api.Pod{}
	for k, v := range labelSet {
		pods, err := podLister.ListPods(labels.SelectorFromSet(map[string]string{k: v}))
		if err != nil {
			return nil, err
		}

		for _, pod := range pods {
			podSet[pod.ObjectMeta.Namespace+pod.ObjectMeta.Name] = pod
		}
	}
	return podSet, nil
}

// Returns the number of shared labels between two label sets.
func commonLabelsCount(labels, others map[string]string) int {
	var count int
	for k, v := range labels {
		if other, ok := others[k]; ok && v == other {
			count++
		}
	}
	return count
}
