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
	pods, err := podLister.ListPods(labels.SelectorFromSet(pod.Labels))
	if err != nil {
		return nil, err
	}
	minions, err := minionLister.List()
	if err != nil {
		return nil, err
	}

	counts := map[string]int{}
	for _, pod := range pods {
		counts[pod.CurrentState.Host]++
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
