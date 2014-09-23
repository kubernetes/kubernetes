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
	"fmt"
	"math/rand"
	"sort"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

type SpreadingScheduler struct {
	pods       PodLister
	random     *rand.Rand
	randomLock sync.Mutex
}

// CalculateSpreadPriority spreads pods by minimizing the number of pods on the same machine with the same labels.
// Importantly, if there are services in the system that span multiple heterogenous sets of pods, this spreading priority
// may not provide optimal spreading for the members of that Service.
// TODO: consider if we want to include Service label sets in the scheduling priority.
func CalculateSpreadPriority(selector labels.Selector, podLister PodLister, minionLister MinionLister) (HostPriorityList, error) {
	pods, err := podLister.ListPods(selector)
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
	for _, minion := range minions {
		result = append(result, HostPriority{host: minion, score: counts[minion]})
	}
	return result, nil
}

// Schedule schedules pods to maximize spreading of identical pods across multiple hosts.
// Does not currently take hostPort scheduling into account.
// TODO: combine priority based and fit based schedulers into a single scheduler.
func (s *SpreadingScheduler) Schedule(pod api.Pod, minionLister MinionLister) (string, error) {
	priorities, err := CalculateSpreadPriority(labels.SelectorFromSet(pod.Labels), s.pods, minionLister)
	if err != nil {
		return "", err
	}
	sort.Sort(priorities)
	if len(priorities) == 0 {
		return "", fmt.Errorf("failed to find a fit: %v", pod)
	}
	return priorities[0].host, nil
}
