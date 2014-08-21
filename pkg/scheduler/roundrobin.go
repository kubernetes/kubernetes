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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// RoundRobinScheduler chooses machines in order.
type RoundRobinScheduler struct {
	currentIndex int
}

func NewRoundRobinScheduler() Scheduler {
	return &RoundRobinScheduler{
		currentIndex: -1,
	}
}

// Schedule schedules a pod on the machine next to the last scheduled machine.
func (s *RoundRobinScheduler) Schedule(pod api.Pod, minionLister MinionLister) (string, error) {
	machines, err := minionLister.List()
	if err != nil {
		return "", err
	}
	s.currentIndex = (s.currentIndex + 1) % len(machines)
	result := machines[s.currentIndex]
	return result, nil
}
