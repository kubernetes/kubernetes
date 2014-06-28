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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// Anything that can list minions for a scheduler.
type MinionLister interface {
	List() (machines []string, err error)
}

// Make a MinionLister from a []string
type FakeMinionLister []string

// Returns minions as a []string
func (f FakeMinionLister) List() ([]string, error) {
	return []string(f), nil
}

// Anything that can list pods for a scheduler
type PodLister interface {
	ListPods(labels.Selector) ([]api.Pod, error)
}

// Make a MinionLister from an []api.Pods
type FakePodLister []api.Pod

func (f FakePodLister) ListPods(s labels.Selector) (selected []api.Pod, err error) {
	for _, pod := range f {
		if s.Matches(labels.Set(pod.Labels)) {
			selected = append(selected, pod)
		}
	}
	return selected, nil
}
