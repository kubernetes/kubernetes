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

// MinionLister interface represents anything that can list minions for a scheduler.
type MinionLister interface {
	List() (list api.NodeList, err error)
}

// FakeMinionLister implements MinionLister on a []string for test purposes.
type FakeMinionLister api.NodeList

// List returns minions as a []string.
func (f FakeMinionLister) List() (api.NodeList, error) {
	return api.NodeList(f), nil
}

// PodLister interface represents anything that can list pods for a scheduler.
type PodLister interface {
	// TODO: make this exactly the same as client's Pods(ns).List() method, by returning a api.PodList
	List(labels.Selector) ([]api.Pod, error)
}

// FakePodLister implements PodLister on an []api.Pods for test purposes.
type FakePodLister []api.Pod

// List returns []api.Pod matching a query.
func (f FakePodLister) List(s labels.Selector) (selected []api.Pod, err error) {
	for _, pod := range f {
		if s.Matches(labels.Set(pod.Labels)) {
			selected = append(selected, pod)
		}
	}
	return selected, nil
}
