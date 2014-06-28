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

package registry

import (
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

type MockPodRegistry struct {
	err  error
	pod  *api.Pod
	pods []api.Pod
	sync.Mutex
}

func MakeMockPodRegistry(pods []api.Pod) *MockPodRegistry {
	return &MockPodRegistry{
		pods: pods,
	}
}

func (registry *MockPodRegistry) ListPods(selector labels.Selector) ([]api.Pod, error) {
	registry.Lock()
	defer registry.Unlock()
	if registry.err != nil {
		return registry.pods, registry.err
	}
	var filtered []api.Pod
	for _, pod := range registry.pods {
		if selector.Matches(labels.Set(pod.Labels)) {
			filtered = append(filtered, pod)
		}
	}
	return filtered, nil
}

func (registry *MockPodRegistry) GetPod(podId string) (*api.Pod, error) {
	registry.Lock()
	defer registry.Unlock()
	return registry.pod, registry.err
}

func (registry *MockPodRegistry) CreatePod(machine string, pod api.Pod) error {
	registry.Lock()
	defer registry.Unlock()
	return registry.err
}

func (registry *MockPodRegistry) UpdatePod(pod api.Pod) error {
	registry.Lock()
	defer registry.Unlock()
	registry.pod = &pod
	return registry.err
}

func (registry *MockPodRegistry) DeletePod(podId string) error {
	registry.Lock()
	defer registry.Unlock()
	return registry.err
}
