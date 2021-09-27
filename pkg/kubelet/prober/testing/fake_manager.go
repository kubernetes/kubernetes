/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"sync"
)

// FakeManager simulates a prober.Manager for testing.
type FakeManager struct {
	// Map of active workers for probes
	workers map[types.UID]struct{}
	// Lock for accessing & mutating workers
	workerLock sync.RWMutex
}

func NewFakeManager() *FakeManager {
	return &FakeManager{
		workers: make(map[types.UID]struct{}),
	}
}

// Unused methods below.

// AddPod simulates adding a Pod.
func (m *FakeManager) AddPod(pod *v1.Pod) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()
	m.workers[pod.UID] = struct{}{}

}

// RemovePod simulates removing a Pod.
func (m *FakeManager) RemovePod(pod *v1.Pod, keepReadinessProbe bool) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()
	delete(m.workers, pod.UID)
}

// CleanupPods simulates cleaning up Pods.
func (m *FakeManager) CleanupPods(desiredPods map[types.UID]sets.Empty) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()
	for key := range m.workers {
		if _, ok := desiredPods[key]; !ok {
			delete(m.workers, key)
		}
	}
}

// Start simulates start syncing the probe status
func (m *FakeManager) Start() {}

// UpdatePodStatus simulates updating the Pod Status.
func (m *FakeManager) UpdatePodStatus(_ types.UID, podStatus *v1.PodStatus) {
	for i := range podStatus.ContainerStatuses {
		podStatus.ContainerStatuses[i].Ready = true
	}
}

// Cleanup simulates cleaning all workers.
func (m *FakeManager) Cleanup() {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()
	m.workers = make(map[types.UID]struct{})
}

// IsProbeStarted simulates checking probes exists.
func (m *FakeManager) IsProbeStarted(pod *v1.Pod) bool {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()
	_, ok := m.workers[pod.UID]
	return ok
}
