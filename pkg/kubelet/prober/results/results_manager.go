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

package results

import (
	"sync"

	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
)

// Manager provides a probe results cache and channel of updates.
type Manager interface {
	// Get returns the cached result for the container with the given ID.
	Get(kubecontainer.ContainerID) (Result, bool)
	// Set sets the cached result for the container with the given ID.
	// The pod is only included to be sent with the update.
	Set(kubecontainer.ContainerID, Result, *v1.Pod)
	// Remove clears the cached result for the container with the given ID.
	Remove(kubecontainer.ContainerID)
	// Updates creates a channel that receives an Update whenever its result changes (but not
	// removed).
	// NOTE: The current implementation only supports a single updates channel.
	Updates() <-chan Update
}

// Result is the type for probe results.
type Result bool

const (
	Success Result = true
	Failure Result = false
)

func (r Result) String() string {
	switch r {
	case Success:
		return "Success"
	case Failure:
		return "Failure"
	default:
		return "UNKNOWN"
	}
}

// Update is an enum of the types of updates sent over the Updates channel.
type Update struct {
	ContainerID kubecontainer.ContainerID
	Result      Result
	PodUID      types.UID
}

// Manager implementation.
type manager struct {
	// guards the cache
	sync.RWMutex
	// map of container ID -> probe Result
	cache map[kubecontainer.ContainerID]Result
	// channel of updates
	updates chan Update
}

var _ Manager = &manager{}

// NewManager creates ane returns an empty results manager.
func NewManager() Manager {
	return &manager{
		cache:   make(map[kubecontainer.ContainerID]Result),
		updates: make(chan Update, 20),
	}
}

func (m *manager) Get(id kubecontainer.ContainerID) (Result, bool) {
	m.RLock()
	defer m.RUnlock()
	result, found := m.cache[id]
	return result, found
}

func (m *manager) Set(id kubecontainer.ContainerID, result Result, pod *v1.Pod) {
	if m.setInternal(id, result) {
		m.updates <- Update{id, result, pod.UID}
	}
}

// Internal helper for locked portion of set. Returns whether an update should be sent.
func (m *manager) setInternal(id kubecontainer.ContainerID, result Result) bool {
	m.Lock()
	defer m.Unlock()
	prev, exists := m.cache[id]
	if !exists || prev != result {
		m.cache[id] = result
		return true
	}
	return false
}

func (m *manager) Remove(id kubecontainer.ContainerID) {
	m.Lock()
	defer m.Unlock()
	delete(m.cache, id)
}

func (m *manager) Updates() <-chan Update {
	return m.updates
}
