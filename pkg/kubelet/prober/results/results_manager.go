/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// Manager provides a probe results cache and channel of updates.
type Manager interface {
	// Get returns the cached result for the container with the given ID.
	Get(id kubecontainer.ContainerID) (Result, bool)
	// Set sets the cached result for the container with the given ID.
	Set(id kubecontainer.ContainerID, result Result)
	// Remove clears the cached result for the container with the given ID.
	Remove(id kubecontainer.ContainerID)
	// Updates creates a channel that receives an Update whenever its result changes (but not
	// removed). If the channel is full, updates will be dropped.
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
}

// Manager implementation.
type manager struct {
	// guards the cache
	sync.RWMutex
	// map of container ID -> probe Result
	cache map[kubecontainer.ContainerID]Result
	// channel of updates (may be nil)
	updates chan Update
}

var _ Manager = &manager{}

// NewManager creates ane returns an empty results manager.
func NewManager() Manager {
	return &manager{cache: make(map[kubecontainer.ContainerID]Result)}
}

func (m *manager) Get(id kubecontainer.ContainerID) (Result, bool) {
	m.RLock()
	defer m.RUnlock()
	result, found := m.cache[id]
	return result, found
}

func (m *manager) Set(id kubecontainer.ContainerID, result Result) {
	m.Lock()
	defer m.Unlock()
	prev, exists := m.cache[id]
	if !exists || prev != result {
		m.cache[id] = result
		m.pushUpdate(Update{id, result})
	}
}

func (m *manager) Remove(id kubecontainer.ContainerID) {
	m.Lock()
	defer m.Unlock()
	delete(m.cache, id)
}

func (m *manager) Updates() <-chan Update {
	if m.updates == nil {
		m.updates = make(chan Update)
	}
	return m.updates
}

// pushUpdates sends (non-blocking) an update on the updates channel if it is initialized.
func (m *manager) pushUpdate(update Update) {
	if m.updates != nil {
		select {
		case m.updates <- update:
		default:
			// Don't block
		}
	}
}
