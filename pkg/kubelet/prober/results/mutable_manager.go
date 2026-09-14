/*
Copyright The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// Manager implementation.
type mutableManager struct {
	// Protects publications and their pending notifications together so a result cannot be
	// committed without registering a notification.
	sync.RWMutex
	cache        map[kubecontainer.ContainerID]Update
	pending      map[kubecontainer.ContainerID]Update
	version      uint64
	dispatching  bool
	dispatchWake chan struct{}
	updates      chan Update
}

var _ Manager = &mutableManager{}

// newMutableManager coalesces notifications without blocking probe result commits.
func newMutableManager() Manager {
	return &mutableManager{
		cache:        make(map[kubecontainer.ContainerID]Update),
		pending:      make(map[kubecontainer.ContainerID]Update),
		dispatchWake: make(chan struct{}, 1),
		updates:      make(chan Update, 20),
	}
}

func (m *mutableManager) Get(id kubecontainer.ContainerID) (Result, bool) {
	m.RLock()
	defer m.RUnlock()

	entry, found := m.cache[id]
	return entry.Result, found
}

func (m *mutableManager) Set(id kubecontainer.ContainerID, result Result, pod *v1.Pod) {
	m.SetWithMetadata(id, result, pod, Metadata{})
}

func (m *mutableManager) SetWithMetadata(id kubecontainer.ContainerID, result Result, pod *v1.Pod, metadata Metadata) {
	m.Lock()
	defer m.Unlock()

	update := Update{
		ContainerID:   id,
		Result:        result,
		PodUID:        pod.UID,
		ContainerName: metadata.ContainerName,
		ProbeID:       metadata.ProbeID,
	}

	previous, exists := m.cache[id]
	previous.Version = 0
	if exists && previous == update {
		return
	}

	m.version++
	update.Version = m.version
	m.cache[id] = update
	m.pending[id] = update

	if !m.dispatching {
		m.dispatching = true
		go m.dispatchUpdates()
	} else {
		m.wakeDispatcher()
	}
}

func (m *mutableManager) IsCurrent(update Update) bool {
	m.RLock()
	defer m.RUnlock()

	current, exists := m.cache[update.ContainerID]
	return exists && current == update
}

// A dispatcher exists only while notifications remain pending. The wake channel lets removal
// release a dispatcher blocked by a full updates channel, and lets newer results replace its work.
func (m *mutableManager) dispatchUpdates() {
	for {
		m.Lock()
		if len(m.pending) == 0 {
			m.dispatching = false
			m.Unlock()
			return
		}

		var update Update
		for _, update = range m.pending {
			break
		}
		m.Unlock()

		select {
		case m.updates <- update:
			m.Lock()

			// A concurrent publication must retain its own notification even if this send
			// delivered an older result. IsCurrent rejects that older result at consumption.
			if m.pending[update.ContainerID] == update {
				delete(m.pending, update.ContainerID)
			}
			m.Unlock()
		case <-m.dispatchWake:
		}
	}
}

func (m *mutableManager) wakeDispatcher() {
	select {
	case m.dispatchWake <- struct{}{}:
	default:
	}
}

func (m *mutableManager) Remove(id kubecontainer.ContainerID) {
	m.Lock()
	defer m.Unlock()

	delete(m.cache, id)
	delete(m.pending, id)

	if m.dispatching {
		m.wakeDispatcher()
	}
}

func (m *mutableManager) Updates() <-chan Update {
	return m.updates
}
