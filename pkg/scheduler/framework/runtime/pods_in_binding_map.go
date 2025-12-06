/*
Copyright 2025 The Kubernetes Authors.

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

package runtime

import (
	"context"
	"sync"

	"k8s.io/apimachinery/pkg/types"
)

// podsInBindingMap is a map of pods currently in the binding cycle.
type podsInBindingMap struct {
	pods map[types.UID]*podInBinding
	mu   sync.RWMutex
}

// NewBindingPodsMap creates an empty podsInBindingMap.
func NewBindingPodsMap() *podsInBindingMap {
	return &podsInBindingMap{
		pods: make(map[types.UID]*podInBinding),
	}
}

// get returns a pod from the map if it exists.
func (bpm *podsInBindingMap) get(uid types.UID) *podInBinding {
	bpm.mu.RLock()
	defer bpm.mu.RUnlock()
	return bpm.pods[uid]
}

// add adds a pod to map, overwriting existing one
func (bpm *podsInBindingMap) add(uid types.UID, cancel context.CancelFunc) {
	bpm.mu.Lock()
	defer bpm.mu.Unlock()
	bpm.pods[uid] = &podInBinding{uid: uid, cancel: cancel}
}

// remove removes a pod from the map.
func (bpm *podsInBindingMap) remove(uid types.UID) {
	bpm.mu.Lock()
	defer bpm.mu.Unlock()
	delete(bpm.pods, uid)
}

// podInBinding describes a pod in the binding cycle of the
// scheduler cycle, but before the bind was called for a pod.
type podInBinding struct {
	uid           types.UID
	inBindedPhase bool
	canceled      bool
	cancel        context.CancelFunc
	mu            sync.Mutex
}

// CancelPod cancels the context running the binding cycle
// for a given pod.
func (bp *podInBinding) CancelPod() bool {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	if bp.inBindedPhase {
		return false
	}
	if !bp.canceled {
		bp.cancel()
	}
	bp.canceled = true
	return true
}

// MarkBinded marks the pod as being in the `bind` part
// of binding cycle, making it impossible to cancel
// the binding cycle for it.
func (bp *podInBinding) MarkBinded() bool {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	if bp.canceled {
		return false
	}
	bp.inBindedPhase = true
	return true
}
