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
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/types"
)

// podsInPrebindMap is a map of pods currently in the prebind phase.
type podsInPrebindMap struct {
	pods map[types.UID]*podInPrebind
	mu   sync.RWMutex
}

// NewPodsInPrebindMap creates an empty podsInPrebindMap.
func NewPodsInPrebindMap() *podsInPrebindMap {
	return &podsInPrebindMap{
		pods: make(map[types.UID]*podInPrebind),
	}
}

// get returns a pod from the map if it exists.
func (bpm *podsInPrebindMap) get(uid types.UID) *podInPrebind {
	bpm.mu.RLock()
	defer bpm.mu.RUnlock()
	return bpm.pods[uid]
}

// add adds a pod to map, overwriting existing one
func (bpm *podsInPrebindMap) add(uid types.UID, cancel context.CancelCauseFunc) {
	bpm.mu.Lock()
	defer bpm.mu.Unlock()
	bpm.pods[uid] = &podInPrebind{cancel: cancel}
}

// remove removes a pod from the map.
func (bpm *podsInPrebindMap) remove(uid types.UID) {
	bpm.mu.Lock()
	defer bpm.mu.Unlock()
	delete(bpm.pods, uid)
}

// podInPrebind describes a pod in the prebind phase, before the bind was called for a pod.
type podInPrebind struct {
	finished bool
	canceled bool
	cancel   context.CancelCauseFunc
	mu       sync.Mutex
}

// CancelPod cancels the context running the prebind phase
// for a given pod.
func (bp *podInPrebind) CancelPod(message string) bool {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	if bp.finished {
		return false
	}
	if !bp.canceled {
		bp.cancel(fmt.Errorf("%s", message))
	}
	bp.canceled = true
	return true
}

// MarkPrebound marks the pod as finished with prebind phase
// of binding cycle, making it impossible to cancel
// the binding cycle for it.
func (bp *podInPrebind) MarkPrebound() bool {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	if bp.canceled {
		return false
	}
	bp.finished = true
	return true
}
