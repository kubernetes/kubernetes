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
	"errors"
	"sync"

	"k8s.io/apimachinery/pkg/types"
	fwk "k8s.io/kube-scheduler/framework"
)

// podsInPreBindMap is a thread-safe map of pods currently in the preBind phase.
type podsInPreBindMap struct {
	pods map[types.UID]*podInPreBind
	mu   sync.RWMutex
}

// NewPodsInPreBindMap creates an empty podsInPreBindMap.
func NewPodsInPreBindMap() *podsInPreBindMap {
	return &podsInPreBindMap{
		pods: make(map[types.UID]*podInPreBind),
	}
}

// get returns a pod from the map if it exists.
func (pbm *podsInPreBindMap) get(uid types.UID) *podInPreBind {
	pbm.mu.RLock()
	defer pbm.mu.RUnlock()
	return pbm.pods[uid]
}

// add adds a pod to map, overwriting existing one.
func (pbm *podsInPreBindMap) add(uid types.UID, cancel context.CancelCauseFunc) {
	pbm.mu.Lock()
	defer pbm.mu.Unlock()
	pbm.pods[uid] = &podInPreBind{cancel: cancel}
}

// remove removes a pod from the map.
func (pbm *podsInPreBindMap) remove(uid types.UID) {
	pbm.mu.Lock()
	defer pbm.mu.Unlock()
	delete(pbm.pods, uid)
}

var _ fwk.PodInPreBind = &podInPreBind{}

// podInPreBind describes a pod in the preBind phase, before the bind was called for a pod.
type podInPreBind struct {
	finished bool
	canceled bool
	cancel   context.CancelCauseFunc
	mu       sync.Mutex
}

// CancelPod cancels the context running the preBind phase
// for a given pod.
func (bp *podInPreBind) CancelPod(message string) bool {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	if bp.finished {
		return false
	}
	if !bp.canceled {
		bp.cancel(errors.New(message))
	}
	bp.canceled = true
	return true
}

// MarkPrebound marks the pod as finished with preBind phase
// of binding cycle, making it impossible to cancel
// the binding cycle for it.
func (bp *podInPreBind) MarkPrebound() bool {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	if bp.canceled {
		return false
	}
	bp.finished = true
	return true
}
