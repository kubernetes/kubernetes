/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	fwk "k8s.io/kube-scheduler/framework"
)

// waitingPodsMap a thread-safe map used to maintain pods waiting in the permit phase.
type waitingPodsMap struct {
	pods map[types.UID]*waitingPod
	mu   sync.RWMutex
}

// NewWaitingPodsMap returns a new waitingPodsMap.
func NewWaitingPodsMap() *waitingPodsMap {
	return &waitingPodsMap{
		pods: make(map[types.UID]*waitingPod),
	}
}

// add a new WaitingPod to the map.
func (m *waitingPodsMap) add(wp *waitingPod) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pods[wp.GetPod().UID] = wp
}

// remove a WaitingPod from the map.
func (m *waitingPodsMap) remove(uid types.UID) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.pods, uid)
}

// get a WaitingPod from the map.
func (m *waitingPodsMap) get(uid types.UID) *waitingPod {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.pods[uid]
}

// iterate acquires a read lock and iterates over the WaitingPods map.
func (m *waitingPodsMap) iterate(callback func(fwk.WaitingPod)) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, v := range m.pods {
		callback(v)
	}
}

// waitingPod represents a pod waiting in the permit phase.
type waitingPod struct {
	pod            *v1.Pod
	pendingPlugins map[string]*time.Timer
	s              chan *fwk.Status
	mu             sync.RWMutex
}

var _ fwk.WaitingPod = &waitingPod{}

// newWaitingPod returns a new waitingPod instance.
func newWaitingPod(pod *v1.Pod, pluginsMaxWaitTime map[string]time.Duration) *waitingPod {
	wp := &waitingPod{
		pod: pod,
		// Allow() and Reject() calls are non-blocking. This property is guaranteed
		// by using non-blocking send to this channel. This channel has a buffer of size 1
		// to ensure that non-blocking send will not be ignored - possible situation when
		// receiving from this channel happens after non-blocking send.
		s: make(chan *fwk.Status, 1),
	}

	wp.pendingPlugins = make(map[string]*time.Timer, len(pluginsMaxWaitTime))
	// The time.AfterFunc calls wp.Reject which iterates through pendingPlugins map. Acquire the
	// lock here so that time.AfterFunc can only execute after newWaitingPod finishes.
	wp.mu.Lock()
	defer wp.mu.Unlock()
	for k, v := range pluginsMaxWaitTime {
		plugin, waitTime := k, v
		wp.pendingPlugins[plugin] = time.AfterFunc(waitTime, func() {
			msg := fmt.Sprintf("rejected due to timeout after waiting %v at plugin %v",
				waitTime, plugin)
			wp.Reject(plugin, msg)
		})
	}

	return wp
}

// GetPod returns a reference to the waiting pod.
func (w *waitingPod) GetPod() *v1.Pod {
	return w.pod
}

// GetPendingPlugins returns a list of pending permit plugin's name.
func (w *waitingPod) GetPendingPlugins() []string {
	w.mu.RLock()
	defer w.mu.RUnlock()
	plugins := make([]string, 0, len(w.pendingPlugins))
	for p := range w.pendingPlugins {
		plugins = append(plugins, p)
	}

	return plugins
}

// Allow declares the waiting pod is allowed to be scheduled by plugin pluginName.
// If this is the last remaining plugin to allow, then a success signal is delivered
// to unblock the pod.
func (w *waitingPod) Allow(pluginName string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if timer, exist := w.pendingPlugins[pluginName]; exist {
		timer.Stop()
		delete(w.pendingPlugins, pluginName)
	}

	// Only signal success status after all plugins have allowed
	if len(w.pendingPlugins) != 0 {
		return
	}

	// The select clause works as a non-blocking send.
	// If there is no receiver, it's a no-op (default case).
	select {
	case w.s <- fwk.NewStatus(fwk.Success, ""):
	default:
	}
}

// Reject declares the waiting pod unschedulable.
func (w *waitingPod) Reject(pluginName, msg string) {
	w.mu.RLock()
	defer w.mu.RUnlock()
	for _, timer := range w.pendingPlugins {
		timer.Stop()
	}

	// The select clause works as a non-blocking send.
	// If there is no receiver, it's a no-op (default case).
	select {
	case w.s <- fwk.NewStatus(fwk.Unschedulable, msg).WithPlugin(pluginName):
	default:
	}
}
