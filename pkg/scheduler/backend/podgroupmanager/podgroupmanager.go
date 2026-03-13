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

package podgroupmanager

import (
	"fmt"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
)

// PodGroupManager is the central source of truth for the state of pods belonging to PodGroup objects.
// It is designed to be driven explicitly by the scheduler's event handlers to ensure thread safety
// and avoid race conditions with the main scheduling queue.
// Note: The current implementation assumes that pod.Spec.SchedulingGroup is immutable.
// Allowing mutability would require changes to the manager, e.g., by properly handling pod updates.
type PodGroupManager interface {
	fwk.PodGroupManager

	// AddPod is called by the scheduler when a Pod/Add event is observed.
	AddPod(pod *v1.Pod)
	// UpdatePod is called by the scheduler when a Pod/Update event is observed.
	UpdatePod(oldPod, newPod *v1.Pod)
	// DeletePod is called by the scheduler when a Pod/Delete event is observed.
	DeletePod(pod *v1.Pod)
}

// podGroupManager is the concrete implementation of the PodGroupManager.
type podGroupManager struct {
	lock sync.RWMutex

	// podGroupStates stores the runtime state for each known pod group.
	podGroupStates map[podGroupKey]*podGroupState

	logger klog.Logger
}

// New initializes a new pod group manager and returns it.
func New(logger klog.Logger) *podGroupManager {
	return &podGroupManager{
		podGroupStates: make(map[podGroupKey]*podGroupState),
		logger:         logger,
	}
}

// AddPod adds a pod to the pod group manager if it has a scheduling group.
// Pod is added to the available pods set for its corresponding pod group.
func (pgm *podGroupManager) AddPod(pod *v1.Pod) {
	if pod.Spec.SchedulingGroup == nil {
		return
	}
	pgm.lock.Lock()
	defer pgm.lock.Unlock()

	key := newPodGroupKey(pod.Namespace, pod.Spec.SchedulingGroup)
	state, ok := pgm.podGroupStates[key]
	if !ok {
		state = newPodGroupState()
		pgm.podGroupStates[key] = state
	}
	state.addPod(pod)
}

// UpdatePod updates a pod in the pod group manager if it has a scheduling group.
// Note: The current implementation assumes that newPod.Spec.SchedulingGroup is immutable.
func (pgm *podGroupManager) UpdatePod(oldPod, newPod *v1.Pod) {
	if newPod.Spec.SchedulingGroup == nil {
		return
	}
	pgm.lock.Lock()
	defer pgm.lock.Unlock()

	key := newPodGroupKey(newPod.Namespace, newPod.Spec.SchedulingGroup)
	state, ok := pgm.podGroupStates[key]
	if !ok {
		// Shouldn't happen, but handling this case gracefully.
		state = newPodGroupState()
		pgm.podGroupStates[key] = state
		state.addPod(newPod)
		pgm.logger.Error(nil, "UpdatePod found no existing PodGroup for pod. Created new PodGroup for the pod", "pod", klog.KObj(newPod), "podGroupKey", klog.KObj(key))
		return
	}
	state.updatePod(oldPod, newPod)
}

// DeletePod removes a pod from the pod group manager if it has a scheduling group.
// Pod is removed from the pods sets for its corresponding pod group.
func (pgm *podGroupManager) DeletePod(pod *v1.Pod) {
	if pod.Spec.SchedulingGroup == nil {
		return
	}
	pgm.lock.Lock()
	defer pgm.lock.Unlock()

	key := newPodGroupKey(pod.Namespace, pod.Spec.SchedulingGroup)
	state, ok := pgm.podGroupStates[key]
	if !ok {
		// The pod group may have already been cleaned up, or the pod was never added.
		return
	}
	state.deletePod(pod.UID)
	// Clean up the map entry if no pods are left in the group.
	if state.empty() {
		delete(pgm.podGroupStates, key)
	}
}

// PodGroupState returns the runtime state of a pod group.
func (pgm *podGroupManager) PodGroupState(namespace string, schedulingGroup *v1.PodSchedulingGroup) (fwk.PodGroupState, error) {
	pgm.lock.RLock()
	defer pgm.lock.RUnlock()

	state, ok := pgm.podGroupStates[newPodGroupKey(namespace, schedulingGroup)]
	if !ok {
		return nil, fmt.Errorf("internal pod group state doesn't exist for a pod's scheduling group")
	}
	return state, nil
}
