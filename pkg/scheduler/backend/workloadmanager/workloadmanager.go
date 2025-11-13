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

package workloadmanager

import (
	"fmt"
	"sync"

	v1 "k8s.io/api/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

// WorkloadManager is the central source of truth for the state of pods belonging to Workload objects.
// It is designed to be driven explicitly by the scheduler's event handlers to ensure thread safety
// and avoid race conditions with the main scheduling queue.
// Note: The current implementation assumes that pod.Spec.Workload is immutable.
// Allowing mutability would require changes to the manager, e.g., by properly handling pod updates.
type WorkloadManager interface {
	fwk.WorkloadManager

	// AddPod is called by the scheduler when a Pod/Add event is observed.
	AddPod(pod *v1.Pod)
	// UpdatePod is called by the scheduler when a Pod/Update event is observed.
	UpdatePod(oldPod, newPod *v1.Pod)
	// DeletePod is called by the scheduler when a Pod/Delete event is observed.
	DeletePod(pod *v1.Pod)
}

// workloadManager is the concrete implementation of the WorkloadManager.
type workloadManager struct {
	lock sync.RWMutex

	// podGroupInfos stores the runtime state for each known pod group.
	podGroupInfos map[podGroupKey]*podGroupInfo
}

// New initializes a new workload manager and returns it.
func New() *workloadManager {
	return &workloadManager{
		podGroupInfos: make(map[podGroupKey]*podGroupInfo),
	}
}

// AddPod adds a pod to the workload manager if it has a workload reference.
// Pod is added to the available pods set for its corresponding pod group.
func (wm *workloadManager) AddPod(pod *v1.Pod) {
	if pod.Spec.WorkloadRef == nil {
		return
	}
	wm.lock.Lock()
	defer wm.lock.Unlock()

	key := newPodGroupKey(pod.Namespace, pod.Spec.WorkloadRef)
	info, ok := wm.podGroupInfos[key]
	if !ok {
		info = newPodGroupInfo()
		wm.podGroupInfos[key] = info
	}
	info.addPod(pod)
}

// UpdatePod updates a pod in the workload manager if it has a workload reference.
// Note: The current implementation assumes that newPod.Spec.Workload is immutable.
func (wm *workloadManager) UpdatePod(oldPod, newPod *v1.Pod) {
	if newPod.Spec.WorkloadRef == nil {
		return
	}
	wm.lock.Lock()
	defer wm.lock.Unlock()

	key := newPodGroupKey(newPod.Namespace, newPod.Spec.WorkloadRef)
	info, ok := wm.podGroupInfos[key]
	if !ok {
		// Shouldn't happen, but handling this case gracefully.
		info = newPodGroupInfo()
		wm.podGroupInfos[key] = info
		info.addPod(newPod)
		return
	}
	info.updatePod(oldPod, newPod)
}

// DeletePod removes a pod from the workload manager if it has a workload reference.
// Pod is removed from the pods sets for its corresponding pod group.
func (wm *workloadManager) DeletePod(pod *v1.Pod) {
	if pod.Spec.WorkloadRef == nil {
		return
	}
	wm.lock.Lock()
	defer wm.lock.Unlock()

	key := newPodGroupKey(pod.Namespace, pod.Spec.WorkloadRef)
	info, ok := wm.podGroupInfos[key]
	if !ok {
		// The pod group may have already been cleaned up, or the pod was never added.
		return
	}
	info.deletePod(pod.UID)
	// Clean up the map entry if no pods are left in the group.
	if info.empty() {
		delete(wm.podGroupInfos, key)
	}
}

// PodGroupInfo returns the state of a pod group.
func (wm *workloadManager) PodGroupInfo(namespace string, workloadRef *v1.WorkloadReference) (fwk.PodGroupInfo, error) {
	wm.lock.RLock()
	defer wm.lock.RUnlock()

	state, ok := wm.podGroupInfos[newPodGroupKey(namespace, workloadRef)]
	if !ok {
		return nil, fmt.Errorf("internal pod group state doesn't exist for a pod's workload")
	}
	return state, nil
}
