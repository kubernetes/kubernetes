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

package allocation

import (
	"path/filepath"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
	"k8s.io/kubernetes/pkg/kubelet/status"
)

// podStatusManagerStateFile is the file name where status manager stores its state
const (
	allocatedPodsStateFile = "allocated_pods_state"
	actuatedPodsStateFile  = "actuated_pods_state"
)

// AllocationManager tracks pod resource allocations.
type Manager interface {
	// GetContainerResourceAllocation returns the AllocatedResources value for the container
	GetContainerResourceAllocation(podUID types.UID, containerName string) (v1.ResourceRequirements, bool)

	// UpdatePodFromAllocation overwrites the pod spec with the allocation.
	// This function does a deep copy only if updates are needed.
	// Returns the updated (or original) pod, and whether there was an allocation stored.
	UpdatePodFromAllocation(pod *v1.Pod) (*v1.Pod, bool)

	// SetAllocatedResources checkpoints the resources allocated to a pod's containers.
	SetAllocatedResources(allocatedPod *v1.Pod) error

	// SetActuatedResources records the actuated resources of the given container (or the entire
	// pod, if actuatedContainer is nil).
	SetActuatedResources(allocatedPod *v1.Pod, actuatedContainer *v1.Container) error

	// GetActuatedResources returns the stored actuated resources for the container, and whether they exist.
	GetActuatedResources(podUID types.UID, containerName string) (v1.ResourceRequirements, bool)

	// RemovePod removes any stored state for the given pod UID.
	RemovePod(uid types.UID)

	// RemoveOrphanedPods removes the stored state for any pods not included in the set of remaining pods.
	RemoveOrphanedPods(remainingPods sets.Set[types.UID])

	// Run starts the allocation manager. This is currently only used to handle periodic retry of
	// pending resizes.
	Run()

	// PushPendingResize queues a pod with a pending resize request for later reevaluation.
	// If a change in the resize queue is detected, pending resizes will be retried.
	PushPendingResize(pod *v1.Pod)

	// RetryPendingResizes signals the allocation manager to retry all pending resize requests.
	RetryPendingResizes()

	// HandlePodResourcesResize determines if a pod resize is feasible. If so,
	// it updates the allocation and sets the pod resize conditions accordingly.
	HandlePodResourcesResize(pod *v1.Pod) error
}

type manager struct {
	allocated state.State
	actuated  state.State

	canResizePod   func(pod *v1.Pod) (bool, string, string)
	triggerPodSync func(pod *v1.Pod)

	statusManager       status.Manager
	skipPeriodicRetries bool
	pendingResizeChan   chan struct{}

	pendingResizesLock     sync.RWMutex
	podsWithPendingResizes []*v1.Pod

	podResizeMutex *sync.Mutex
}

func NewManager(
	checkpointDirectory string,
	statusManager status.Manager,
	podResizeMutex *sync.Mutex,
	canResizePod func(pod *v1.Pod) (bool, string, string),
	triggerPodSync func(pod *v1.Pod)) Manager {

	return newManager(
		newStateImpl(checkpointDirectory, allocatedPodsStateFile),
		newStateImpl(checkpointDirectory, allocatedPodsStateFile),
		statusManager,
		podResizeMutex,
		canResizePod,
		triggerPodSync,
		false,
	)
}

func newStateImpl(checkpointDirectory, checkpointName string) state.State {
	if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		return state.NewNoopStateCheckpoint()
	}

	stateImpl, err := state.NewStateCheckpoint(checkpointDirectory, checkpointName)
	if err != nil {
		// This is a critical, non-recoverable failure.
		klog.ErrorS(err, "Failed to initialize allocation checkpoint manager",
			"checkpointPath", filepath.Join(checkpointDirectory, checkpointName))
		panic(err)
	}

	return stateImpl
}

// NewInMemoryManager returns an allocation manager that doesn't persist state.
// For testing purposes only!
func NewInMemoryManager(
	statusManager status.Manager,
	podResizeMutex *sync.Mutex,
	canResizePod func(pod *v1.Pod) (bool, string, string),
	triggerPodSync func(pod *v1.Pod)) Manager {
	return newManager(
		state.NewStateMemory(nil),
		state.NewStateMemory(nil),
		statusManager,
		podResizeMutex,
		canResizePod,
		triggerPodSync,
		true,
	)
}

func newManager(allocated, actuated state.State,
	statusManager status.Manager,
	podResizeMutex *sync.Mutex,
	canResizePod func(pod *v1.Pod) (bool, string, string),
	triggerPodSync func(pod *v1.Pod),
	skipPeriodicRetries bool) Manager {

	m := &manager{
		allocated:           allocated,
		actuated:            actuated,
		canResizePod:        canResizePod,
		triggerPodSync:      triggerPodSync,
		skipPeriodicRetries: skipPeriodicRetries,
		statusManager:       statusManager,
		podResizeMutex:      podResizeMutex,
		pendingResizeChan:   make(chan struct{}, 1),
	}

	return m
}

func (m *manager) Run() {
	// Returns a list of successful retried pods.
	retryPendingResizes := func() []*v1.Pod {
		m.pendingResizesLock.Lock()
		defer m.pendingResizesLock.Unlock()

		var newPendingResizes []*v1.Pod
		var successfulResizes []*v1.Pod

		// Retry all pending resizes.
		for _, pod := range m.podsWithPendingResizes {
			oldResizeStatus := m.statusManager.GetPodResizeConditions(pod.UID)

			err := m.HandlePodResourcesResize(pod)
			if err != nil {
				klog.ErrorS(err, "Failed to handle pod resources resize", "pod", klog.KObj(pod))
				newPendingResizes = append(newPendingResizes, pod)
			}

			switch {
			case m.statusManager.IsPodResizeDeferred(pod.UID):
				klog.V(4).InfoS("Pod resize is still deferred after retry", "pod", klog.KObj(pod))
				newPendingResizes = append(newPendingResizes, pod)
			case m.statusManager.IsPodResizeInfeasible(pod.UID):
				klog.V(4).InfoS("Pod resize is infeasible after retry", "pod", klog.KObj(pod))
			default:
				klog.V(4).InfoS("Pod resize successfully allocated", "pod", klog.KObj(pod))
				successfulResizes = append(successfulResizes, pod)
			}

			// If the pod resize status has changed, we need to update the pod status.
			newResizeStatus := m.statusManager.GetPodResizeConditions(pod.UID)
			if !apiequality.Semantic.DeepEqual(oldResizeStatus, newResizeStatus) {
				m.triggerPodSync(pod)
			}
		}

		m.podsWithPendingResizes = newPendingResizes
		return successfulResizes
	}

	// Start a goroutine to periodically check for pending resizes and process them if needed.
	// We retry all pending resizes every 3 minutes or when explicitly signaled.
	ticker := time.NewTicker(3 * time.Minute)
	go func() {
		for {
			select {
			case <-ticker.C:
				if !m.skipPeriodicRetries {
					successfulResizes := retryPendingResizes()
					for _, po := range successfulResizes {
						klog.InfoS("Successfully retried resize after timeout", "pod", klog.KObj(po))
					}
				}
			case <-m.pendingResizeChan:
				retryPendingResizes()
			}
		}
	}()

}

// GetContainerResourceAllocation returns the last checkpointed AllocatedResources values
// If checkpoint manager has not been initialized, it returns nil, false
func (m *manager) GetContainerResourceAllocation(podUID types.UID, containerName string) (v1.ResourceRequirements, bool) {
	return m.allocated.GetContainerResources(podUID, containerName)
}

// UpdatePodFromAllocation overwrites the pod spec with the allocation.
// This function does a deep copy only if updates are needed.
func (m *manager) UpdatePodFromAllocation(pod *v1.Pod) (*v1.Pod, bool) {
	// TODO(tallclair): This clones the whole cache, but we only need 1 pod.
	allocs := m.allocated.GetPodResourceInfoMap()
	return updatePodFromAllocation(pod, allocs)
}

func updatePodFromAllocation(pod *v1.Pod, allocs state.PodResourceInfoMap) (*v1.Pod, bool) {
	allocated, found := allocs[pod.UID]
	if !found {
		return pod, false
	}

	updated := false
	containerAlloc := func(c v1.Container) (v1.ResourceRequirements, bool) {
		if cAlloc, ok := allocated.ContainerResources[c.Name]; ok {
			if !apiequality.Semantic.DeepEqual(c.Resources, cAlloc) {
				// Allocation differs from pod spec, retrieve the allocation
				if !updated {
					// If this is the first update to be performed, copy the pod
					pod = pod.DeepCopy()
					updated = true
				}
				return cAlloc, true
			}
		}
		return v1.ResourceRequirements{}, false
	}

	for i, c := range pod.Spec.Containers {
		if cAlloc, found := containerAlloc(c); found {
			// Allocation differs from pod spec, update
			pod.Spec.Containers[i].Resources = cAlloc
		}
	}
	for i, c := range pod.Spec.InitContainers {
		if cAlloc, found := containerAlloc(c); found {
			// Allocation differs from pod spec, update
			pod.Spec.InitContainers[i].Resources = cAlloc
		}
	}
	return pod, updated
}

// SetAllocatedResources checkpoints the resources allocated to a pod's containers
func (m *manager) SetAllocatedResources(pod *v1.Pod) error {
	return m.allocated.SetPodResourceInfo(pod.UID, allocationFromPod(pod))
}

func allocationFromPod(pod *v1.Pod) state.PodResourceInfo {
	var podAlloc state.PodResourceInfo
	podAlloc.ContainerResources = make(map[string]v1.ResourceRequirements)
	for _, container := range pod.Spec.Containers {
		alloc := *container.Resources.DeepCopy()
		podAlloc.ContainerResources[container.Name] = alloc
	}

	for _, container := range pod.Spec.InitContainers {
		if podutil.IsRestartableInitContainer(&container) {
			alloc := *container.Resources.DeepCopy()
			podAlloc.ContainerResources[container.Name] = alloc
		}
	}

	return podAlloc
}

func (m *manager) RemovePod(uid types.UID) {
	if err := m.allocated.RemovePod(uid); err != nil {
		// If the deletion fails, it will be retried by RemoveOrphanedPods, so we can safely ignore the error.
		klog.V(3).ErrorS(err, "Failed to delete pod allocation", "podUID", uid)
	}

	if err := m.actuated.RemovePod(uid); err != nil {
		// If the deletion fails, it will be retried by RemoveOrphanedPods, so we can safely ignore the error.
		klog.V(3).ErrorS(err, "Failed to delete pod allocation", "podUID", uid)
	}

	m.removePendingResizes(uid)
}

func (m *manager) RemoveOrphanedPods(remainingPods sets.Set[types.UID]) {
	m.allocated.RemoveOrphanedPods(remainingPods)
	m.actuated.RemoveOrphanedPods(remainingPods)

	// Remove orphaned pods from the pending resizes list.
	m.pendingResizesLock.Lock()
	var orphanedPods []types.UID
	for _, pod := range m.podsWithPendingResizes {
		if !remainingPods.Has(pod.UID) {
			orphanedPods = append(orphanedPods, pod.UID)
		}
	}
	m.pendingResizesLock.Unlock()
	m.removePendingResizes(orphanedPods...)
}

func (m *manager) SetActuatedResources(allocatedPod *v1.Pod, actuatedContainer *v1.Container) error {
	if actuatedContainer == nil {
		alloc := allocationFromPod(allocatedPod)
		return m.actuated.SetPodResourceInfo(allocatedPod.UID, alloc)
	}

	return m.actuated.SetContainerResources(allocatedPod.UID, actuatedContainer.Name, actuatedContainer.Resources)
}

func (m *manager) GetActuatedResources(podUID types.UID, containerName string) (v1.ResourceRequirements, bool) {
	return m.actuated.GetContainerResources(podUID, containerName)
}

func (m *manager) PushPendingResize(pod *v1.Pod) {
	m.pendingResizesLock.Lock()
	defer m.pendingResizesLock.Unlock()

	for _, p := range m.podsWithPendingResizes {
		if p.UID == pod.UID {
			if apiequality.Semantic.DeepEqual(p.Spec.Containers, pod.Spec.Containers) &&
				apiequality.Semantic.DeepEqual(p.Spec.InitContainers, pod.Spec.InitContainers) {
				// Pod is already in the pending resizes list with the same spec.
				return
			}

			m.removePendingResizes(pod.UID)
			break
		}
	}

	// Add the pod to the pending resizes list
	m.podsWithPendingResizes = append(m.podsWithPendingResizes, pod)

	// TODO (natasha41575): Sort the pending resizes list by priority.
	// See https://github.com/kubernetes/enhancements/pull/5266.

	m.RetryPendingResizes()
}

func (m *manager) RetryPendingResizes() {
	// Signal the pending resize channel to trigger a retry.
	// Do not block if the channel is already full.
	select {
	case m.pendingResizeChan <- struct{}{}:
	default:
	}
}

func (m *manager) HandlePodResourcesResize(pod *v1.Pod) error {
	if _, updated := m.UpdatePodFromAllocation(pod); !updated {
		// Desired resources == allocated resources. Pod allocation does not need to be updated.
		m.statusManager.ClearPodResizePendingCondition(pod.UID)
		return nil
	}

	m.podResizeMutex.Lock()
	defer m.podResizeMutex.Unlock()
	// Desired resources != allocated resources. Can we update the allocation to the desired resources?
	fit, reason, message := m.canResizePod(pod)
	if fit {
		// Update pod resource allocation checkpoint
		if err := m.SetAllocatedResources(pod); err != nil {
			return err
		}
		m.statusManager.ClearPodResizePendingCondition(pod.UID)
		m.statusManager.ClearPodResizeInProgressCondition(pod.UID)
		return nil
	}

	if reason != "" {
		m.statusManager.SetPodResizePendingCondition(pod.UID, reason, message)
	}

	return nil
}

func (m *manager) removePendingResizes(podUIDs ...types.UID) {
	m.pendingResizesLock.Lock()
	defer m.pendingResizesLock.Unlock()

	var newPendingResizes []*v1.Pod
	for _, p := range m.podsWithPendingResizes {
		var found bool
		for _, uid := range podUIDs {
			if p.UID == uid {
				found = true
				break
			}
		}
		if !found {
			newPendingResizes = append(newPendingResizes, p)
		}
	}

	m.podsWithPendingResizes = newPendingResizes
}
