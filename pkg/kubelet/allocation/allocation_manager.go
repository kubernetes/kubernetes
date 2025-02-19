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
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
)

// podStatusManagerStateFile is the file name where status manager stores its state
const podStatusManagerStateFile = "pod_status_manager_state"

// AllocationManager tracks pod resource allocations.
type Manager interface {
	// GetContainerResourceAllocation returns the AllocatedResources value for the container
	GetContainerResourceAllocation(podUID string, containerName string) (v1.ResourceRequirements, bool)

	// UpdatePodFromAllocation overwrites the pod spec with the allocation.
	// This function does a deep copy only if updates are needed.
	// Returns the updated (or original) pod, and whether there was an allocation stored.
	UpdatePodFromAllocation(pod *v1.Pod) (*v1.Pod, bool)

	// SetPodAllocation checkpoints the resources allocated to a pod's containers.
	SetPodAllocation(pod *v1.Pod) error

	// DeletePodAllocation removes any stored state for the given pod UID.
	DeletePodAllocation(uid types.UID)

	// RemoveOrphanedPods removes the stored state for any pods not included in the set of remaining pods.
	RemoveOrphanedPods(remainingPods sets.Set[types.UID])
}

type manager struct {
	state state.State
}

func NewManager(checkpointDirectory string) Manager {
	m := &manager{}

	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		stateImpl, err := state.NewStateCheckpoint(checkpointDirectory, podStatusManagerStateFile)
		if err != nil {
			// This is a crictical, non-recoverable failure.
			klog.ErrorS(err, "Could not initialize pod allocation checkpoint manager, please drain node and remove policy state file")
			panic(err)
		}
		m.state = stateImpl
	} else {
		m.state = state.NewNoopStateCheckpoint()
	}

	return m
}

// NewInMemoryManager returns an allocation manager that doesn't persist state.
// For testing purposes only!
func NewInMemoryManager() Manager {
	return &manager{
		state: state.NewStateMemory(nil),
	}
}

// GetContainerResourceAllocation returns the last checkpointed AllocatedResources values
// If checkpoint manager has not been initialized, it returns nil, false
func (m *manager) GetContainerResourceAllocation(podUID string, containerName string) (v1.ResourceRequirements, bool) {
	return m.state.GetContainerResourceAllocation(podUID, containerName)
}

// UpdatePodFromAllocation overwrites the pod spec with the allocation.
// This function does a deep copy only if updates are needed.
func (m *manager) UpdatePodFromAllocation(pod *v1.Pod) (*v1.Pod, bool) {
	// TODO(tallclair): This clones the whole cache, but we only need 1 pod.
	allocs := m.state.GetPodResourceAllocation()
	return updatePodFromAllocation(pod, allocs)
}

func updatePodFromAllocation(pod *v1.Pod, allocs state.PodResourceAllocation) (*v1.Pod, bool) {
	allocated, found := allocs[string(pod.UID)]
	if !found {
		return pod, false
	}

	updated := false
	containerAlloc := func(c v1.Container) (v1.ResourceRequirements, bool) {
		if cAlloc, ok := allocated[c.Name]; ok {
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

// SetPodAllocation checkpoints the resources allocated to a pod's containers
func (m *manager) SetPodAllocation(pod *v1.Pod) error {
	podAlloc := make(map[string]v1.ResourceRequirements)
	for _, container := range pod.Spec.Containers {
		alloc := *container.Resources.DeepCopy()
		podAlloc[container.Name] = alloc
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.SidecarContainers) {
		for _, container := range pod.Spec.InitContainers {
			if podutil.IsRestartableInitContainer(&container) {
				alloc := *container.Resources.DeepCopy()
				podAlloc[container.Name] = alloc
			}
		}
	}

	return m.state.SetPodResourceAllocation(string(pod.UID), podAlloc)
}

func (m *manager) DeletePodAllocation(uid types.UID) {
	if err := m.state.Delete(string(uid), ""); err != nil {
		// If the deletion fails, it will be retried by RemoveOrphanedPods, so we can safely ignore the error.
		klog.V(3).ErrorS(err, "Failed to delete pod allocation", "podUID", uid)
	}
}

func (m *manager) RemoveOrphanedPods(remainingPods sets.Set[types.UID]) {
	m.state.RemoveOrphanedPods(remainingPods)
}
