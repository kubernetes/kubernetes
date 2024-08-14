/*
Copyright 2021 The Kubernetes Authors.

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

package state

import (
	"errors"
	"fmt"
	"path"
	"sync"

	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"

	checkpointerrors "k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
)

var _ State = &stateCheckpoint{}

type stateCheckpoint struct {
	mux               sync.RWMutex
	cache             State
	checkpointManager checkpointmanager.CheckpointManager
	checkpointName    string
}

// NewStateCheckpoint creates new State for keeping track of pod resource allocations with checkpoint backend
func NewStateCheckpoint(stateDir, checkpointName string) (State, error) {
	checkpointManager, err := checkpointmanager.NewCheckpointManager(stateDir)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize checkpoint manager for pod allocation tracking: %v", err)
	}
	stateCheckpoint := &stateCheckpoint{
		cache:             NewStateMemory(),
		checkpointManager: checkpointManager,
		checkpointName:    checkpointName,
	}

	if err := stateCheckpoint.restoreState(); err != nil {
		//lint:ignore ST1005 user-facing error message
		return nil, fmt.Errorf("could not restore state from checkpoint: %v, please drain this node and delete pod allocation checkpoint file %q before restarting Kubelet", err, path.Join(stateDir, checkpointName))
	}
	return stateCheckpoint, nil
}

// restores state from a checkpoint and creates it if it doesn't exist
func (sc *stateCheckpoint) restoreState() error {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	var err error

	checkpoint := NewPodResourceAllocationCheckpoint()

	if err = sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpoint); err != nil {
		if errors.Is(err, checkpointerrors.ErrCheckpointNotFound) {
			return sc.storeState()
		}
		return err
	}

	sc.cache.SetPodResourceAllocation(checkpoint.AllocationEntries)
	sc.cache.SetResizeStatus(checkpoint.ResizeStatusEntries)
	klog.V(2).InfoS("State checkpoint: restored pod resource allocation state from checkpoint")
	return nil
}

// saves state to a checkpoint, caller is responsible for locking
func (sc *stateCheckpoint) storeState() error {
	checkpoint := NewPodResourceAllocationCheckpoint()

	podAllocation := sc.cache.GetPodResourceAllocation()
	for pod := range podAllocation {
		checkpoint.AllocationEntries[pod] = make(map[string]v1.ResourceList)
		for container, alloc := range podAllocation[pod] {
			checkpoint.AllocationEntries[pod][container] = alloc
		}
	}

	podResizeStatus := sc.cache.GetResizeStatus()
	checkpoint.ResizeStatusEntries = make(map[string]v1.PodResizeStatus)
	for pUID, rStatus := range podResizeStatus {
		checkpoint.ResizeStatusEntries[pUID] = rStatus
	}

	err := sc.checkpointManager.CreateCheckpoint(sc.checkpointName, checkpoint)
	if err != nil {
		klog.ErrorS(err, "Failed to save pod allocation checkpoint")
		return err
	}
	return nil
}

// GetContainerResourceAllocation returns current resources allocated to a pod's container
func (sc *stateCheckpoint) GetContainerResourceAllocation(podUID string, containerName string) (v1.ResourceList, bool) {
	sc.mux.RLock()
	defer sc.mux.RUnlock()
	return sc.cache.GetContainerResourceAllocation(podUID, containerName)
}

// GetPodResourceAllocation returns current pod resource allocation
func (sc *stateCheckpoint) GetPodResourceAllocation() PodResourceAllocation {
	sc.mux.RLock()
	defer sc.mux.RUnlock()
	return sc.cache.GetPodResourceAllocation()
}

// GetPodResizeStatus returns the last resize decision for a pod
func (sc *stateCheckpoint) GetPodResizeStatus(podUID string) (v1.PodResizeStatus, bool) {
	sc.mux.RLock()
	defer sc.mux.RUnlock()
	return sc.cache.GetPodResizeStatus(podUID)
}

// GetResizeStatus returns the set of resize decisions made
func (sc *stateCheckpoint) GetResizeStatus() PodResizeStatus {
	sc.mux.RLock()
	defer sc.mux.RUnlock()
	return sc.cache.GetResizeStatus()
}

// SetContainerResourceAllocation sets resources allocated to a pod's container
func (sc *stateCheckpoint) SetContainerResourceAllocation(podUID string, containerName string, alloc v1.ResourceList) error {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.SetContainerResourceAllocation(podUID, containerName, alloc)
	return sc.storeState()
}

// SetPodResourceAllocation sets pod resource allocation
func (sc *stateCheckpoint) SetPodResourceAllocation(a PodResourceAllocation) error {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.SetPodResourceAllocation(a)
	return sc.storeState()
}

// SetPodResizeStatus sets the last resize decision for a pod
func (sc *stateCheckpoint) SetPodResizeStatus(podUID string, resizeStatus v1.PodResizeStatus) error {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.SetPodResizeStatus(podUID, resizeStatus)
	return sc.storeState()
}

// SetResizeStatus sets the resize decisions
func (sc *stateCheckpoint) SetResizeStatus(rs PodResizeStatus) error {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.SetResizeStatus(rs)
	return sc.storeState()
}

// Delete deletes allocations for specified pod
func (sc *stateCheckpoint) Delete(podUID string, containerName string) error {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.Delete(podUID, containerName)
	return sc.storeState()
}

// ClearState clears the state and saves it in a checkpoint
func (sc *stateCheckpoint) ClearState() error {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.ClearState()
	return sc.storeState()
}

type noopStateCheckpoint struct{}

// NewNoopStateCheckpoint creates a dummy state checkpoint manager
func NewNoopStateCheckpoint() State {
	return &noopStateCheckpoint{}
}

func (sc *noopStateCheckpoint) GetContainerResourceAllocation(_ string, _ string) (v1.ResourceList, bool) {
	return nil, false
}

func (sc *noopStateCheckpoint) GetPodResourceAllocation() PodResourceAllocation {
	return nil
}

func (sc *noopStateCheckpoint) GetPodResizeStatus(_ string) (v1.PodResizeStatus, bool) {
	return "", false
}

func (sc *noopStateCheckpoint) GetResizeStatus() PodResizeStatus {
	return nil
}

func (sc *noopStateCheckpoint) SetContainerResourceAllocation(_ string, _ string, _ v1.ResourceList) error {
	return nil
}

func (sc *noopStateCheckpoint) SetPodResourceAllocation(_ PodResourceAllocation) error {
	return nil
}

func (sc *noopStateCheckpoint) SetPodResizeStatus(_ string, _ v1.PodResizeStatus) error {
	return nil
}

func (sc *noopStateCheckpoint) SetResizeStatus(_ PodResizeStatus) error {
	return nil
}

func (sc *noopStateCheckpoint) Delete(_ string, _ string) error {
	return nil
}

func (sc *noopStateCheckpoint) ClearState() error {
	return nil
}
