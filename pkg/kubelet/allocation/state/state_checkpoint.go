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
	"fmt"
	"path"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
)

var _ State = &stateCheckpoint{}

type stateCheckpoint struct {
	mux               sync.RWMutex
	cache             *stateMemory
	checkpointManager checkpointmanager.CheckpointManager
	checkpointName    string
	lastChecksum      checksum.Checksum
}

// NewStateCheckpoint creates new State for keeping track of pod resource information with checkpoint backend
func NewStateCheckpoint(logger klog.Logger, stateDir, checkpointName string) (State, error) {
	checkpointManager, err := checkpointmanager.NewCheckpointManager(stateDir)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize checkpoint manager for pod resource information tracking: %w", err)
	}

	pra, checksum, migrated, err := restoreState(logger, checkpointManager, checkpointName)
	if err != nil {
		//lint:ignore ST1005 user-facing error message
		return nil, fmt.Errorf("could not restore state from checkpoint: %w, please drain this node and delete pod resource information checkpoint file %q before restarting Kubelet",
			err, path.Join(stateDir, checkpointName))
	}

	stateCheckpoint := &stateCheckpoint{
		cache:             newStateMemory(logger, pra),
		checkpointManager: checkpointManager,
		checkpointName:    checkpointName,
		lastChecksum:      checksum,
	}

	if migrated {
		logger.Info("Saving migrated V2 checkpoint to disk")
		if err := stateCheckpoint.storeState(logger); err != nil {
			logger.Error(err, "Failed to save migrated V2 checkpoint to disk")
		}
	}

	return stateCheckpoint, nil
}

// restores state from a checkpoint and creates it if it doesn't exist
func restoreState(logger klog.Logger, checkpointManager checkpointmanager.CheckpointManager, checkpointName string) (PodMap, checksum.Checksum, bool, error) {
	checkpoint := &Checkpoint{}
	err := checkpointManager.GetCheckpoint(checkpointName, checkpoint)
	if err == errors.ErrCheckpointNotFound {
		return nil, 0, false, nil
	}
	if err != nil {
		return nil, 0, false, err
	}

	if checkpoint.Version == checkpointVersionV2 {
		podList, errProto := checkpoint.getPodList()
		if errProto != nil {
			return nil, 0, false, fmt.Errorf("failed to decode V2 protobuf checkpoint: %w", errProto)
		}
		podMap := make(PodMap)
		for _, pod := range podList.Items {
			podMap[pod.UID] = pod.DeepCopy()
		}
		logger.V(2).Info("State checkpoint: restored pod resource state from V2 checkpoint")
		return podMap, checkpoint.Checksum, false, nil
	}

	// Fallback to legacy JSON V1 format (unversioned)
	logger.Info("Unversioned checkpoint found, migrating legacy JSON V1 format to V2", "checkpoint", checkpointName)
	podList, errMigrate := migrateV1ToV2(checkpoint.Data)
	if errMigrate != nil {
		return nil, 0, false, fmt.Errorf("failed to migrate legacy JSON V1 checkpoint: %w", errMigrate)
	}
	podMap := make(PodMap)
	for _, pod := range podList.Items {
		podMap[pod.UID] = pod.DeepCopy()
	}
	logger.V(2).Info("State checkpoint: restored and migrated pod resource state from legacy JSON V1 checkpoint")
	return podMap, 0, true, nil
}

// saves state to a checkpoint, caller is responsible for locking
func (sc *stateCheckpoint) storeState(logger klog.Logger) error {
	podList := sc.cache.toPodList()

	checkpoint, err := NewCheckpoint(podList)
	if err != nil {
		logger.Error(err, "Failed to create pod resource information checkpoint")
		return err
	}
	if checkpoint.Checksum == sc.lastChecksum {
		// No changes to the checkpoint => no need to re-write it.
		return nil
	}
	err = sc.checkpointManager.CreateCheckpoint(sc.checkpointName, checkpoint)
	if err != nil {
		logger.Error(err, "Failed to save pod resource information checkpoint")
		return err
	}
	sc.lastChecksum = checkpoint.Checksum
	return nil
}

// GetContainerResources returns current resources information to a pod's container
func (sc *stateCheckpoint) GetContainerResources(podUID types.UID, containerName string) (v1.ResourceRequirements, bool) {
	sc.mux.RLock()
	defer sc.mux.RUnlock()
	return sc.cache.GetContainerResources(podUID, containerName)
}

// GetPodLevelResources returns current resources information at pod-level
func (sc *stateCheckpoint) GetPodLevelResources(podUID types.UID) (*v1.ResourceRequirements, bool) {
	sc.mux.RLock()
	defer sc.mux.RUnlock()
	return sc.cache.GetPodLevelResources(podUID)
}

// GetEmptyDirVolumeLimit returns current resources information for emptyDir volume
func (sc *stateCheckpoint) GetEmptyDirVolumeLimit(podUID types.UID, volumeName string) (*resource.Quantity, bool) {
	sc.mux.RLock()
	defer sc.mux.RUnlock()
	return sc.cache.GetEmptyDirVolumeLimit(podUID, volumeName)
}

// GetPodMap returns current pod map
func (sc *stateCheckpoint) GetPodMap() PodMap {
	sc.mux.RLock()
	defer sc.mux.RUnlock()
	return sc.cache.GetPodMap()
}

// GetPodUIDs returns the UIDs of all pods in the state
func (sc *stateCheckpoint) GetPodUIDs() []types.UID {
	sc.mux.RLock()
	defer sc.mux.RUnlock()
	return sc.cache.GetPodUIDs()
}

// GetPod returns current pod
func (sc *stateCheckpoint) GetPod(podUID types.UID) (*v1.Pod, bool) {
	sc.mux.RLock()
	defer sc.mux.RUnlock()
	return sc.cache.GetPod(podUID)
}

// HasPod returns whether a pod with the given UID exists in the state
func (sc *stateCheckpoint) HasPod(podUID types.UID) bool {
	sc.mux.RLock()
	defer sc.mux.RUnlock()
	return sc.cache.HasPod(podUID)
}

// SetContainerResources sets resources information for a pod's container
func (sc *stateCheckpoint) SetContainerResources(logger klog.Logger, podUID types.UID, containerName string, containerType podutil.ContainerType, resources v1.ResourceRequirements) error {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	err := sc.cache.SetContainerResources(logger, podUID, containerName, containerType, resources)
	if err != nil {
		return err
	}
	return sc.storeState(logger)
}

// SetPodLevelResources sets resources information for a pod's resources at pod-level.
func (sc *stateCheckpoint) SetPodLevelResources(logger klog.Logger, podUID types.UID, resInfo *v1.ResourceRequirements) error {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	err := sc.cache.SetPodLevelResources(logger, podUID, resInfo)
	if err != nil {
		return err
	}
	return sc.storeState(logger)
}

// SetEmptyDirVolumeLimit sets the size limit for a pod's emptyDir volume.
func (sc *stateCheckpoint) SetEmptyDirVolumeLimit(podUID types.UID, volumeName string, limit *resource.Quantity) error {
	logger := klog.TODO()
	sc.mux.Lock()
	defer sc.mux.Unlock()
	err := sc.cache.SetEmptyDirVolumeLimit(podUID, volumeName, limit)
	if err != nil {
		return err
	}
	return sc.storeState(logger)
}

// SetPod sets pod
func (sc *stateCheckpoint) SetPod(logger klog.Logger, pod *v1.Pod) error {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	err := sc.cache.SetPod(logger, pod)
	if err != nil {
		return err
	}
	return sc.storeState(logger)
}

// RemovePod deletes resource information for specified pod
func (sc *stateCheckpoint) RemovePod(logger klog.Logger, podUID types.UID) error {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	// Skip writing the checkpoint for pod deletion, since there is no side effect to
	// keeping a deleted pod. Deleted pods will eventually be cleaned up by RemoveOrphanedPods.
	// The deletion will be stored the next time a non-delete update is made.
	return sc.cache.RemovePod(logger, podUID)
}

func (sc *stateCheckpoint) RemoveOrphanedPods(remainingPods sets.Set[types.UID]) {
	sc.cache.RemoveOrphanedPods(remainingPods)
	// Don't bother updating the stored state. If Kubelet is restarted before the cache is written,
	// the orphaned pods will be removed the next time this method is called.
}

type noopStateCheckpoint struct{}

// NewNoopStateCheckpoint creates a dummy state checkpoint manager
func NewNoopStateCheckpoint() State {
	return &noopStateCheckpoint{}
}

func (sc *noopStateCheckpoint) GetContainerResources(_ types.UID, _ string) (v1.ResourceRequirements, bool) {
	return v1.ResourceRequirements{}, false
}

func (sc *noopStateCheckpoint) GetPodLevelResources(_ types.UID) (*v1.ResourceRequirements, bool) {
	return nil, false
}

func (sc *noopStateCheckpoint) GetEmptyDirVolumeLimit(_ types.UID, _ string) (*resource.Quantity, bool) {
	return nil, false
}

func (sc *noopStateCheckpoint) GetPodMap() PodMap {
	return nil
}

func (sc *noopStateCheckpoint) GetPodUIDs() []types.UID {
	return nil
}

func (sc *noopStateCheckpoint) GetPod(_ types.UID) (*v1.Pod, bool) {
	return nil, false
}

func (sc *noopStateCheckpoint) HasPod(_ types.UID) bool {
	return false
}

func (sc *noopStateCheckpoint) SetContainerResources(_ klog.Logger, _ types.UID, _ string, _ podutil.ContainerType, _ v1.ResourceRequirements) error {
	return nil
}

func (sc *noopStateCheckpoint) SetPodLevelResources(_ klog.Logger, _ types.UID, _ *v1.ResourceRequirements) error {
	return nil
}

func (sc *noopStateCheckpoint) SetEmptyDirVolumeLimit(_ types.UID, _ string, _ *resource.Quantity) error {
	return nil
}

func (sc *noopStateCheckpoint) SetPod(_ klog.Logger, _ *v1.Pod) error {
	return nil
}

func (sc *noopStateCheckpoint) RemovePod(_ klog.Logger, _ types.UID) error {
	return nil
}

func (sc *noopStateCheckpoint) RemoveOrphanedPods(_ sets.Set[types.UID]) {}
