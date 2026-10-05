/*
Copyright 2020 The Kubernetes Authors.

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
	"path/filepath"
	"sync"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	cperrors "k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
)

var _ State = &stateCheckpoint{}

type stateCheckpoint struct {
	sync.RWMutex
	logger            klog.Logger
	cache             State
	policyName        string
	checkpointManager checkpointmanager.CheckpointManager
	checkpointName    string
}

// NewCheckpointState creates new State for keeping track of memory/pod assignment with checkpoint backend
func NewCheckpointState(logger klog.Logger, stateDir, checkpointName, policyName string) (State, error) {
	logger = klog.LoggerWithName(logger, "Memory Manager state checkpoint")
	checkpointManager, err := checkpointmanager.NewCheckpointManager(stateDir)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize checkpoint manager: %v", err)
	}
	stateCheckpoint := &stateCheckpoint{
		logger:            logger,
		cache:             NewMemoryState(logger),
		policyName:        policyName,
		checkpointManager: checkpointManager,
		checkpointName:    checkpointName,
	}

	if err := stateCheckpoint.restoreState(); err != nil {
		//nolint:staticcheck // ST1005 user-facing error message
		return nil, fmt.Errorf("could not restore state from checkpoint: %v, please drain this node and delete the memory manager checkpoint file %q before restarting Kubelet",
			err, filepath.Join(stateDir, checkpointName))
	}

	return stateCheckpoint, nil
}

// migrateV1CheckpointToV2Checkpoint converts checkpoints from the v1 format to the v2 format
func (sc *stateCheckpoint) migrateV1CheckpointToV2Checkpoint(src *MemoryManagerCheckpointV1, dst *MemoryManagerCheckpointV2) {
	dst.PolicyName = src.PolicyName
	dst.MachineState = src.MachineState.Clone()
	dst.Entries = src.Entries.Clone()
}

// migrateV2CheckpointToV3Checkpoint converts checkpoints from the v2 format to the v3 format
func (sc *stateCheckpoint) migrateV2CheckpointToV3Checkpoint(src *MemoryManagerCheckpointV2, dst *MemoryManagerCheckpointV3) {
	dst.CheckpointData.PolicyName = src.PolicyName
	dst.CheckpointData.MachineState = src.MachineState.Clone()
	dst.CheckpointData.Entries = src.Entries.Clone()
	dst.CheckpointData.PodEntries = src.PodEntries.Clone()
}

// restores state from a checkpoint and creates it if it doesn't exist
func (sc *stateCheckpoint) restoreState() error {
	sc.Lock()
	defer sc.Unlock()
	cp, err := sc.loadAndMigrateCheckpointV3()
	if err != nil {
		if errors.Is(err, cperrors.ErrCheckpointNotFound) {
			return sc.storeState()
		}
		return err
	}

	if sc.policyName != cp.CheckpointData.PolicyName {
		return fmt.Errorf("[memorymanager] configured policy %q differs from state checkpoint policy %q", sc.policyName, cp.CheckpointData.PolicyName)
	}
	sc.cache.SetMachineState(cp.CheckpointData.MachineState)
	sc.cache.SetMemoryAssignments(cp.CheckpointData.Entries)
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		sc.cache.SetPodMemoryAssignments(cp.CheckpointData.PodEntries)
	}

	sc.logger.V(2).Info("State checkpoint: restored state from checkpoint")

	return nil
}

// loadAndMigrateCheckpointV3 loads the latest checkpoint and migrates it from older versions if needed.
func (sc *stateCheckpoint) loadAndMigrateCheckpointV3() (*MemoryManagerCheckpointV3, error) {
	// Try to load as V3.
	checkpointV3 := newMemoryManagerCheckpointV3()
	err := sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpointV3)
	if err == nil {
		return checkpointV3, nil
	}
	if errors.Is(err, cperrors.ErrCheckpointNotFound) {
		return nil, err
	}

	if len(checkpointV3.Data) > 0 || checkpointV3.DataChecksum != 0 {
		// This is a corrupted V3 checkpoint version. Don't fall back to previous versions, but return error.
		err = fmt.Errorf("could not load v3 checkpoint: %w", err)
		sc.logger.Error(err, "not falling back to previous versions")
		return nil, err
	}

	// Log the V3 load error and fall back to V2.
	sc.logger.Info("could not load v3 checkpoint for memory manager, falling back to v2", "err", err)

	// Try to load as V2.
	checkpointV2, err := sc.loadAndMigrateCheckpointV2()
	if err != nil {
		return nil, err
	}

	// Reset V3 (as it might contain some remaining data from previous load attempt)
	checkpointV3 = newMemoryManagerCheckpointV3()

	// Loaded V2, now migrate to V3.
	sc.logger.Info("migrating memory manager checkpoint from v2 to v3")
	sc.migrateV2CheckpointToV3Checkpoint(checkpointV2, checkpointV3)

	return checkpointV3, nil
}

// loadAndMigrateCheckpointV2 loads the checkpoint in V2 and migrates it from older versions if needed.
func (sc *stateCheckpoint) loadAndMigrateCheckpointV2() (*MemoryManagerCheckpointV2, error) {
	// Try to load as V2.
	checkpointV2 := newMemoryManagerCheckpointV2()
	err := sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpointV2)
	if err == nil {
		return checkpointV2, nil
	}

	// Log the V2 load error and fall back to V1.
	if errors.Is(err, cperrors.CorruptCheckpointError{}) {
		sc.logger.Error(err, "V2 checkpoint for memory manager is corrupt, falling back to V1")
	} else {
		sc.logger.Info("could not load V2 checkpoint for memory manager, falling back to V1", "err", err)
	}

	// Try to load as V1.
	checkpointV1, err := sc.loadCheckpointV1()
	if err != nil {
		// This will return ErrCheckpointNotFound if it's not found, or a corruption error.
		return nil, err
	}

	// Reset V2 (as it might contain some remaining data from previous load attempt)
	checkpointV2 = newMemoryManagerCheckpointV2()

	// Loaded V1, now migrate to V2.
	sc.logger.Info("migrating memory manager checkpoint from v1 to v2")
	sc.migrateV1CheckpointToV2Checkpoint(checkpointV1, checkpointV2)

	return checkpointV2, nil
}

func (sc *stateCheckpoint) loadCheckpointV1() (*MemoryManagerCheckpointV1, error) {
	checkpointV1 := newMemoryManagerCheckpointV1()
	err := sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpointV1)
	if err != nil {
		return nil, err
	}
	return checkpointV1, nil
}

// saves state to a checkpoint, caller is responsible for locking
func (sc *stateCheckpoint) storeState() error {
	checkpoint := newMemoryManagerCheckpoint()
	checkpoint.CheckpointData.PolicyName = sc.policyName
	checkpoint.CheckpointData.MachineState = sc.cache.GetMachineState()
	checkpoint.CheckpointData.Entries = sc.cache.GetMemoryAssignments()

	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		checkpoint.CheckpointData.PodEntries = sc.cache.GetPodMemoryAssignments()
	}

	err := sc.checkpointManager.CreateCheckpoint(sc.checkpointName, checkpoint)
	if err != nil {
		sc.logger.Error(err, "Could not save checkpoint")
		return err
	}
	return nil
}

// GetMemoryState returns Memory Map stored in the State
func (sc *stateCheckpoint) GetMachineState() NUMANodeMap {
	sc.RLock()
	defer sc.RUnlock()

	return sc.cache.GetMachineState()
}

// GetMemoryBlocks returns memory assignments of a container
func (sc *stateCheckpoint) GetMemoryBlocks(podUID string, containerName string) []Block {
	sc.RLock()
	defer sc.RUnlock()

	return sc.cache.GetMemoryBlocks(podUID, containerName)
}

// GetMemoryAssignments returns ContainerMemoryAssignments
func (sc *stateCheckpoint) GetMemoryAssignments() ContainerMemoryAssignments {
	sc.RLock()
	defer sc.RUnlock()

	return sc.cache.GetMemoryAssignments()
}

// GetPodMemoryBlocks returns memory assignments of a pod
func (sc *stateCheckpoint) GetPodMemoryBlocks(podUID string) []Block {
	sc.RLock()
	defer sc.RUnlock()

	return sc.cache.GetPodMemoryBlocks(podUID)
}

// GetPodMemoryAssignments returns all pod-level memory assignments
func (sc *stateCheckpoint) GetPodMemoryAssignments() PodMemoryAssignments {
	sc.RLock()
	defer sc.RUnlock()

	return sc.cache.GetPodMemoryAssignments()
}

// SetMachineState stores NUMANodeMap in State
func (sc *stateCheckpoint) SetMachineState(memoryMap NUMANodeMap) {
	sc.Lock()
	defer sc.Unlock()

	sc.cache.SetMachineState(memoryMap)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint")
	}
}

// SetMemoryBlocks stores memory assignments of container
func (sc *stateCheckpoint) SetMemoryBlocks(podUID string, containerName string, blocks []Block) {
	sc.Lock()
	defer sc.Unlock()

	sc.cache.SetMemoryBlocks(podUID, containerName, blocks)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint", "podUID", podUID, "containerName", containerName)
	}
}

// SetMemoryAssignments sets ContainerMemoryAssignments by using the passed parameter
func (sc *stateCheckpoint) SetMemoryAssignments(assignments ContainerMemoryAssignments) {
	sc.Lock()
	defer sc.Unlock()

	sc.cache.SetMemoryAssignments(assignments)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint")
	}
}

func (sc *stateCheckpoint) SetPodMemoryAssignments(assignments PodMemoryAssignments) {
	sc.Lock()
	defer sc.Unlock()

	sc.cache.SetPodMemoryAssignments(assignments)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint")
	}
}

// SetPodMemoryBlocks stores memory assignments of a pod
func (sc *stateCheckpoint) SetPodMemoryBlocks(podUID string, blocks []Block) {
	sc.Lock()
	defer sc.Unlock()

	sc.cache.SetPodMemoryBlocks(podUID, blocks)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint", "podUID", podUID)
	}
}

// Delete deletes corresponding Blocks from ContainerMemoryAssignments
func (sc *stateCheckpoint) Delete(podUID string, containerName string) {
	sc.Lock()
	defer sc.Unlock()

	sc.cache.Delete(podUID, containerName)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint", "podUID", podUID, "containerName", containerName)
	}
}

// DeletePod deletes pod-level CPU assignments for specified pod. It does not
// affect container-level assignments.
func (sc *stateCheckpoint) DeletePod(podUID string) {
	sc.Lock()
	defer sc.Unlock()

	sc.cache.DeletePod(podUID)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint", "podUID", podUID)
	}
}

// ClearState clears machineState and ContainerMemoryAssignments
func (sc *stateCheckpoint) ClearState() {
	sc.Lock()
	defer sc.Unlock()

	sc.cache.ClearState()
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint")
	}
}
