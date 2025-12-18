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
	"maps"
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
	if src.PolicyName != "" {
		dst.PolicyName = src.PolicyName
	}
	if src.MachineState != nil {
		dst.MachineState = src.MachineState
	}
	if len(src.Entries) > 0 {
		dst.Entries = src.Entries
	}
}

// restores state from a checkpoint and creates it if it doesn't exist
func (sc *stateCheckpoint) restoreState() error {
	sc.Lock()
	defer sc.Unlock()

	var checkpoint any
	var err error

	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		checkpoint, err = sc.loadAndMigrateCheckpointV2()
	} else {
		checkpoint, err = sc.loadCheckpointV1()
	}

	if err != nil {
		if errors.Is(err, cperrors.ErrCheckpointNotFound) {
			return sc.storeState()
		}
		return err
	}

	switch cp := checkpoint.(type) {
	case *MemoryManagerCheckpointV2:
		if sc.policyName != cp.PolicyName {
			return fmt.Errorf("[memorymanager] configured policy %q differs from state checkpoint policy %q", sc.policyName, cp.PolicyName)
		}
		sc.cache.SetMachineState(cp.MachineState)
		sc.cache.SetMemoryAssignments(cp.Entries)
		sc.cache.SetPodMemoryAssignments(cp.PodEntries)
	case *MemoryManagerCheckpointV1:
		if sc.policyName != cp.PolicyName {
			return fmt.Errorf("[memorymanager] configured policy %q differs from state checkpoint policy %q", sc.policyName, cp.PolicyName)
		}
		sc.cache.SetMachineState(cp.MachineState)
		sc.cache.SetMemoryAssignments(cp.Entries)
	default:
		return fmt.Errorf("unknown checkpoint type: %T", cp)
	}

	sc.logger.V(2).Info("State checkpoint: restored state from checkpoint")

	return nil
}

// loadAndMigrateCheckpointV2 loads the latest checkpoint and migrates it from older versions if needed.
func (sc *stateCheckpoint) loadAndMigrateCheckpointV2() (*MemoryManagerCheckpointV2, error) {
	// Try to load as V2.
	checkpointV2 := newMemoryManagerCheckpointV2()
	err := sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpointV2)
	if err == nil {
		return checkpointV2, nil
	}
	if !errors.Is(err, cperrors.ErrCheckpointNotFound) {
		sc.logger.Error(err, "Could not retrieve V2 checkpoint for memory manager, falling back to V1")
	}

	// Try to load as V1.
	checkpointV1 := newMemoryManagerCheckpointV1()
	err = sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpointV1)
	if err != nil {
		// This will return ErrCheckpointNotFound if it's not found, or a corruption error.
		return nil, err
	}

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
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		return sc.storeStateV2()
	}
	return sc.storeStateV1()
}

func (sc *stateCheckpoint) storeStateV2() error {
	checkpoint := newMemoryManagerCheckpoint()
	checkpoint.PolicyName = sc.policyName
	checkpoint.MachineState = sc.cache.GetMachineState()
	checkpoint.Entries = sc.cache.GetMemoryAssignments()

	podAssignments := sc.cache.GetPodMemoryAssignments()
	maps.Copy(checkpoint.PodEntries, podAssignments)

	err := sc.checkpointManager.CreateCheckpoint(sc.checkpointName, checkpoint)
	if err != nil {
		sc.logger.Error(err, "Could not save checkpoint")
		return err
	}
	return nil
}

func (sc *stateCheckpoint) storeStateV1() error {
	checkpoint := newMemoryManagerCheckpointV1()
	checkpoint.PolicyName = sc.policyName
	checkpoint.MachineState = sc.cache.GetMachineState()
	checkpoint.Entries = sc.cache.GetMemoryAssignments()

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
