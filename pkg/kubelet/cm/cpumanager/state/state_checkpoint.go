/*
Copyright 2018 The Kubernetes Authors.

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

	"github.com/go-logr/logr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	cperrors "k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/utils/cpuset"
)

var _ State = &stateCheckpoint{}

type stateCheckpoint struct {
	mux               sync.RWMutex
	logger            logr.Logger
	policyName        string
	cache             State
	checkpointManager checkpointmanager.CheckpointManager
	checkpointName    string
	initialContainers containermap.ContainerMap
}

// NewCheckpointState creates new State for keeping track of cpu/pod assignment with checkpoint backend
func NewCheckpointState(logger logr.Logger, stateDir, checkpointName, policyName string, initialContainers containermap.ContainerMap) (State, error) {
	// we store a logger instance because the checkpointmanager code gets no context yet, so it's pointless to add on our outer layer
	// since we store a checkpoint, we can use the relatively expensive "WithName".
	logger = klog.LoggerWithName(logger, "CPUManager state checkpoint")
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
		initialContainers: initialContainers,
	}

	if err := stateCheckpoint.restoreState(); err != nil {
		//nolint:staticcheck // ST1005 user-facing error message
		return nil, fmt.Errorf("could not restore state from checkpoint: %v, please drain this node and delete the CPU manager checkpoint file %q before restarting Kubelet",
			err, filepath.Join(stateDir, checkpointName))
	}

	return stateCheckpoint, nil
}

// migrateV1CheckpointToV2Checkpoint() converts checkpoints from the v1 format to the v2 format
func (sc *stateCheckpoint) migrateV1CheckpointToV2Checkpoint(src *CPUManagerCheckpointV1, dst *CPUManagerCheckpointV2) error {
	if src.PolicyName != "" {
		dst.PolicyName = src.PolicyName
	}
	if src.DefaultCPUSet != "" {
		dst.DefaultCPUSet = src.DefaultCPUSet
	}
	for containerID, cset := range src.Entries {
		podUID, containerName, err := sc.initialContainers.GetContainerRef(containerID)
		if err != nil {
			return fmt.Errorf("containerID '%v' not found in initial containers list", containerID)
		}
		if dst.Entries == nil {
			dst.Entries = make(map[string]map[string]string)
		}
		if _, exists := dst.Entries[podUID]; !exists {
			dst.Entries[podUID] = make(map[string]string)
		}
		dst.Entries[podUID][containerName] = cset
	}
	return nil
}

// migrateV2CheckpointToV3Checkpoint() converts checkpoints from the v1 format to the v2 format
func (sc *stateCheckpoint) migrateV2CheckpointToV3Checkpoint(src *CPUManagerCheckpointV2, dst *CPUManagerCheckpointV3) {
	if src.PolicyName != "" {
		dst.PolicyName = src.PolicyName
	}
	if src.DefaultCPUSet != "" {
		dst.DefaultCPUSet = src.DefaultCPUSet
	}
	if len(src.Entries) > 0 {
		dst.Entries = src.Entries
	}
}

// restores state from a checkpoint and creates it if it doesn't exist
func (sc *stateCheckpoint) restoreState() error {
	sc.mux.Lock()
	defer sc.mux.Unlock()

	var checkpoint any
	var err error
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		checkpoint, err = sc.loadAndMigrateCheckpointV3()
	} else {
		checkpoint, err = sc.loadAndMigrateCheckpointV2()
	}

	if err != nil {
		if errors.Is(err, cperrors.ErrCheckpointNotFound) {
			return sc.storeState()
		}
		return err
	}

	var tmpDefaultCPUSet cpuset.CPUSet
	var tmpContainerCPUSet cpuset.CPUSet
	tmpAssignments := ContainerCPUAssignments{}
	tmpPodAssignments := PodCPUAssignments{}

	switch cp := checkpoint.(type) {
	case *CPUManagerCheckpointV3:
		if sc.policyName != cp.PolicyName {
			return fmt.Errorf("configured policy %q differs from state checkpoint policy %q", sc.policyName, cp.PolicyName)
		}
		if tmpDefaultCPUSet, err = cpuset.Parse(cp.DefaultCPUSet); err != nil {
			return fmt.Errorf("could not parse default cpu set %q: %w", cp.DefaultCPUSet, err)
		}
		for pod := range cp.Entries {
			tmpAssignments[pod] = make(map[string]cpuset.CPUSet, len(cp.Entries[pod]))
			for container, cpuString := range cp.Entries[pod] {
				if tmpContainerCPUSet, err = cpuset.Parse(cpuString); err != nil {
					return fmt.Errorf("could not parse cpuset %q for container %q in pod %q: %w", cpuString, container, pod, err)
				}
				tmpAssignments[pod][container] = tmpContainerCPUSet
			}
		}
		maps.Copy(tmpPodAssignments, cp.PodEntries)
		sc.cache.SetPodCPUAssignments(tmpPodAssignments)
	case *CPUManagerCheckpointV2:
		if sc.policyName != cp.PolicyName {
			return fmt.Errorf("configured policy %q differs from state checkpoint policy %q", sc.policyName, cp.PolicyName)
		}
		if tmpDefaultCPUSet, err = cpuset.Parse(cp.DefaultCPUSet); err != nil {
			return fmt.Errorf("could not parse default cpu set %q: %w", cp.DefaultCPUSet, err)
		}
		for pod := range cp.Entries {
			tmpAssignments[pod] = make(map[string]cpuset.CPUSet, len(cp.Entries[pod]))
			for container, cpuString := range cp.Entries[pod] {
				if tmpContainerCPUSet, err = cpuset.Parse(cpuString); err != nil {
					return fmt.Errorf("could not parse cpuset %q for container %q in pod %q: %w", cpuString, container, pod, err)
				}
				tmpAssignments[pod][container] = tmpContainerCPUSet
			}
		}
	default:
		return fmt.Errorf("unknown checkpoint version %T", cp)
	}

	sc.cache.SetDefaultCPUSet(tmpDefaultCPUSet)
	sc.cache.SetCPUAssignments(tmpAssignments)

	sc.logger.V(2).Info("restored state from checkpoint")
	sc.logger.V(2).Info("defaultCPUSet", "defaultCpuSet", tmpDefaultCPUSet.String())

	return nil
}

// loadAndMigrateCheckpoint loads the latest checkpoint and migrates it from older versions if needed.
func (sc *stateCheckpoint) loadAndMigrateCheckpointV3() (*CPUManagerCheckpointV3, error) {
	// Try to load as V3.
	checkpointV3 := newCPUManagerCheckpointV3()
	err := sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpointV3)
	if err == nil {
		return checkpointV3, nil
	}
	// Log the V3 load error and fall back to V2.
	sc.logger.Error(err, "could not load V3 checkpoint, falling back to V2")

	// Try to load as V2.
	checkpointV2, err := sc.loadAndMigrateCheckpointV2()
	if err != nil {
		return nil, err
	}

	// Loaded V2, now migrate to V3.
	sc.logger.Info("migrating cpu manager checkpoint from v2 to v3")
	sc.migrateV2CheckpointToV3Checkpoint(checkpointV2, checkpointV3)

	return checkpointV3, nil
}

func (sc *stateCheckpoint) loadAndMigrateCheckpointV2() (*CPUManagerCheckpointV2, error) {
	// Try to load as V2.
	checkpointV2 := newCPUManagerCheckpointV2()
	err := sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpointV2)
	if err == nil {
		return checkpointV2, nil
	}
	// Log the V2 load error and fall back to V1.
	sc.logger.Error(err, "could not load V2 checkpoint, falling back to V1")

	// Try to load as V1.
	checkpointV1 := newCPUManagerCheckpointV1()
	err = sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpointV1)
	if err == nil {
		// Loaded V1, now migrate V1 -> V2.
		sc.logger.Info("migrating cpu manager checkpoint from v1 to v3")
		tmpV2 := newCPUManagerCheckpointV2()
		if migrationErr := sc.migrateV1CheckpointToV2Checkpoint(checkpointV1, tmpV2); migrationErr != nil {
			return nil, fmt.Errorf("failed to migrate checkpoint from v1 to v2: %w", migrationErr)
		}
		return tmpV2, nil
	}

	// All attempts failed. Return the last error we got (from the V1 read attempt).
	return nil, err
}

// saves state to a checkpoint, caller is responsible for locking
func (sc *stateCheckpoint) storeState() error {
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		return sc.storeStateV3()
	}
	return sc.storeStateV2()
}

// saves state to a checkpoint, caller is responsible for locking
func (sc *stateCheckpoint) storeStateV3() error {
	checkpoint := newCPUManagerCheckpoint()
	checkpoint.PolicyName = sc.policyName
	checkpoint.DefaultCPUSet = sc.cache.GetDefaultCPUSet().String()

	assignments := sc.cache.GetCPUAssignments()
	for pod := range assignments {
		checkpoint.Entries[pod] = make(map[string]string, len(assignments[pod]))
		for container, cset := range assignments[pod] {
			checkpoint.Entries[pod][container] = cset.String()
		}
	}

	podAssignments := sc.cache.GetPodCPUAssignments()
	maps.Copy(checkpoint.PodEntries, podAssignments)

	err := sc.checkpointManager.CreateCheckpoint(sc.checkpointName, checkpoint)
	if err != nil {
		sc.logger.Error(err, "Failed to save checkpoint")
		return err
	}
	return nil
}

// saves state to a checkpoint, caller is responsible for locking
func (sc *stateCheckpoint) storeStateV2() error {
	checkpoint := newCPUManagerCheckpointV2()
	checkpoint.PolicyName = sc.policyName
	checkpoint.DefaultCPUSet = sc.cache.GetDefaultCPUSet().String()

	assignments := sc.cache.GetCPUAssignments()
	for pod := range assignments {
		checkpoint.Entries[pod] = make(map[string]string, len(assignments[pod]))
		for container, cset := range assignments[pod] {
			checkpoint.Entries[pod][container] = cset.String()
		}
	}

	err := sc.checkpointManager.CreateCheckpoint(sc.checkpointName, checkpoint)
	if err != nil {
		sc.logger.Error(err, "Failed to save checkpoint")
		return err
	}
	return nil
}

// GetCPUSet returns current CPU set
func (sc *stateCheckpoint) GetCPUSet(podUID string, containerName string) (cpuset.CPUSet, bool) {
	sc.mux.RLock()
	defer sc.mux.RUnlock()

	res, ok := sc.cache.GetCPUSet(podUID, containerName)
	return res, ok
}

// GetDefaultCPUSet returns default CPU set
func (sc *stateCheckpoint) GetDefaultCPUSet() cpuset.CPUSet {
	sc.mux.RLock()
	defer sc.mux.RUnlock()

	return sc.cache.GetDefaultCPUSet()
}

// GetCPUSetOrDefault returns current CPU set, or default one if it wasn't changed
func (sc *stateCheckpoint) GetCPUSetOrDefault(podUID string, containerName string) cpuset.CPUSet {
	sc.mux.RLock()
	defer sc.mux.RUnlock()

	return sc.cache.GetCPUSetOrDefault(podUID, containerName)
}

// GetCPUAssignments returns current CPU to pod assignments
func (sc *stateCheckpoint) GetCPUAssignments() ContainerCPUAssignments {
	sc.mux.RLock()
	defer sc.mux.RUnlock()

	return sc.cache.GetCPUAssignments()
}

// GetPodCPUAssignments returns pod-level CPU assignments
func (sc *stateCheckpoint) GetPodCPUAssignments() PodCPUAssignments {
	sc.mux.RLock()
	defer sc.mux.RUnlock()

	return sc.cache.GetPodCPUAssignments()
}

func (sc *stateCheckpoint) SetPodCPUAssignments(assignments PodCPUAssignments) {
	sc.mux.Lock()
	defer sc.mux.Unlock()

	sc.cache.SetPodCPUAssignments(assignments)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint")
	}
}

// GetPodCPUSet returns pod-level CPU set
func (sc *stateCheckpoint) GetPodCPUSet(podUID string) (cpuset.CPUSet, bool) {
	sc.mux.RLock()
	defer sc.mux.RUnlock()
	return sc.cache.GetPodCPUSet(podUID)
}

// SetCPUSet sets CPU set
func (sc *stateCheckpoint) SetCPUSet(podUID string, containerName string, cset cpuset.CPUSet) {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.SetCPUSet(podUID, containerName, cset)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint", "podUID", podUID, "containerName", containerName)
	}
}

// SetDefaultCPUSet sets default CPU set
func (sc *stateCheckpoint) SetDefaultCPUSet(cset cpuset.CPUSet) {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.SetDefaultCPUSet(cset)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint")
	}
}

// SetCPUAssignments sets CPU to pod assignments
func (sc *stateCheckpoint) SetCPUAssignments(a ContainerCPUAssignments) {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.SetCPUAssignments(a)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint")
	}
}

// SetPodCPUSet sets pod-level CPU set
func (sc *stateCheckpoint) SetPodCPUSet(podUID string, cset cpuset.CPUSet) {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.SetPodCPUSet(podUID, cset)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint", "podUID", podUID)
	}
}

// Delete deletes assignment for specified pod
func (sc *stateCheckpoint) Delete(podUID string, containerName string) {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.Delete(podUID, containerName)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint", "podUID", podUID, "containerName", containerName)
	}
}

// DeletePod deletes pod-level CPU assignments for specified pod. It does not
// affect container-level assignments.
func (sc *stateCheckpoint) DeletePod(podUID string) {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.DeletePod(podUID)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint", "podUID", podUID)
	}
}

// ClearState clears the state and saves it in a checkpoint
func (sc *stateCheckpoint) ClearState() {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.ClearState()
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint")
	}
}
