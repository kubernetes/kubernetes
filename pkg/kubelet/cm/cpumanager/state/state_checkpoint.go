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

// NewCheckpointState creates new State for keeping track of CPU/pod assignment with checkpoint backend
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

// migrateV2CheckpointToV3Checkpoint() converts checkpoints from the v2 format to the v3 format
func (sc *stateCheckpoint) migrateV2CheckpointToV3Checkpoint(src *CPUManagerCheckpointV2, dst *CPUManagerCheckpointV3) {
	if src.PolicyName != "" {
		dst.PolicyName = src.PolicyName
	}
	if src.DefaultCPUSet != "" {
		dst.DefaultCPUSet = src.DefaultCPUSet
	}
	if len(src.Entries) > 0 {
		dst.Entries = make(map[string]map[string]string, len(src.Entries))
		for podUID, containerEntries := range src.Entries {
			dst.Entries[podUID] = make(map[string]string, len(containerEntries))
			maps.Copy(dst.Entries[podUID], containerEntries)
		}
	}
}

// migrateV3CheckpointToV4Checkpoint() converts checkpoints from the v3 format to the v4 format
func (sc *stateCheckpoint) migrateV3CheckpointToV4Checkpoint(src *CPUManagerCheckpointV3, dst *CPUManagerCheckpointV4) {
	if src.PolicyName != "" {
		dst.CheckpointData.PolicyName = src.PolicyName
	}
	if src.DefaultCPUSet != "" {
		dst.CheckpointData.DefaultCPUSet = src.DefaultCPUSet
	}
	if len(src.Entries) > 0 {
		dst.CheckpointData.Entries = make(map[string]map[string]string, len(src.Entries))
		for podUID, containerEntries := range src.Entries {
			dst.CheckpointData.Entries[podUID] = make(map[string]string, len(containerEntries))
			maps.Copy(dst.CheckpointData.Entries[podUID], containerEntries)
		}
	}
	if len(src.PodEntries) > 0 {
		dst.CheckpointData.PodEntries = make(PodCPUAssignments, len(src.PodEntries))
		for podUID, podEntries := range src.PodEntries {
			dst.CheckpointData.PodEntries[podUID] = PodEntry{CPUSet: podEntries.CPUSet.Clone()}
		}
	}
}

// restores state from a checkpoint and creates it if it doesn't exist
func (sc *stateCheckpoint) restoreState() error {
	sc.mux.Lock()
	defer sc.mux.Unlock()

	cp, err := sc.loadAndMigrateCheckpointV4()
	if err != nil {
		if errors.Is(err, cperrors.ErrCheckpointNotFound) {
			return sc.storeState()
		}
		return err
	}

	var tmpDefaultCPUSet cpuset.CPUSet
	var tmpContainerCPUSet cpuset.CPUSet
	var tmpContainerOriginalCPUSet cpuset.CPUSet
	var tmpContainerResizedCPUSet cpuset.CPUSet
	tmpAssignments := ContainerCPUAssignments{}
	tmpAllocations := ContainerCPUAllocations{}

	if sc.policyName != cp.CheckpointData.PolicyName {
		return fmt.Errorf("configured policy %q differs from state checkpoint policy %q", sc.policyName, cp.CheckpointData.PolicyName)
	}
	if tmpDefaultCPUSet, err = cpuset.Parse(cp.CheckpointData.DefaultCPUSet); err != nil {
		return fmt.Errorf("could not parse default CPU set %q: %w", cp.CheckpointData.DefaultCPUSet, err)
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveCPUs) {
		for pod := range cp.CheckpointData.Entries {
			tmpAssignments[pod] = make(map[string]cpuset.CPUSet, len(cp.CheckpointData.Entries[pod]))
			for container, cpuString := range cp.CheckpointData.Entries[pod] {
				if tmpContainerCPUSet, err = cpuset.Parse(cpuString); err != nil {
					return fmt.Errorf("could not parse cpuset %q for container %q in pod %q: %w", cpuString, container, pod, err)
				}
				tmpAssignments[pod][container] = tmpContainerCPUSet
			}
		}
	} else {
		for pod := range cp.CheckpointData.Allocations {
			tmpAllocations[pod] = make(map[string]ContainerCPUAllocation, len(cp.CheckpointData.Allocations[pod]))
			for container, cpuAllocation := range cp.CheckpointData.Allocations[pod] {
				if tmpContainerOriginalCPUSet, err = cpuset.Parse(cpuAllocation.Original); err != nil {
					return fmt.Errorf("could not parse Original cpuset %q for container %q in pod %q: %w", cpuAllocation.Original, container, pod, err)
				}
				if tmpContainerResizedCPUSet, err = cpuset.Parse(cpuAllocation.Resized); err != nil {
					return fmt.Errorf("could not parse Resized cpuset %q for container %q in pod %q: %w", cpuAllocation.Resized, container, pod, err)
				}
				tmpAllocations[pod][container] = ContainerCPUAllocation{Original: tmpContainerOriginalCPUSet, Resized: tmpContainerResizedCPUSet}
			}
		}
	}

	sc.cache.SetDefaultCPUSet(tmpDefaultCPUSet)
	if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveCPUs) {
		sc.cache.SetCPUAssignments(tmpAssignments)
	} else {
		sc.cache.SetCPUAllocations(tmpAllocations)
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		sc.cache.SetPodCPUAssignments(cp.CheckpointData.PodEntries)
	}

	sc.logger.V(2).Info("restored state from checkpoint")
	sc.logger.V(2).Info("defaultCPUSet", "defaultCpuSet", tmpDefaultCPUSet.String())

	return nil
}

// loadAndMigrateCheckpointV4 loads the latest checkpoint and migrates it from older versions if needed.
func (sc *stateCheckpoint) loadAndMigrateCheckpointV4() (*CPUManagerCheckpointV4, error) {
	sc.logger.Info("trying to load v4 CPU manager checkpoint")
	checkpointV4 := newCPUManagerCheckpointV4()
	err := sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpointV4)
	if err == nil {
		return checkpointV4, nil
	}
	if errors.Is(err, cperrors.ErrCheckpointNotFound) {
		return nil, err
	}

	if len(checkpointV4.Data) > 0 || checkpointV4.DataChecksum != 0 {
		// This is a corrupted V4 checkpoint version. Don't fall back to previous versions, but return error.
		err = fmt.Errorf("could not load v4 checkpoint: %w", err)
		sc.logger.Error(err, "not falling back to previous versions")
		return nil, err
	}

	// Log the V4 load error and fall back to V3.
	sc.logger.Info("could not load v4 checkpoint, falling back to v3", "err", err)

	// Try to load as V3.
	checkpointV3, err := sc.loadAndMigrateCheckpointV3()
	if err != nil {
		return nil, err
	}

	// Reset V4 (as it might contain some remaining data from previous load attempt)
	checkpointV4 = newCPUManagerCheckpointV4()

	// Loaded V3, now migrate to V4.
	sc.logger.Info("migrating CPU manager checkpoint from v3 to v4")
	sc.migrateV3CheckpointToV4Checkpoint(checkpointV3, checkpointV4)

	return checkpointV4, nil
}

// loadAndMigrateCheckpointV3 loads the checkpoint in V3 and migrates it from older versions if needed.
func (sc *stateCheckpoint) loadAndMigrateCheckpointV3() (*CPUManagerCheckpointV3, error) {
	sc.logger.Info("trying to load v3 CPU manager checkpoint")
	checkpointV3 := newCPUManagerCheckpointV3()
	err := sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpointV3)
	if err == nil {
		return checkpointV3, nil
	}
	if errors.Is(err, cperrors.ErrCheckpointNotFound) {
		return nil, err
	}

	// Log the V3 load error and fall back to V2.
	sc.logger.Info("could not load v3 checkpoint, falling back to v2", "err", err)

	// Try to load as V2.
	checkpointV2, err := sc.loadCheckpointV2()
	if err != nil {
		return nil, err
	}

	// Reset V3 (as it might contain some remaining data from previous load attempt)
	checkpointV3 = newCPUManagerCheckpointV3()

	// Loaded V2, now migrate to V3.
	sc.logger.Info("migrating CPU manager checkpoint from v2 to v3")
	sc.migrateV2CheckpointToV3Checkpoint(checkpointV2, checkpointV3)

	return checkpointV3, nil
}

// loadCheckpointV2 loads the checkpoint in V2.
// This is the oldest supported version so there is no try to migrate from V1.
func (sc *stateCheckpoint) loadCheckpointV2() (*CPUManagerCheckpointV2, error) {
	sc.logger.Info("trying to load v2 cpu manager checkpoint")
	checkpointV2 := newCPUManagerCheckpointV2()
	err := sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpointV2)
	if err == nil {
		return checkpointV2, nil
	}

	// All attempts failed. Return the last error we got.
	return nil, err
}

// saves state to a checkpoint, caller is responsible for locking
func (sc *stateCheckpoint) storeState() error {
	checkpoint := newCPUManagerCheckpoint()
	checkpoint.CheckpointData.PolicyName = sc.policyName
	checkpoint.CheckpointData.DefaultCPUSet = sc.cache.GetDefaultCPUSet().String()

	if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveCPUs) {
		assignments := sc.cache.GetCPUAssignments()
		for pod := range assignments {
			checkpoint.CheckpointData.Entries[pod] = make(map[string]string, len(assignments[pod]))
			for container, cset := range assignments[pod] {
				checkpoint.CheckpointData.Entries[pod][container] = cset.String()
			}
		}
	} else {
		allocations := sc.cache.GetCPUAllocations()
		for pod := range allocations {
			checkpoint.CheckpointData.Allocations[pod] = make(map[string]ContainerCPUs, len(allocations[pod]))
			for container, cpuAllocation := range allocations[pod] {
				checkpoint.CheckpointData.Allocations[pod][container] = ContainerCPUs{Original: cpuAllocation.Original.String(), Resized: cpuAllocation.Resized.String()}
			}
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		podAssignments := sc.cache.GetPodCPUAssignments()
		maps.Copy(checkpoint.CheckpointData.PodEntries, podAssignments)
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

// GetOriginalCPUSet returns current CPU set
func (sc *stateCheckpoint) GetOriginalCPUSet(podUID string, containerName string) (cpuset.CPUSet, bool) {
	sc.mux.RLock()
	defer sc.mux.RUnlock()

	return sc.cache.GetOriginalCPUSet(podUID, containerName)
}

// GetResizedCPUSet returns current CPU set
func (sc *stateCheckpoint) GetResizedCPUSet(podUID string, containerName string) (cpuset.CPUSet, bool) {
	sc.mux.RLock()
	defer sc.mux.RUnlock()

	return sc.cache.GetResizedCPUSet(podUID, containerName)
}

// GetCPUAllocations returns current CPU to pod allocations
// with InPlacePodVerticalScalingExclusiveCPUs
func (sc *stateCheckpoint) GetCPUAllocations() ContainerCPUAllocations {
	sc.mux.RLock()
	defer sc.mux.RUnlock()

	return sc.cache.GetCPUAllocations()
}

// SetCPUAllocations sets CPU to pod allocations
// with InPlacePodVerticalScalingExclusiveCPUs
func (sc *stateCheckpoint) SetCPUAllocations(a ContainerCPUAllocations) {
	sc.mux.Lock()
	defer sc.mux.Unlock()
	sc.cache.SetCPUAllocations(a)
	err := sc.storeState()
	if err != nil {
		sc.logger.Error(err, "Failed to store state to checkpoint")
	}
}
