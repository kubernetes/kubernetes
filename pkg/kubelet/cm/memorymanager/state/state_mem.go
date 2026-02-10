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
	"maps"
	"sync"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
)

type stateMemory struct {
	sync.RWMutex
	logger         klog.Logger
	assignments    ContainerMemoryAssignments
	podAssignments PodMemoryAssignments
	machineState   NUMANodeMap
}

var _ State = &stateMemory{}

// NewMemoryState creates new State for keeping track of cpu/pod assignment
func NewMemoryState(logger klog.Logger) State {
	logger.Info("Initializing new in-memory state store")

	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		return &stateMemory{
			logger:         logger,
			assignments:    ContainerMemoryAssignments{},
			podAssignments: PodMemoryAssignments{},
			machineState:   NUMANodeMap{},
		}
	}

	return &stateMemory{
		logger:       logger,
		assignments:  ContainerMemoryAssignments{},
		machineState: NUMANodeMap{},
	}
}

// GetMemoryState returns Memory Map stored in the State
func (s *stateMemory) GetMachineState() NUMANodeMap {
	s.RLock()
	defer s.RUnlock()

	return s.machineState.Clone()
}

// GetMemoryBlocks returns memory assignments of a container
func (s *stateMemory) GetMemoryBlocks(podUID string, containerName string) []Block {
	s.RLock()
	defer s.RUnlock()

	if res, ok := s.assignments[podUID][containerName]; ok {
		return append([]Block{}, res...)
	}
	return nil
}

// GetMemoryAssignments returns ContainerMemoryAssignments
func (s *stateMemory) GetMemoryAssignments() ContainerMemoryAssignments {
	s.RLock()
	defer s.RUnlock()

	return s.assignments.Clone()
}

// GetPodMemoryBlocks returns memory assignments of a pod
func (s *stateMemory) GetPodMemoryBlocks(podUID string) []Block {
	s.RLock()
	defer s.RUnlock()

	if res, ok := s.podAssignments[podUID]; ok {
		return append([]Block{}, res.MemoryBlocks...)
	}
	return nil
}

// GetPodMemoryAssignments returns all pod-level memory assignments
func (s *stateMemory) GetPodMemoryAssignments() PodMemoryAssignments {
	s.RLock()
	defer s.RUnlock()

	clone := make(PodMemoryAssignments)
	maps.Copy(clone, s.podAssignments)
	return clone
}

// SetMachineState stores NUMANodeMap in State
func (s *stateMemory) SetMachineState(nodeMap NUMANodeMap) {
	s.Lock()
	defer s.Unlock()

	s.machineState = nodeMap.Clone()
	s.logger.Info("Updated machine memory state")
}

// SetMemoryBlocks stores memory assignments of container
func (s *stateMemory) SetMemoryBlocks(podUID string, containerName string, blocks []Block) {
	s.Lock()
	defer s.Unlock()

	if _, ok := s.assignments[podUID]; !ok {
		s.assignments[podUID] = map[string][]Block{}
	}

	s.assignments[podUID][containerName] = append([]Block{}, blocks...)
	s.logger.Info("Updated memory state", "podUID", podUID, "containerName", containerName)
}

// SetMemoryAssignments sets ContainerMemoryAssignments by using the passed parameter
func (s *stateMemory) SetMemoryAssignments(assignments ContainerMemoryAssignments) {
	s.Lock()
	defer s.Unlock()

	s.assignments = assignments.Clone()
	s.logger.V(5).Info("Updated Memory assignments", "assignments", assignments)
}

// SetPodMemoryAssignments sets PodMemoryAssignments by using the passed parameter
func (s *stateMemory) SetPodMemoryAssignments(assignments PodMemoryAssignments) {
	s.Lock()
	defer s.Unlock()

	s.podAssignments = make(PodMemoryAssignments)
	maps.Copy(s.podAssignments, assignments)
	s.logger.V(5).Info("Updated Pod Memory assignments", "assignments", assignments)
}

// SetPodMemoryBlocks stores memory assignments of a pod
func (s *stateMemory) SetPodMemoryBlocks(podUID string, blocks []Block) {
	s.Lock()
	defer s.Unlock()

	podAssigments := s.podAssignments[podUID]
	podAssigments.MemoryBlocks = append([]Block{}, blocks...)
	s.podAssignments[podUID] = podAssigments
	s.logger.Info("Updated pod memory state", "podUID", podUID)
}

// Delete deletes corresponding Blocks from ContainerMemoryAssignments
func (s *stateMemory) Delete(podUID string, containerName string) {
	s.Lock()
	defer s.Unlock()

	if _, ok := s.assignments[podUID]; !ok {
		return
	}

	delete(s.assignments[podUID], containerName)
	if len(s.assignments[podUID]) == 0 {
		delete(s.assignments, podUID)
	}
	s.logger.V(2).Info("Deleted memory assignment", "podUID", podUID, "containerName", containerName)
}

// DeletePod deletes pod-level CPU assignments for specified pod. It does not
// affect container-level assignments.
func (s *stateMemory) DeletePod(podUID string) {
	s.Lock()
	defer s.Unlock()

	delete(s.podAssignments, podUID)
	s.logger.V(2).Info("Deleted pod memory assignment", "podUID", podUID)
}

// ClearState clears machineState and ContainerMemoryAssignments
func (s *stateMemory) ClearState() {
	s.Lock()
	defer s.Unlock()

	s.machineState = NUMANodeMap{}
	s.assignments = make(ContainerMemoryAssignments)
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		s.podAssignments = make(PodMemoryAssignments)
	}
	s.logger.V(2).Info("Cleared state")
}
