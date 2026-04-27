/*
Copyright 2017 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"maps"

	"k8s.io/utils/cpuset"
)

// ContainerCPUAssignments type used in cpu manager state
type ContainerCPUAssignments map[string]map[string]cpuset.CPUSet

// Clone returns a copy of ContainerCPUAssignments
func (as ContainerCPUAssignments) Clone() ContainerCPUAssignments {
	ret := make(ContainerCPUAssignments, len(as))
	for pod := range as {
		ret[pod] = make(map[string]cpuset.CPUSet, len(as[pod]))
		maps.Copy(ret[pod], as[pod])
	}
	return ret
}

// PodCPUAssignments contains pod-level CPU assignments.
type PodCPUAssignments map[string]PodEntry

// PodEntry represents pod-level CPU assignments for a pod
type PodEntry struct {
	CPUSet cpuset.CPUSet `json:"cpuSet,omitempty"`
}

type podEntryJSON struct {
	CPUSet string `json:"cpuSet,omitempty"`
}

// MarshalJSON implements the json.Marshaler interface.
// It is required because CPUSet is stored as a string in the checkpoint JSON,
// but used as a struct in the internal state.
func (p PodEntry) MarshalJSON() ([]byte, error) {
	return json.Marshal(podEntryJSON{
		CPUSet: p.CPUSet.String(),
	})
}

// UnmarshalJSON implements the json.Unmarshaler interface.
// It is required because CPUSet is stored as a string in the checkpoint JSON,
// but used as a struct in the internal state.
func (p *PodEntry) UnmarshalJSON(b []byte) error {
	var entry podEntryJSON
	if err := json.Unmarshal(b, &entry); err != nil {
		return err
	}
	cset, err := cpuset.Parse(entry.CPUSet)
	if err != nil {
		return fmt.Errorf("failed to parse cpuset: %w", err)
	}
	p.CPUSet = cset
	return nil
}

// Clone returns a copy of PodCPUAssignments
func (a PodCPUAssignments) Clone() PodCPUAssignments {
	clone := make(PodCPUAssignments, len(a))
	maps.Copy(clone, a)
	return clone
}

// Reader interface used to read current cpu/pod assignment state
type Reader interface {
	GetCPUSet(podUID string, containerName string) (cpuset.CPUSet, bool)
	GetDefaultCPUSet() cpuset.CPUSet
	GetCPUSetOrDefault(podUID string, containerName string) cpuset.CPUSet
	GetCPUAssignments() ContainerCPUAssignments
	// GetPodCPUSet returns the pod-level CPU assignments of a pod
	GetPodCPUSet(podUID string) (cpuset.CPUSet, bool)
	// GetPodCPUAssignments returns all pod-level CPU assignments
	GetPodCPUAssignments() PodCPUAssignments
}

type writer interface {
	SetCPUSet(podUID string, containerName string, cpuset cpuset.CPUSet)
	SetDefaultCPUSet(cpuset cpuset.CPUSet)
	SetCPUAssignments(ContainerCPUAssignments)
	Delete(podUID string, containerName string)
	ClearState()
	// SetPodCPUSet stores pod-level CPU assignments of a pod
	SetPodCPUSet(podUID string, cpuset cpuset.CPUSet)
	// SetPodCPUAssignments sets PodCPUAssignments by using the passed parameter
	SetPodCPUAssignments(PodCPUAssignments)
	// DeletePod deletes pod-level CPU assignments for specified pod
	DeletePod(podUID string)
}

// State interface provides methods for tracking and setting cpu/pod assignment
type State interface {
	Reader
	writer
}
