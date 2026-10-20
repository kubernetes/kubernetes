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
	"slices"
	"strings"

	"k8s.io/utils/cpuset"
)

// ContainerCPUAssignments type used in cpu manager state
// without in-place vertical scaling.
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

// String returns a string representation of ContainerCPUAssignments.
// Pod and container names are sorted alphabetically to ensure deterministic output.
// The sorting overhead is acceptable since this method is used for logging and debugging only.
func (as ContainerCPUAssignments) String() string {
	var sb strings.Builder
	sb.WriteString("{")

	for i, pod := range slices.Sorted(maps.Keys(as)) {
		if i > 0 {
			sb.WriteString(",")
		}
		containerMap := as[pod]
		fmt.Fprintf(&sb, "%q:{", pod)

		for j, container := range slices.Sorted(maps.Keys(containerMap)) {
			if j > 0 {
				sb.WriteString(",")
			}
			fmt.Fprintf(&sb, "%q:%q", container, containerMap[container].String())
		}
		sb.WriteString("}")
	}
	sb.WriteString("}")
	return sb.String()
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

// ContainerCPUAllocation tracks the current allocation state
// ContainerCPUBaselines tracks the allocation state at admission time.
// If in-place vertical scaling is enabled, the allocation state at admission
// time is required to proper resource accounting; otherwise the field is
// unused and only the current state is relevant.

type ContainerCPUBaseline struct {
	Baseline cpuset.CPUSet
}

type ContainerCPUBaselines map[string]map[string]ContainerCPUBaseline

// Clone returns a deep copy of ContainerCPUBaselines
func (as ContainerCPUBaselines) Clone() ContainerCPUBaselines {
	ret := make(ContainerCPUBaselines, len(as))
	for pod := range as {
		ret[pod] = make(map[string]ContainerCPUBaseline, len(as[pod]))
		for container, orig := range as[pod] {
			ret[pod][container] = ContainerCPUBaseline{
				Baseline: orig.Baseline.Clone(),
			}
		}
	}
	return ret
}

// PendingResize is a resize of a pod that is staged but not committed yet.
type PendingResize struct {
	// Containers are the target assignments of the changed containers, by container name.
	Containers map[string]cpuset.CPUSet
	// Pod is the target pod-level assignment, or nil if it does not change.
	Pod *cpuset.CPUSet
}

// Clone returns a deep copy of PendingResize
func (p PendingResize) Clone() PendingResize {
	ret := PendingResize{Containers: maps.Clone(p.Containers)}
	if p.Pod != nil {
		ret.Pod = new(p.Pod.Clone())
	}
	return ret
}

// PendingResizes contains the pending resizes, by pod UID.
type PendingResizes map[string]PendingResize

// Clone returns a deep copy of PendingResizes
func (p PendingResizes) Clone() PendingResizes {
	ret := make(PendingResizes, len(p))
	for podUID, pending := range p {
		ret[podUID] = pending.Clone()
	}
	return ret
}

func podCPUs(assignments ContainerCPUAssignments, podAssignments PodCPUAssignments, podUID string) cpuset.CPUSet {
	cpus := podAssignments[podUID].CPUSet
	for _, cset := range assignments[podUID] {
		cpus = cpus.Union(cset)
	}
	return cpus
}

// PendingResizeDelta compares current container and pod level assignments with
// pending resize for the pod and calculates which CPUs should be taken from the
// default CPU set and which should be returned to it. It takes the assignments
// instead of a Reader, so that the state can use it while holding its lock.
func PendingResizeDelta(assignments ContainerCPUAssignments, podAssignments PodCPUAssignments, podUID string, pending PendingResize) (taken, released cpuset.CPUSet) {
	before := podCPUs(assignments, podAssignments, podUID)
	after := podAssignments[podUID].CPUSet
	if pending.Pod != nil {
		after = *pending.Pod
	}
	for containerName, cset := range assignments[podUID] {
		if target, ok := pending.Containers[containerName]; ok {
			cset = target
		}
		after = after.Union(cset)
	}
	for _, target := range pending.Containers {
		after = after.Union(target)
	}
	return after.Difference(before), before.Difference(after)
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
	// GetCPUBaselines returns all container-level CPU assignments recorded at admission time, prior to any resize
	GetCPUBaselines() ContainerCPUBaselines
	// GetBaselineCPUSet returns container-level CPU assignments recorded at admission time, prior to any resize
	GetBaselineCPUSet(podUID string, containerName string) (cpuset.CPUSet, bool)
	// GetPendingResize returns the pending resize of a pod
	GetPendingResize(podUID string) (PendingResize, bool)
	// GetPendingResizes returns all pending resizes
	GetPendingResizes() PendingResizes
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
	// SetCPUBaselines stores all container-level CPU assignments recorded at admission time, prior to any resize
	SetCPUBaselines(ContainerCPUBaselines)
	// SetPendingResize stores the pending resize of a pod, without applying it
	SetPendingResize(podUID string, pending PendingResize)
	// SetPendingResizes stores all pending resizes
	SetPendingResizes(PendingResizes)
	// CommitPendingResize applies the pending resize of a pod and deletes it, at once
	CommitPendingResize(podUID string)
	// DeletePendingResize deletes the pending resize of a pod, without applying it
	DeletePendingResize(podUID string)
}

// State interface provides methods for tracking and setting cpu/pod assignment
type State interface {
	Reader
	writer
}
