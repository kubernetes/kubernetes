/*
Copyright 2026 The ostk Authors.

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

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SchedulerStateEntryStatus describes the lifecycle phase of an assume-cache entry.
type SchedulerStateEntryStatus string

const (
	// EntryInFlight means the scheduler has assumed a node but etcd has not confirmed.
	EntryInFlight SchedulerStateEntryStatus = "in-flight"
	// EntryMerged means the assume matched the eventual bind — gen table converged.
	EntryMerged SchedulerStateEntryStatus = "merged"
	// EntryConflict means the assume diverged from the bind — CAS failed.
	EntryConflict SchedulerStateEntryStatus = "conflict"
	// EntryBound means the pod is bound and the entry is terminal.
	EntryBound SchedulerStateEntryStatus = "bound"
)

// SchedulerStateEntry tracks a single pod's assume-cache generation.
type SchedulerStateEntry struct {
	// podName is the name of the pod this entry tracks.
	PodName string `json:"podName" protobuf:"bytes,1,opt,name=podName"`
	// assumedNode is the node the scheduler assumed for this pod.
	AssumedNode string `json:"assumedNode" protobuf:"bytes,2,opt,name=assumedNode"`
	// assumeGen is the generation at which the scheduler made the assumption.
	AssumeGen int64 `json:"assumeGen" protobuf:"varint,3,opt,name=assumeGen"`
	// etcdGen is the generation confirmed by etcd after the bind completes.
	EtcdGen int64 `json:"etcdGen" protobuf:"varint,4,opt,name=etcdGen"`
	// status is the lifecycle phase of this entry.
	Status SchedulerStateEntryStatus `json:"status" protobuf:"bytes,5,opt,name=status"`
}

// SchedulerStateSpec contains the assume-cache generation table.
type SchedulerStateSpec struct {
	// entries is the list of assume-cache entries tracked by this scheduler state.
	Entries []SchedulerStateEntry `json:"entries" protobuf:"bytes,1,rep,name=entries"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SchedulerState exposes the scheduler's assume-cache as a gen table.
// Each entry tracks the generation gap between what the scheduler assumed
// and what etcd confirmed, enabling CAS-style conflict detection.
type SchedulerState struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec contains the assume-cache generation table.
	Spec SchedulerStateSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SchedulerStateList is a list of SchedulerState objects.
type SchedulerStateList struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of SchedulerState objects.
	Items []SchedulerState `json:"items" protobuf:"bytes,2,rep,name=items"`
}
