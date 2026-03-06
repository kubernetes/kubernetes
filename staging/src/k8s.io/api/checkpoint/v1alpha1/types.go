/*
Copyright 2026 The Kubernetes Authors.

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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// PodCheckpointPhase represents the phase of a PodCheckpoint.
type PodCheckpointPhase string

const (
	// PodCheckpointPending means the checkpoint has been accepted but not yet started.
	PodCheckpointPending PodCheckpointPhase = "Pending"
	// PodCheckpointInProgress means the checkpoint is currently being created.
	PodCheckpointInProgress PodCheckpointPhase = "InProgress"
	// PodCheckpointReady means the checkpoint has been successfully created.
	PodCheckpointReady PodCheckpointPhase = "Ready"
	// PodCheckpointFailed means the checkpoint creation failed.
	PodCheckpointFailed PodCheckpointPhase = "Failed"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodCheckpoint represents a checkpoint of a running pod.
type PodCheckpoint struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec defines the desired checkpoint operation.
	// +optional
	Spec PodCheckpointSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// status represents the current status of the checkpoint.
	// +optional
	Status PodCheckpointStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// PodCheckpointSpec defines the desired state of a PodCheckpoint.
type PodCheckpointSpec struct {
	// sourcePodName is the name of the pod to checkpoint. The pod must
	// exist in the same namespace as the PodCheckpoint resource.
	// +required
	SourcePodName string `json:"sourcePodName" protobuf:"bytes,1,opt,name=sourcePodName"`
}

// PodCheckpointStatus represents the current status of a PodCheckpoint.
type PodCheckpointStatus struct {
	// phase represents the current phase of the checkpoint.
	// +optional
	Phase PodCheckpointPhase `json:"phase,omitempty" protobuf:"bytes,1,opt,name=phase"`

	// nodeName is the name of the node where the checkpoint was created.
	// +optional
	NodeName string `json:"nodeName,omitempty" protobuf:"bytes,2,opt,name=nodeName"`

	// checkpointLocation is the path where the checkpoint archive is stored.
	// +optional
	CheckpointLocation string `json:"checkpointLocation,omitempty" protobuf:"bytes,3,opt,name=checkpointLocation"`

	// message is a human-readable message indicating details about the checkpoint.
	// +optional
	Message string `json:"message,omitempty" protobuf:"bytes,4,opt,name=message"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodCheckpointList is a list of PodCheckpoint objects.
type PodCheckpointList struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of PodCheckpoint objects.
	Items []PodCheckpoint `json:"items" protobuf:"bytes,2,rep,name=items"`
}
