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
	corev1 "k8s.io/api/core/v1"
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

// PodRestorePhase represents the phase of a PodRestore.
type PodRestorePhase string

const (
	// PodRestorePending means the restore has been accepted but not yet started.
	PodRestorePending PodRestorePhase = "Pending"
	// PodRestoreRestoring means the restore is in progress.
	PodRestoreRestoring PodRestorePhase = "Restoring"
	// PodRestoreCompleted means the restore has completed successfully.
	PodRestoreCompleted PodRestorePhase = "Completed"
	// PodRestoreFailed means the restore failed.
	PodRestoreFailed PodRestorePhase = "Failed"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodRestore represents a restore operation from a PodCheckpoint.
type PodRestore struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec defines the desired restore operation.
	// +optional
	Spec PodRestoreSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// status represents the current status of the restore.
	// +optional
	Status PodRestoreStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// PodRestoreSpec defines the desired state of a PodRestore.
type PodRestoreSpec struct {
	// checkpointName is the name of the PodCheckpoint to restore from.
	// The PodCheckpoint must exist in the same namespace and be in Ready phase.
	// +required
	CheckpointName string `json:"checkpointName" protobuf:"bytes,1,opt,name=checkpointName"`

	// podTemplate defines the pod template for the restored pod.
	// The RestoreFrom field will be set automatically by the controller.
	// +required
	PodTemplate corev1.PodTemplateSpec `json:"podTemplate" protobuf:"bytes,2,opt,name=podTemplate"`
}

// PodRestoreStatus represents the current status of a PodRestore.
type PodRestoreStatus struct {
	// phase represents the current phase of the restore.
	// +optional
	Phase PodRestorePhase `json:"phase,omitempty" protobuf:"bytes,1,opt,name=phase"`

	// restoredPodName is the name of the pod that was created from the restore.
	// +optional
	RestoredPodName string `json:"restoredPodName,omitempty" protobuf:"bytes,2,opt,name=restoredPodName"`

	// message is a human-readable message indicating details about the restore.
	// +optional
	Message string `json:"message,omitempty" protobuf:"bytes,3,opt,name=message"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodRestoreList is a list of PodRestore objects.
type PodRestoreList struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of PodRestore objects.
	Items []PodRestore `json:"items" protobuf:"bytes,2,rep,name=items"`
}
