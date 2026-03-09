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

package checkpoint

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// PodCheckpointPhase represents the phase of a PodCheckpoint.
type PodCheckpointPhase string

const (
	PodCheckpointPending    PodCheckpointPhase = "Pending"
	PodCheckpointInProgress PodCheckpointPhase = "InProgress"
	PodCheckpointReady      PodCheckpointPhase = "Ready"
	PodCheckpointFailed     PodCheckpointPhase = "Failed"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodCheckpoint represents a checkpoint of a running pod.
type PodCheckpoint struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// spec defines the desired checkpoint operation.
	// +optional
	Spec PodCheckpointSpec

	// status represents the current status of the checkpoint.
	// +optional
	Status PodCheckpointStatus
}

// PodCheckpointSpec defines the desired state of a PodCheckpoint.
type PodCheckpointSpec struct {
	// sourcePodName is the name of the pod to checkpoint.
	SourcePodName string
}

// PodCheckpointStatus represents the current status of a PodCheckpoint.
type PodCheckpointStatus struct {
	// phase represents the current phase of the checkpoint.
	// +optional
	Phase PodCheckpointPhase

	// nodeName is the name of the node where the checkpoint was created.
	// +optional
	NodeName string

	// checkpointLocation is the path where the checkpoint archive is stored.
	// +optional
	CheckpointLocation string

	// message is a human-readable message indicating details about the checkpoint.
	// +optional
	Message string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodCheckpointList is a list of PodCheckpoint objects.
type PodCheckpointList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []PodCheckpoint
}

// PodRestorePhase represents the phase of a PodRestore.
type PodRestorePhase string

const (
	PodRestorePending   PodRestorePhase = "Pending"
	PodRestoreRestoring PodRestorePhase = "Restoring"
	PodRestoreCompleted PodRestorePhase = "Completed"
	PodRestoreFailed    PodRestorePhase = "Failed"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodRestore represents a restore operation from a PodCheckpoint.
type PodRestore struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// spec defines the desired restore operation.
	// +optional
	Spec PodRestoreSpec

	// status represents the current status of the restore.
	// +optional
	Status PodRestoreStatus
}

// PodRestoreSpec defines the desired state of a PodRestore.
type PodRestoreSpec struct {
	// checkpointName is the name of the PodCheckpoint to restore from.
	CheckpointName string

	// podTemplate defines the pod template for the restored pod.
	PodTemplate api.PodTemplateSpec
}

// PodRestoreStatus represents the current status of a PodRestore.
type PodRestoreStatus struct {
	// phase represents the current phase of the restore.
	// +optional
	Phase PodRestorePhase

	// restoredPodName is the name of the pod created from the restore.
	// +optional
	RestoredPodName string

	// message is a human-readable message indicating details about the restore.
	// +optional
	Message string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodRestoreList is a list of PodRestore objects.
type PodRestoreList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []PodRestore
}
