/*
Copyright The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/types"
	api "k8s.io/kubernetes/pkg/apis/core"
)

const (
	// PodCheckpointReady is the summary condition type for a PodCheckpoint.
	// It is True once the checkpoint data has been written and is ready to
	// restore from, and False while the checkpoint is pending/in progress or
	// after a failure (distinguished by the condition's reason).
	PodCheckpointReady = "Ready"
)

// These are the well-known reasons used on the PodCheckpoint Ready condition.
const (
	PodCheckpointReasonPending    = "Pending"
	PodCheckpointReasonInProgress = "CheckpointInProgress"
	PodCheckpointReasonCompleted  = "CheckpointCompleted"
	PodCheckpointReasonFailed     = "CheckpointFailed"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodCheckpoint represents a checkpoint of a running pod.
type PodCheckpoint struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// spec defines the desired checkpoint operation.
	// +required
	Spec PodCheckpointSpec

	// status represents the current status of the checkpoint.
	// +optional
	Status PodCheckpointStatus
}

// PodCheckpointSpec defines the desired state of a PodCheckpoint.
type PodCheckpointSpec struct {
	// sourcePod identifies the pod to checkpoint. Required in alpha
	// (validation rejects an unset reference); optional in the schema so a
	// future selector-based or controller-populated mode can relax it.
	// Immutable.
	// +optional
	SourcePod *PodReference

	// timeoutSeconds is the maximum number of seconds the checkpoint operation
	// may take, between 1 and 3600. If unset, the kubelet's configured
	// checkpoint timeout is used; values larger than that configured ceiling
	// are clamped to it. The kubelet enforces the effective timeout with the
	// CRI call deadline, which bounds how long the Pod can stay frozen.
	// +optional
	TimeoutSeconds *int32
}

// PodReference identifies a pod in the same namespace by name and, optionally,
// pins it to a single pod instance by UID.
type PodReference struct {
	// name is the name of the pod.
	Name string

	// uid pins the reference to a specific pod instance when set: the
	// checkpoint is taken only if the live pod named name has this exact UID,
	// and fails otherwise.
	// +optional
	UID *types.UID
}

// PodCheckpointStatus represents the current status of a PodCheckpoint.
type PodCheckpointStatus struct {
	// nodeName is the name of the node where the checkpoint was created.
	// +optional
	NodeName *string

	// sourcePodUID is the UID of the pod instance the controller actually
	// checkpointed. It is recorded on the first reconcile so a later UID change
	// for the same name is detected and fails the checkpoint.
	// +optional
	SourcePodUID *types.UID

	// checkpointLocation describes where the checkpoint's data is stored. It is
	// a discriminated union over storage backends; in alpha only the node-local
	// backend (NodeLocal) is set, recording a path relative to the kubelet's
	// configured checkpoint root directory.
	// +optional
	CheckpointLocation *CheckpointSource

	// completionTime is the time the checkpoint data became Ready, set by the
	// kubelet. Used for freshness and retention/GC.
	// Distinct from metadata.creationTimestamp (when the object was created).
	// +optional
	CompletionTime *metav1.Time

	// checkpointedPodTemplate is a sanitized PodTemplateSpec (object metadata
	// plus the pod spec) captured from the source pod at checkpoint time. It is
	// the authoritative record a restore is validated against: a pod restoring
	// from this checkpoint must match this template. It is controller-written
	// and node-local/identity fields are excluded so the record stays portable.
	// +optional
	CheckpointedPodTemplate *api.PodTemplateSpec

	// checkpointedContainers lists the containers captured in the checkpoint:
	// all regular containers plus any running restartable init (sidecar)
	// containers, as a convenience for clients. Container names are unique
	// within a pod, so a single list covers both. Completed non-restartable
	// init containers are not captured; on restore they are reflected as
	// completed and not re-run. The authoritative record is
	// checkpointedPodTemplate.
	// +optional
	// +listType=map
	// +listMapKey=name
	CheckpointedContainers []PodCheckpointContainerStatus

	// conditions represent the latest available observations of the
	// checkpoint's state. The "Ready" condition summarizes whether the
	// checkpoint data has been created and is ready to restore from.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition
}

// PodCheckpointContainerStatus identifies a container captured in a checkpoint.
type PodCheckpointContainerStatus struct {
	// name is the name of the checkpointed container.
	Name string

	// image is the image the container was running at checkpoint time.
	// +optional
	Image *string
}

// CheckpointSource describes where a checkpoint's data is stored. Discriminated
// union: the member matching Type is set.
// +union
type CheckpointSource struct {
	// type is the storage backend holding the checkpoint data. It selects
	// which union member is set; "NodeLocal" is the only backend in alpha.
	// +unionDiscriminator
	Type CheckpointSourceType
	// nodeLocal locates checkpoint data stored on the node that took the
	// checkpoint. It is set when type is "NodeLocal".
	// +optional
	NodeLocal *NodeLocalCheckpointSource
}

// +enum
type CheckpointSourceType string

const (
	CheckpointSourceTypeNodeLocal CheckpointSourceType = "NodeLocal"
)

// NodeLocalCheckpointSource locates a checkpoint stored on the node that took it.
type NodeLocalCheckpointSource struct {
	// path is the location of the checkpoint data, relative to the kubelet's
	// configured checkpoint root directory.
	Path string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodCheckpointList is a list of PodCheckpoint objects.
type PodCheckpointList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []PodCheckpoint
}
