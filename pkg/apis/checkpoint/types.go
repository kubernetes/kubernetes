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
	"k8s.io/apimachinery/pkg/types"
	api "k8s.io/kubernetes/pkg/apis/core"
)

const (
	// PodCheckpointReady is the summary condition type for a PodCheckpoint.
	// It is True once the checkpoint archive has been written and is ready to
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
	// +optional
	Spec PodCheckpointSpec

	// status represents the current status of the checkpoint.
	// +optional
	Status PodCheckpointStatus
}

// PodCheckpointSpec defines the desired state of a PodCheckpoint.
type PodCheckpointSpec struct {
	// sourcePodName is the name of the pod to checkpoint. Required in alpha
	// (validation rejects an empty value); optional in the schema so a future
	// selector-based or controller-populated mode can relax it.
	// +optional
	SourcePodName string

	// sourcePodUID, if set, pins the checkpoint to a specific pod instance: the
	// controller checkpoints the pod only if the live pod named sourcePodName has
	// this exact UID, and fails the checkpoint otherwise. Immutable.
	// +optional
	SourcePodUID *types.UID

	// timeoutSeconds is the maximum number of seconds the checkpoint operation
	// may take before the container runtime aborts it. If unset or 0, the
	// container runtime default is used. The kubelet clamps this to its
	// configured checkpoint timeout ceiling (a KubeletConfiguration field),
	// which bounds how long the Pod can stay frozen.
	// +optional
	TimeoutSeconds *int32
}

// PodCheckpointStatus represents the current status of a PodCheckpoint.
type PodCheckpointStatus struct {
	// nodeName is the name of the node where the checkpoint was created.
	// +optional
	NodeName string

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

	// completionTime is the time the checkpoint completed (archive written /
	// became Ready), set by the kubelet. Used for freshness and retention/GC.
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

	// checkpointedContainers lists the regular (non-init) containers captured in
	// the checkpoint as a convenience for clients. The authoritative set is
	// recorded in checkpointedPodTemplate.
	// +optional
	// +listType=map
	// +listMapKey=name
	CheckpointedContainers []PodCheckpointContainerStatus

	// checkpointedInitContainers lists the init containers captured in the
	// checkpoint, kept separate from checkpointedContainers to mirror PodStatus.
	// It records completed non-restartable init containers and any running
	// restartable init containers (sidecars). On restore, completed init
	// containers are reflected as completed and not re-run; running sidecars are
	// restored like regular containers.
	// +optional
	// +listType=map
	// +listMapKey=name
	CheckpointedInitContainers []PodCheckpointContainerStatus

	// conditions represent the latest available observations of the
	// checkpoint's state. The "Ready" condition summarizes whether the
	// checkpoint archive has been created and is ready to restore from.
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
	Image string
}

// CheckpointSource describes where a checkpoint's data is stored. Discriminated
// union: the member matching Type is set.
// +union
type CheckpointSource struct {
	// +unionDiscriminator
	Type CheckpointSourceType
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
	// Path relative to the kubelet's configured checkpoint root directory.
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
