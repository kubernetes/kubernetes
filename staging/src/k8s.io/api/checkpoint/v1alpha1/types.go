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
	"k8s.io/apimachinery/pkg/types"
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
	// PodCheckpointReasonPending is set while the checkpoint has been accepted
	// but not yet started.
	PodCheckpointReasonPending = "Pending"
	// PodCheckpointReasonInProgress is set while the checkpoint is being created.
	PodCheckpointReasonInProgress = "CheckpointInProgress"
	// PodCheckpointReasonCompleted is set once the checkpoint has been
	// successfully created (Ready=True).
	PodCheckpointReasonCompleted = "CheckpointCompleted"
	// PodCheckpointReasonFailed is set when checkpoint creation failed.
	PodCheckpointReasonFailed = "CheckpointFailed"
	// PodCheckpointReasonSourcePodReplaced is set when the live pod named by
	// spec.sourcePodName no longer has the pinned UID (spec.sourcePodUID or the
	// UID previously recorded in status.sourcePodUID): the original instance was
	// replaced, so the checkpoint is failed rather than capturing a different pod.
	PodCheckpointReasonSourcePodReplaced = "SourcePodReplaced"
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
	// exist in the same namespace as the PodCheckpoint resource. Required in
	// alpha (validation rejects an empty value); it is marked optional in the
	// schema so a future selector-based or controller-populated mode can relax it
	// without an incompatible API change.
	// +optional
	SourcePodName string `json:"sourcePodName,omitempty" protobuf:"bytes,1,opt,name=sourcePodName"`

	// sourcePodUID, if set, pins the checkpoint to a specific pod instance: the
	// controller checkpoints the pod only if the live pod named sourcePodName has
	// this exact UID, and fails the checkpoint otherwise (reason
	// SourcePodReplaced). A pod name can be reused (the original pod may be
	// deleted and a controller may recreate a new pod with the same name), so a
	// name alone does not identify an instance. This matters most when the
	// controller is unavailable for a while (for example during a leader-election
	// change): by the time it reconciles, sourcePodName may resolve to a
	// different pod. To close that window the UID must be captured at creation
	// time, so callers that need instance pinning set this field when creating
	// the PodCheckpoint. Immutable.
	// +optional
	SourcePodUID *types.UID `json:"sourcePodUID,omitempty" protobuf:"bytes,4,opt,name=sourcePodUID,casttype=k8s.io/apimachinery/pkg/types.UID"`

	// timeoutSeconds is the maximum number of seconds the checkpoint operation
	// may take before the container runtime aborts it. If unset or 0, the
	// container runtime default is used. The kubelet clamps this to its
	// configured checkpoint timeout ceiling (a KubeletConfiguration field),
	// which bounds how long the Pod can stay frozen.
	// +optional
	TimeoutSeconds *int32 `json:"timeoutSeconds,omitempty" protobuf:"varint,3,opt,name=timeoutSeconds"`
}

// PodCheckpointStatus represents the current status of a PodCheckpoint.
type PodCheckpointStatus struct {
	// nodeName is the name of the node where the checkpoint was created.
	// +optional
	NodeName string `json:"nodeName,omitempty" protobuf:"bytes,2,opt,name=nodeName"`

	// sourcePodUID is the UID of the pod instance the controller actually
	// checkpointed (or is checkpointing). It is recorded on the first reconcile
	// for visibility and so that a later UID change for the same name is detected
	// and fails the checkpoint. This guards only changes observed after the first
	// reconcile; to also cover a controller that was down for the entire window,
	// set spec.sourcePodUID at creation time.
	// +optional
	SourcePodUID *types.UID `json:"sourcePodUID,omitempty" protobuf:"bytes,7,opt,name=sourcePodUID,casttype=k8s.io/apimachinery/pkg/types.UID"`

	// checkpointLocation describes where the checkpoint's data is stored. It is
	// a discriminated union over storage backends; in alpha only the node-local
	// backend (nodeLocal) is set, recording a path relative to the kubelet's
	// configured checkpoint root directory.
	// +optional
	CheckpointLocation *CheckpointSource `json:"checkpointLocation,omitempty" protobuf:"bytes,3,opt,name=checkpointLocation"`

	// completionTime is the time the checkpoint completed (archive written /
	// became Ready), set by the kubelet. Used for freshness and retention/GC.
	// Distinct from metadata.creationTimestamp (when the object was created).
	// +optional
	CompletionTime *metav1.Time `json:"completionTime,omitempty" protobuf:"bytes,9,opt,name=completionTime"`

	// checkpointedPodTemplate is a sanitized PodTemplateSpec (object metadata
	// plus the pod spec) captured from the source pod at checkpoint time. It is
	// the authoritative record a restore is validated against: a pod restoring
	// from this checkpoint must match this template. Because it is part of
	// status it is controller-written and immutable to users, which makes it a
	// tamper-proof anchor for the equality check. Node-local and
	// cluster-specific fields (for example spec.nodeName, uid, resourceVersion,
	// managedFields) are excluded so the record stays portable.
	// +optional
	CheckpointedPodTemplate *corev1.PodTemplateSpec `json:"checkpointedPodTemplate,omitempty" protobuf:"bytes,5,opt,name=checkpointedPodTemplate"`

	// checkpointedContainers lists the regular (non-init) containers captured in
	// the checkpoint as a convenience for clients. The authoritative set is
	// recorded in checkpointedPodTemplate.
	// +optional
	// +listType=map
	// +listMapKey=name
	CheckpointedContainers []PodCheckpointContainerStatus `json:"checkpointedContainers,omitempty" protobuf:"bytes,6,rep,name=checkpointedContainers"`

	// checkpointedInitContainers lists the init containers captured in the
	// checkpoint, kept separate from checkpointedContainers to mirror PodStatus.
	// It records completed non-restartable init containers and any running
	// restartable init containers (sidecars). On restore, completed init
	// containers are reflected as completed and not re-run; running sidecars are
	// restored like regular containers.
	// +optional
	// +listType=map
	// +listMapKey=name
	CheckpointedInitContainers []PodCheckpointContainerStatus `json:"checkpointedInitContainers,omitempty" protobuf:"bytes,8,rep,name=checkpointedInitContainers"`

	// conditions represent the latest available observations of the
	// checkpoint's state. The "Ready" condition summarizes whether the
	// checkpoint archive has been created and is ready to restore from; its
	// reason and message carry the detail previously exposed via phase/message.
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,4,rep,name=conditions"`
}

// PodCheckpointContainerStatus identifies a container captured in a checkpoint.
type PodCheckpointContainerStatus struct {
	// name is the name of the checkpointed container.
	// +required
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// image is the image the container was running at checkpoint time.
	// +optional
	Image string `json:"image,omitempty" protobuf:"bytes,2,opt,name=image"`
}

// CheckpointSource describes where a checkpoint's data is stored. Discriminated
// union: the member matching Type is set.
// +union
type CheckpointSource struct {
	// +unionDiscriminator
	Type CheckpointSourceType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=CheckpointSourceType"`
	// +optional
	NodeLocal *NodeLocalCheckpointSource `json:"nodeLocal,omitempty" protobuf:"bytes,2,opt,name=nodeLocal"`
}

// +enum
type CheckpointSourceType string

const (
	CheckpointSourceTypeNodeLocal CheckpointSourceType = "NodeLocal"
)

// NodeLocalCheckpointSource locates a checkpoint stored on the node that took it.
type NodeLocalCheckpointSource struct {
	// Path relative to the kubelet's configured checkpoint root directory.
	Path string `json:"path" protobuf:"bytes,1,opt,name=path"`
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
