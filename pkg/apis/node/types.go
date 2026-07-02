/*
Copyright 2018 The Kubernetes Authors.

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

package node

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/apis/core"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RuntimeClass defines a class of container runtime supported in the cluster.
// The RuntimeClass is used to determine which container runtime is used to run
// all containers in a pod. RuntimeClasses are (currently) manually defined by a
// user or cluster provisioner, and referenced in the PodSpec. The Kubelet is
// responsible for resolving the RuntimeClassName reference before running the
// pod.  For more details, see
// https://git.k8s.io/enhancements/keps/sig-node/585-runtime-class/README.md
type RuntimeClass struct {
	metav1.TypeMeta

	// +optional
	metav1.ObjectMeta

	// handler specifies the underlying runtime and configuration that the CRI
	// implementation will use to handle pods of this class. The possible values
	// are specific to the node & CRI configuration.  It is assumed that all
	// handlers are available on every node, and handlers of the same name are
	// equivalent on every node.
	// For example, a handler called "runc" might specify that the runc OCI
	// runtime (using native Linux containers) will be used to run the containers
	// in a pod.
	// The Handler must conform to the DNS Label (RFC 1123) requirements, and is
	// immutable.
	Handler string

	// overhead represents the resource overhead associated with running a pod for a
	// given RuntimeClass. For more details, see
	// https://git.k8s.io/enhancements/keps/sig-network/580-pod-readiness-gates
	// +optional
	Overhead *Overhead

	// scheduling holds the scheduling constraints to ensure that pods running
	// with this RuntimeClass are scheduled to nodes that support it.
	// If scheduling is nil, this RuntimeClass is assumed to be supported by all
	// nodes.
	// +optional
	Scheduling *Scheduling

	// podCheckpointOptions permits user-supplied checkpoint option keys for this
	// runtime handler. An unset policy permits no user options.
	// +optional
	PodCheckpointOptions *RuntimeClassPodCheckpointOptions
}

// RuntimeClassPodCheckpointOptions configures the runtime-specific option keys users
// may supply. Administrator configuration remains separate in the runtime.
type RuntimeClassPodCheckpointOptions struct {
	// allowedCheckpointOptions lists keys permitted in checkpoint requests.
	// +optional
	AllowedCheckpointOptions []string
}

// Overhead structure represents the resource overhead associated with running a pod.
type Overhead struct {
	// podFixed represents the fixed resource overhead associated with running a pod.
	// +optional
	PodFixed core.ResourceList
}

// Scheduling specifies the scheduling constraints for nodes supporting a
// RuntimeClass.
type Scheduling struct {
	// nodeSelector lists labels that must be present on nodes that support this
	// RuntimeClass. Pods using this RuntimeClass can only be scheduled to a
	// node matched by this selector. The RuntimeClass nodeSelector is merged
	// with a pod's existing nodeSelector. Any conflicts will cause the pod to
	// be rejected in admission.
	// +optional
	NodeSelector map[string]string

	// tolerations are appended (excluding duplicates) to pods running with this
	// RuntimeClass during admission, effectively unioning the set of nodes
	// tolerated by the pod and the RuntimeClass.
	// +optional
	Tolerations []core.Toleration
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RuntimeClassList is a list of RuntimeClass objects.
type RuntimeClassList struct {
	metav1.TypeMeta

	// +optional
	metav1.ListMeta

	// items is a list of schema objects.
	Items []RuntimeClass
}

// PodCheckpointRestoreProtectionFinalizer retains a deleting checkpoint while
// an admitted Pod still depends on it for restore.
const PodCheckpointRestoreProtectionFinalizer = "node.k8s.io/restore-lock"

const (
	// PodCheckpointConditionReady is the summary condition type for a PodCheckpoint.
	// It is True once the checkpoint data has been written and is ready to
	// restore from, and False while the checkpoint is pending/in progress or
	// after a failure (distinguished by the condition's reason).
	PodCheckpointConditionReady = "Ready"
)

// These are the well-known reasons used on the PodCheckpoint Ready condition.
const (
	PodCheckpointReasonPending           = "Pending"
	PodCheckpointReasonInProgress        = "CheckpointInProgress"
	PodCheckpointReasonCompleted         = "CheckpointCompleted"
	PodCheckpointReasonFailed            = "CheckpointFailed"
	PodCheckpointReasonSourcePodReplaced = "SourcePodReplaced"
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
	// sourcePod identifies the pod to checkpoint. The pod must exist in the
	// same namespace as the PodCheckpoint. Required and immutable.
	// +optional
	SourcePod *PodReference

	// timeoutSeconds is the maximum number of seconds the checkpoint operation
	// may take, between 1 and 3600. If unset, the kubelet's configured
	// checkpoint timeout is used; values larger than that configured ceiling
	// are clamped to it. The kubelet enforces the effective timeout with the
	// CRI call deadline, which bounds how long the Pod can stay frozen.
	// Immutable because the operation's deadline is fixed when it starts.
	// +optional
	TimeoutSeconds *int32

	// checkpointOptions contains opaque runtime-specific options for this
	// checkpoint operation. Empty options use runtime defaults. Keys require the
	// source Pod's RuntimeClass checkpoint allowlist, and runtimes validate safe
	// values as untrusted user input. Administrator settings belong in node or
	// runtime configuration. They are not reused as restore options. If an option
	// creates a restore requirement, the runtime records it in the checkpoint
	// data it owns.
	// At most 64 entries are allowed, with keys of at most 256 bytes and values
	// of at most 4096 bytes.
	// +optional
	// +mapType=atomic
	CheckpointOptions map[string]string
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
	// conditions represent the latest available observations of the
	// checkpoint's state. The "Ready" condition summarizes whether the
	// checkpoint data has been created and is ready to restore from.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition

	// nodeName is the source node assigned to execute this checkpoint. The
	// controller selects the source Pod's node before capture, and the kubelet
	// preserves this assignment when writing the outcome. A value here does not
	// imply that checkpoint data is ready; check the Ready condition.
	// +optional
	NodeName *string

	// sourcePodUID identifies the source Pod instance selected by the controller
	// when it assigns nodeName. The kubelet preserves this UID when capturing
	// the template and publishing the checkpoint outcome. If the Pod name is
	// reused for another UID, the checkpoint fails instead of capturing the
	// replacement. Set spec.sourcePod.uid at creation to pin the instance before
	// the controller assigns it.
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
	// from this checkpoint must match this template. The kubelet sets it once;
	// subsequent status updates cannot modify or clear it. Node-local/identity
	// fields, ephemeral containers, and scheduling gates are excluded. Other
	// scheduling directives and resource requests and limits must match on restore.
	// Images of regular and restartable init containers are resolved repository
	// digest references from the runtime; restoring Pods must use the same
	// references. Completed non-restartable init containers are not restored,
	// so their image references are unchanged.
	// +optional
	CheckpointedPodTemplate *core.PodTemplateSpec

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
}

// PodCheckpointContainerStatus identifies a container captured in a checkpoint.
type PodCheckpointContainerStatus struct {
	// name is the name of the checkpointed container.
	Name string
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
