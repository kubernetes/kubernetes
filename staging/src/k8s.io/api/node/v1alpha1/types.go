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

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:defaulter-gen=false
// +k8s:prerelease-lifecycle-gen:introduced=1.12
// +k8s:prerelease-lifecycle-gen:removed=1.24

// RuntimeClass defines a class of container runtime supported in the cluster.
// The RuntimeClass is used to determine which container runtime is used to run
// all containers in a pod. RuntimeClasses are (currently) manually defined by a
// user or cluster provisioner, and referenced in the PodSpec. The Kubelet is
// responsible for resolving the RuntimeClassName reference before running the
// pod.  For more details, see
// https://git.k8s.io/enhancements/keps/sig-node/585-runtime-class
type RuntimeClass struct {
	metav1.TypeMeta `json:""`

	// metadata is the standard object metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec represents specification of the RuntimeClass
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +required
	Spec RuntimeClassSpec `json:"spec" protobuf:"bytes,2,name=spec"`
}

// RuntimeClassSpec is a specification of a RuntimeClass. It contains parameters
// that are required to describe the RuntimeClass to the Container Runtime
// Interface (CRI) implementation, as well as any other components that need to
// understand how the pod will be run. The RuntimeClassSpec is immutable.
type RuntimeClassSpec struct {
	// runtimeHandler specifies the underlying runtime and configuration that the
	// CRI implementation will use to handle pods of this class. The possible
	// values are specific to the node & CRI configuration.  It is assumed that
	// all handlers are available on every node, and handlers of the same name are
	// equivalent on every node.
	// For example, a handler called "runc" might specify that the runc OCI
	// runtime (using native Linux containers) will be used to run the containers
	// in a pod.
	// The runtimeHandler must be lowercase, conform to the DNS Label (RFC 1123)
	// requirements, and is immutable.
	// +required
	// +k8s:format="k8s-short-name"
	// +k8s:immutable
	// +k8s:required
	RuntimeHandler string `json:"runtimeHandler" protobuf:"bytes,1,opt,name=runtimeHandler"`

	// overhead represents the resource overhead associated with running a pod for a
	// given RuntimeClass. For more details, see
	// https://git.k8s.io/enhancements/keps/sig-node/688-pod-overhead/README.md
	// +optional
	Overhead *Overhead `json:"overhead,omitempty" protobuf:"bytes,2,opt,name=overhead"`

	// scheduling holds the scheduling constraints to ensure that pods running
	// with this RuntimeClass are scheduled to nodes that support it.
	// If scheduling is nil, this RuntimeClass is assumed to be supported by all
	// nodes.
	// +optional
	// +k8s:alpha(since: "1.37")=+k8s:optional
	Scheduling *Scheduling `json:"scheduling,omitempty" protobuf:"bytes,3,opt,name=scheduling"`
}

// Overhead structure represents the resource overhead associated with running a pod.
type Overhead struct {
	// podFixed represents the fixed resource overhead associated with running a pod.
	// +optional
	PodFixed corev1.ResourceList `json:"podFixed,omitempty" protobuf:"bytes,1,opt,name=podFixed,casttype=k8s.io/api/core/v1.ResourceList,castkey=k8s.io/api/core/v1.ResourceName,castvalue=k8s.io/apimachinery/pkg/api/resource.Quantity"`
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
	// +mapType=atomic
	NodeSelector map[string]string `json:"nodeSelector,omitempty" protobuf:"bytes,1,opt,name=nodeSelector"`

	// tolerations are appended (excluding duplicates) to pods running with this
	// RuntimeClass during admission, effectively unioning the set of nodes
	// tolerated by the pod and the RuntimeClass.
	// +optional
	// +listType=atomic
	// +k8s:alpha(since: "1.37")=+k8s:optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty" protobuf:"bytes,2,rep,name=tolerations"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:defaulter-gen=false
// +k8s:prerelease-lifecycle-gen:introduced=1.12
// +k8s:prerelease-lifecycle-gen:removed=1.24

// RuntimeClassList is a list of RuntimeClass objects.
type RuntimeClassList struct {
	metav1.TypeMeta `json:""`

	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of schema objects.
	Items []RuntimeClass `json:"items" protobuf:"bytes,2,rep,name=items"`
}

const (
	// PodCheckpointConditionReady is the summary condition type for a PodCheckpoint.
	// It is True once the checkpoint data has been written and is ready to
	// restore from, and False while the checkpoint is pending/in progress or
	// after a failure (distinguished by the condition's reason).
	PodCheckpointConditionReady = "Ready"
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
	// spec.sourcePod.name no longer has the pinned UID (spec.sourcePod.uid or
	// the UID previously recorded in status.sourcePodUID): the original instance
	// was replaced, so the checkpoint is failed rather than capturing a
	// different pod.
	PodCheckpointReasonSourcePodReplaced = "SourcePodReplaced"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.38
// +k8s:supportsSubresource="/status"

// PodCheckpoint represents a checkpoint of a running pod.
type PodCheckpoint struct {
	metav1.TypeMeta `json:",inline"`
	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec defines the desired checkpoint operation.
	// +required
	// +k8s:required
	Spec PodCheckpointSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// status represents the current status of the checkpoint.
	// +optional
	// +k8s:optional
	Status PodCheckpointStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// PodCheckpointSpec defines the desired state of a PodCheckpoint.
type PodCheckpointSpec struct {
	// sourcePod identifies the pod to checkpoint. The pod must exist in the
	// same namespace as the PodCheckpoint resource. Immutable.
	// +required
	// +k8s:required
	// +k8s:immutable
	SourcePod *PodReference `json:"sourcePod,omitempty" protobuf:"bytes,1,opt,name=sourcePod"`

	// timeoutSeconds is the maximum number of seconds the checkpoint operation
	// may take, between 1 and 3600. If unset, the kubelet's configured
	// checkpoint timeout is used; values larger than that configured ceiling
	// are clamped to it. The kubelet enforces the effective timeout with the
	// CRI call deadline, which bounds how long the Pod can stay frozen.
	// Immutable because the operation's deadline is fixed when it starts.
	// +optional
	// +k8s:optional
	// +k8s:immutable
	// +k8s:minimum=1
	// +k8s:maximum=3600
	TimeoutSeconds *int32 `json:"timeoutSeconds,omitempty" protobuf:"varint,2,opt,name=timeoutSeconds"`
}

// PodReference identifies a pod in the same namespace by name and, optionally,
// pins it to a single pod instance by UID.
// +structType=atomic
type PodReference struct {
	// name is the name of the pod.
	// +required
	// +k8s:required
	// +k8s:format="k8s-long-name"
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// uid pins the reference to a specific pod instance when set: the
	// checkpoint is taken only if the live pod named name has this exact UID,
	// and fails otherwise (reason SourcePodReplaced). A pod name can be reused
	// (the original pod may be deleted and a controller may recreate a new pod
	// with the same name), so a name alone does not identify an instance. This
	// matters most when the acting component is unavailable for a while (for
	// example during a leader-election change): by the time it reconciles, name
	// may resolve to a different pod. To close that window the UID must be
	// captured at creation time, so callers that need instance pinning set this
	// field when creating the PodCheckpoint.
	// +optional
	// +k8s:optional
	// +k8s:minLength=1
	UID *types.UID `json:"uid,omitempty" protobuf:"bytes,2,opt,name=uid,casttype=k8s.io/apimachinery/pkg/types.UID"`
}

// PodCheckpointStatus represents the current status of a PodCheckpoint.
type PodCheckpointStatus struct {
	// conditions represent the latest available observations of the
	// checkpoint's state. The "Ready" condition summarizes whether the
	// checkpoint data has been created and is ready to restore from; its
	// reason and message carry the detail previously exposed via phase/message.
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	// +k8s:optional
	// +k8s:alpha(since: "1.38")=+k8s:listType=map
	// +k8s:alpha(since: "1.38")=+k8s:listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`

	// nodeName is the source node assigned to execute this checkpoint. The
	// controller selects the source Pod's node before capture, and the kubelet
	// preserves this assignment when writing the outcome. A value here does not
	// imply that checkpoint data is ready; check the Ready condition.
	// +optional
	// +k8s:optional
	// +k8s:format=k8s-long-name
	NodeName *string `json:"nodeName,omitempty" protobuf:"bytes,2,opt,name=nodeName"`

	// sourcePodUID identifies the source Pod instance selected by the controller
	// when it assigns nodeName. The kubelet preserves this UID when capturing
	// the template and publishing the checkpoint outcome. If the Pod name is
	// reused for another UID, the checkpoint fails instead of capturing the
	// replacement. Set spec.sourcePod.uid at creation to pin the instance before
	// the controller assigns it.
	// +optional
	// +k8s:optional
	// +k8s:minLength=1
	SourcePodUID *types.UID `json:"sourcePodUID,omitempty" protobuf:"bytes,3,opt,name=sourcePodUID,casttype=k8s.io/apimachinery/pkg/types.UID"`

	// checkpointLocation describes where the checkpoint's data is stored. It is
	// a discriminated union over storage backends; in alpha only the node-local
	// backend (nodeLocal) is set, recording a path relative to the kubelet's
	// configured checkpoint root directory.
	// +optional
	// +k8s:optional
	CheckpointLocation *CheckpointSource `json:"checkpointLocation,omitempty" protobuf:"bytes,4,opt,name=checkpointLocation"`

	// completionTime is the time the checkpoint data became Ready, set by the
	// kubelet. Used for freshness and retention/GC.
	// Distinct from metadata.creationTimestamp (when the object was created).
	// +optional
	// +k8s:optional
	CompletionTime *metav1.Time `json:"completionTime,omitempty" protobuf:"bytes,5,opt,name=completionTime"`

	// checkpointedPodTemplate is a sanitized PodTemplateSpec (object metadata
	// plus the pod spec) captured from the source pod at checkpoint time. It is
	// the authoritative record a restore is validated against: a pod restoring
	// from this checkpoint must match this template. The kubelet sets it once;
	// subsequent status updates cannot modify or clear it, so the record used
	// for restore validation remains unchanged. Node-local and cluster-specific
	// fields (for example spec.nodeName, uid, resourceVersion, managedFields) are
	// excluded so the record stays portable. Ephemeral containers are excluded
	// because they are not checkpointed; scheduling gates are excluded because
	// they only delay scheduling. Other scheduling directives and resource
	// requests and limits are retained and must match on restore.
	// Images of checkpointed containers (regular containers and restartable
	// init containers) are recorded as resolved repository digest references
	// from the runtime. A restoring Pod must use those same image references;
	// the original image tags are not accepted in their place. Images of
	// completed non-restartable init containers are unchanged because those
	// containers are not restored.
	// +optional
	// +k8s:optional
	// +k8s:opaqueType
	// +k8s:update=NoModify
	// +k8s:update=NoUnset
	CheckpointedPodTemplate *corev1.PodTemplateSpec `json:"checkpointedPodTemplate,omitempty" protobuf:"bytes,6,opt,name=checkpointedPodTemplate"`

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
	// +k8s:optional
	// +k8s:listType=map
	// +k8s:listMapKey=name
	CheckpointedContainers []PodCheckpointContainerStatus `json:"checkpointedContainers,omitempty" protobuf:"bytes,7,rep,name=checkpointedContainers"`
}

// PodCheckpointContainerStatus identifies a container captured in a checkpoint.
type PodCheckpointContainerStatus struct {
	// name is the name of the checkpointed container.
	// +required
	// +k8s:required
	// +k8s:format=k8s-short-name
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
}

// CheckpointSource describes where a checkpoint's data is stored. Discriminated
// union: the member matching Type is set.
// +union
type CheckpointSource struct {
	// type is the storage backend holding the checkpoint data. It selects
	// which union member is set; "NodeLocal" is the only backend in alpha.
	// +unionDiscriminator
	// +required
	// +k8s:required
	// +k8s:unionDiscriminator
	Type CheckpointSourceType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=CheckpointSourceType"`
	// nodeLocal locates checkpoint data stored on the node that took the
	// checkpoint. It is set when type is "NodeLocal".
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	NodeLocal *NodeLocalCheckpointSource `json:"nodeLocal,omitempty" protobuf:"bytes,2,opt,name=nodeLocal"`
}

// +enum
// +k8s:enum
type CheckpointSourceType string

const (
	CheckpointSourceTypeNodeLocal CheckpointSourceType = "NodeLocal"
)

// NodeLocalCheckpointSource locates a checkpoint stored on the node that took it.
type NodeLocalCheckpointSource struct {
	// path is the location of the checkpoint data, relative to the kubelet's
	// configured checkpoint root directory.
	// +required
	// +k8s:required
	Path string `json:"path" protobuf:"bytes,1,opt,name=path"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.38

// PodCheckpointList is a list of PodCheckpoint objects.
type PodCheckpointList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of PodCheckpoint objects.
	Items []PodCheckpoint `json:"items" protobuf:"bytes,2,rep,name=items"`
}
