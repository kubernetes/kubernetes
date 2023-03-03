/*
Copyright 2022 The Kubernetes Authors.

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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// ResourceClaim describes which resources are needed by a resource consumer.
// Its status tracks whether the resource has been allocated and what the
// resulting attributes are.
//
// This is an alpha type and requires enabling the DynamicResourceAllocation
// feature gate.
type ResourceClaim struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec describes the desired attributes of a resource that then needs
	// to be allocated. It can only be set once when creating the
	// ResourceClaim.
	Spec ResourceClaimSpec `json:"spec" protobuf:"bytes,2,name=spec"`

	// Status describes whether the resource is available and with which
	// attributes.
	// +optional
	Status ResourceClaimStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// ResourceClaimSpec defines how a resource is to be allocated.
type ResourceClaimSpec struct {
	// ResourceClassName references the driver and additional parameters
	// via the name of a ResourceClass that was created as part of the
	// driver deployment.
	ResourceClassName string `json:"resourceClassName" protobuf:"bytes,1,name=resourceClassName"`

	// ParametersRef references a separate object with arbitrary parameters
	// that will be used by the driver when allocating a resource for the
	// claim.
	//
	// The object must be in the same namespace as the ResourceClaim.
	// +optional
	ParametersRef *ResourceClaimParametersReference `json:"parametersRef,omitempty" protobuf:"bytes,2,opt,name=parametersRef"`

	// Allocation can start immediately or when a Pod wants to use the
	// resource. "WaitForFirstConsumer" is the default.
	// +optional
	AllocationMode AllocationMode `json:"allocationMode,omitempty" protobuf:"bytes,3,opt,name=allocationMode"`
}

// AllocationMode describes whether a ResourceClaim gets allocated immediately
// when it gets created (AllocationModeImmediate) or whether allocation is
// delayed until it is needed for a Pod
// (AllocationModeWaitForFirstConsumer). Other modes might get added in the
// future.
type AllocationMode string

const (
	// When a ResourceClaim has AllocationModeWaitForFirstConsumer, allocation is
	// delayed until a Pod gets scheduled that needs the ResourceClaim. The
	// scheduler will consider all resource requirements of that Pod and
	// trigger allocation for a node that fits the Pod.
	AllocationModeWaitForFirstConsumer AllocationMode = "WaitForFirstConsumer"

	// When a ResourceClaim has AllocationModeImmediate, allocation starts
	// as soon as the ResourceClaim gets created. This is done without
	// considering the needs of Pods that will use the ResourceClaim
	// because those Pods are not known yet.
	AllocationModeImmediate AllocationMode = "Immediate"
)

// ResourceClaimStatus tracks whether the resource has been allocated and what
// the resulting attributes are.
type ResourceClaimStatus struct {
	// DriverName is a copy of the driver name from the ResourceClass at
	// the time when allocation started.
	// +optional
	DriverName string `json:"driverName,omitempty" protobuf:"bytes,1,opt,name=driverName"`

	// Allocation is set by the resource driver once a resource has been
	// allocated successfully. If this is not specified, the resource is
	// not yet allocated.
	// +optional
	Allocation *AllocationResult `json:"allocation,omitempty" protobuf:"bytes,2,opt,name=allocation"`

	// ReservedFor indicates which entities are currently allowed to use
	// the claim. A Pod which references a ResourceClaim which is not
	// reserved for that Pod will not be started.
	//
	// There can be at most 32 such reservations. This may get increased in
	// the future, but not reduced.
	//
	// +listType=map
	// +listMapKey=uid
	// +optional
	ReservedFor []ResourceClaimConsumerReference `json:"reservedFor,omitempty" protobuf:"bytes,3,opt,name=reservedFor"`

	// DeallocationRequested indicates that a ResourceClaim is to be
	// deallocated.
	//
	// The driver then must deallocate this claim and reset the field
	// together with clearing the Allocation field.
	//
	// While DeallocationRequested is set, no new consumers may be added to
	// ReservedFor.
	// +optional
	DeallocationRequested bool `json:"deallocationRequested,omitempty" protobuf:"varint,4,opt,name=deallocationRequested"`
}

// ReservedForMaxSize is the maximum number of entries in
// claim.status.reservedFor.
const ResourceClaimReservedForMaxSize = 32

// AllocationResult contains attributed of an allocated resource.
type AllocationResult struct {
	// ResourceHandle contains arbitrary data returned by the driver after a
	// successful allocation. This is opaque for
	// Kubernetes. Driver documentation may explain to users how to
	// interpret this data if needed.
	//
	// The maximum size of this field is 16KiB. This may get
	// increased in the future, but not reduced.
	// +optional
	ResourceHandle string `json:"resourceHandle,omitempty" protobuf:"bytes,1,opt,name=resourceHandle"`

	// This field will get set by the resource driver after it has
	// allocated the resource driver to inform the scheduler where it can
	// schedule Pods using the ResourceClaim.
	//
	// Setting this field is optional. If null, the resource is available
	// everywhere.
	// +optional
	AvailableOnNodes *v1.NodeSelector `json:"availableOnNodes,omitempty" protobuf:"bytes,2,opt,name=availableOnNodes"`

	// Shareable determines whether the resource supports more
	// than one consumer at a time.
	// +optional
	Shareable bool `json:"shareable,omitempty" protobuf:"varint,3,opt,name=shareable"`
}

// ResourceHandleMaxSize is the maximum size of allocation.resourceHandle.
const ResourceHandleMaxSize = 16 * 1024

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// ResourceClaimList is a collection of claims.
type ResourceClaimList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of resource claims.
	Items []ResourceClaim `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// PodScheduling objects hold information that is needed to schedule
// a Pod with ResourceClaims that use "WaitForFirstConsumer" allocation
// mode.
//
// This is an alpha type and requires enabling the DynamicResourceAllocation
// feature gate.
type PodScheduling struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec describes where resources for the Pod are needed.
	Spec PodSchedulingSpec `json:"spec" protobuf:"bytes,2,name=spec"`

	// Status describes where resources for the Pod can be allocated.
	// +optional
	Status PodSchedulingStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// PodSchedulingSpec describes where resources for the Pod are needed.
type PodSchedulingSpec struct {
	// SelectedNode is the node for which allocation of ResourceClaims that
	// are referenced by the Pod and that use "WaitForFirstConsumer"
	// allocation is to be attempted.
	// +optional
	SelectedNode string `json:"selectedNode,omitempty" protobuf:"bytes,1,opt,name=selectedNode"`

	// PotentialNodes lists nodes where the Pod might be able to run.
	//
	// The size of this field is limited to 128. This is large enough for
	// many clusters. Larger clusters may need more attempts to find a node
	// that suits all pending resources. This may get increased in the
	// future, but not reduced.
	//
	// +listType=set
	// +optional
	PotentialNodes []string `json:"potentialNodes,omitempty" protobuf:"bytes,2,opt,name=potentialNodes"`
}

// PodSchedulingStatus describes where resources for the Pod can be allocated.
type PodSchedulingStatus struct {
	// ResourceClaims describes resource availability for each
	// pod.spec.resourceClaim entry where the corresponding ResourceClaim
	// uses "WaitForFirstConsumer" allocation mode.
	//
	// +listType=map
	// +listMapKey=name
	// +optional
	ResourceClaims []ResourceClaimSchedulingStatus `json:"resourceClaims,omitempty" protobuf:"bytes,1,opt,name=resourceClaims"`

	// If there ever is a need to support other kinds of resources
	// than ResourceClaim, then new fields could get added here
	// for those other resources.
}

// ResourceClaimSchedulingStatus contains information about one particular
// ResourceClaim with "WaitForFirstConsumer" allocation mode.
type ResourceClaimSchedulingStatus struct {
	// Name matches the pod.spec.resourceClaims[*].Name field.
	// +optional
	Name string `json:"name,omitempty" protobuf:"bytes,1,opt,name=name"`

	// UnsuitableNodes lists nodes that the ResourceClaim cannot be
	// allocated for.
	//
	// The size of this field is limited to 128, the same as for
	// PodSchedulingSpec.PotentialNodes. This may get increased in the
	// future, but not reduced.
	//
	// +listType=set
	// +optional
	UnsuitableNodes []string `json:"unsuitableNodes,omitempty" protobuf:"bytes,2,opt,name=unsuitableNodes"`
}

// PodSchedulingNodeListMaxSize defines the maximum number of entries in the
// node lists that are stored in PodScheduling objects. This limit is part
// of the API.
const PodSchedulingNodeListMaxSize = 128

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// PodSchedulingList is a collection of Pod scheduling objects.
type PodSchedulingList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of PodScheduling objects.
	Items []PodScheduling `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// ResourceClass is used by administrators to influence how resources
// are allocated.
//
// This is an alpha type and requires enabling the DynamicResourceAllocation
// feature gate.
type ResourceClass struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// DriverName defines the name of the dynamic resource driver that is
	// used for allocation of a ResourceClaim that uses this class.
	//
	// Resource drivers have a unique name in forward domain order
	// (acme.example.com).
	DriverName string `json:"driverName" protobuf:"bytes,2,name=driverName"`

	// ParametersRef references an arbitrary separate object that may hold
	// parameters that will be used by the driver when allocating a
	// resource that uses this class. A dynamic resource driver can
	// distinguish between parameters stored here and and those stored in
	// ResourceClaimSpec.
	// +optional
	ParametersRef *ResourceClassParametersReference `json:"parametersRef,omitempty" protobuf:"bytes,3,opt,name=parametersRef"`

	// Only nodes matching the selector will be considered by the scheduler
	// when trying to find a Node that fits a Pod when that Pod uses
	// a ResourceClaim that has not been allocated yet.
	//
	// Setting this field is optional. If null, all nodes are candidates.
	// +optional
	SuitableNodes *v1.NodeSelector `json:"suitableNodes,omitempty" protobuf:"bytes,4,opt,name=suitableNodes"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// ResourceClassList is a collection of classes.
type ResourceClassList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of resource classes.
	Items []ResourceClass `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// ResourceClassParametersReference contains enough information to let you
// locate the parameters for a ResourceClass.
type ResourceClassParametersReference struct {
	// APIGroup is the group for the resource being referenced. It is
	// empty for the core API. This matches the group in the APIVersion
	// that is used when creating the resources.
	// +optional
	APIGroup string `json:"apiGroup,omitempty" protobuf:"bytes,1,opt,name=apiGroup"`
	// Kind is the type of resource being referenced. This is the same
	// value as in the parameter object's metadata.
	Kind string `json:"kind" protobuf:"bytes,2,name=kind"`
	// Name is the name of resource being referenced.
	Name string `json:"name" protobuf:"bytes,3,name=name"`
	// Namespace that contains the referenced resource. Must be empty
	// for cluster-scoped resources and non-empty for namespaced
	// resources.
	// +optional
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,4,opt,name=namespace"`
}

// ResourceClaimParametersReference contains enough information to let you
// locate the parameters for a ResourceClaim. The object must be in the same
// namespace as the ResourceClaim.
type ResourceClaimParametersReference struct {
	// APIGroup is the group for the resource being referenced. It is
	// empty for the core API. This matches the group in the APIVersion
	// that is used when creating the resources.
	// +optional
	APIGroup string `json:"apiGroup,omitempty" protobuf:"bytes,1,opt,name=apiGroup"`
	// Kind is the type of resource being referenced. This is the same
	// value as in the parameter object's metadata, for example "ConfigMap".
	Kind string `json:"kind" protobuf:"bytes,2,name=kind"`
	// Name is the name of resource being referenced.
	Name string `json:"name" protobuf:"bytes,3,name=name"`
}

// ResourceClaimConsumerReference contains enough information to let you
// locate the consumer of a ResourceClaim. The user must be a resource in the same
// namespace as the ResourceClaim.
type ResourceClaimConsumerReference struct {
	// APIGroup is the group for the resource being referenced. It is
	// empty for the core API. This matches the group in the APIVersion
	// that is used when creating the resources.
	// +optional
	APIGroup string `json:"apiGroup,omitempty" protobuf:"bytes,1,opt,name=apiGroup"`
	// Resource is the type of resource being referenced, for example "pods".
	Resource string `json:"resource" protobuf:"bytes,3,name=resource"`
	// Name is the name of resource being referenced.
	Name string `json:"name" protobuf:"bytes,4,name=name"`
	// UID identifies exactly one incarnation of the resource.
	UID types.UID `json:"uid" protobuf:"bytes,5,name=uid"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// ResourceClaimTemplate is used to produce ResourceClaim objects.
type ResourceClaimTemplate struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Describes the ResourceClaim that is to be generated.
	//
	// This field is immutable. A ResourceClaim will get created by the
	// control plane for a Pod when needed and then not get updated
	// anymore.
	Spec ResourceClaimTemplateSpec `json:"spec" protobuf:"bytes,2,name=spec"`
}

// ResourceClaimTemplateSpec contains the metadata and fields for a ResourceClaim.
type ResourceClaimTemplateSpec struct {
	// ObjectMeta may contain labels and annotations that will be copied into the PVC
	// when creating it. No other fields are allowed and will be rejected during
	// validation.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec for the ResourceClaim. The entire content is copied unchanged
	// into the ResourceClaim that gets created from this template. The
	// same fields as in a ResourceClaim are also valid here.
	Spec ResourceClaimSpec `json:"spec" protobuf:"bytes,2,name=spec"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// ResourceClaimTemplateList is a collection of claim templates.
type ResourceClaimTemplateList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of resource claim templates.
	Items []ResourceClaimTemplate `json:"items" protobuf:"bytes,2,rep,name=items"`
}
