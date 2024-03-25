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

package v1alpha2

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
)

const (
	// Finalizer is the finalizer that gets set for claims
	// which were allocated through a builtin controller.
	Finalizer = "dra.k8s.io/delete-protection"
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

	// Allocation is set by the resource driver once a resource or set of
	// resources has been allocated successfully. If this is not specified, the
	// resources have not been allocated yet.
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
	// +patchStrategy=merge
	// +patchMergeKey=uid
	// +optional
	ReservedFor []ResourceClaimConsumerReference `json:"reservedFor,omitempty" protobuf:"bytes,3,opt,name=reservedFor" patchStrategy:"merge" patchMergeKey:"uid"`

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

// AllocationResult contains attributes of an allocated resource.
type AllocationResult struct {
	// ResourceHandles contain the state associated with an allocation that
	// should be maintained throughout the lifetime of a claim. Each
	// ResourceHandle contains data that should be passed to a specific kubelet
	// plugin once it lands on a node. This data is returned by the driver
	// after a successful allocation and is opaque to Kubernetes. Driver
	// documentation may explain to users how to interpret this data if needed.
	//
	// Setting this field is optional. It has a maximum size of 32 entries.
	// If null (or empty), it is assumed this allocation will be processed by a
	// single kubelet plugin with no ResourceHandle data attached. The name of
	// the kubelet plugin invoked will match the DriverName set in the
	// ResourceClaimStatus this AllocationResult is embedded in.
	//
	// +listType=atomic
	// +optional
	ResourceHandles []ResourceHandle `json:"resourceHandles,omitempty" protobuf:"bytes,1,opt,name=resourceHandles"`

	// This field will get set by the resource driver after it has allocated
	// the resource to inform the scheduler where it can schedule Pods using
	// the ResourceClaim.
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

// AllocationResultResourceHandlesMaxSize represents the maximum number of
// entries in allocation.resourceHandles.
const AllocationResultResourceHandlesMaxSize = 32

// ResourceHandle holds opaque resource data for processing by a specific kubelet plugin.
type ResourceHandle struct {
	// DriverName specifies the name of the resource driver whose kubelet
	// plugin should be invoked to process this ResourceHandle's data once it
	// lands on a node. This may differ from the DriverName set in
	// ResourceClaimStatus this ResourceHandle is embedded in.
	DriverName string `json:"driverName,omitempty" protobuf:"bytes,1,opt,name=driverName"`

	// Data contains the opaque data associated with this ResourceHandle. It is
	// set by the controller component of the resource driver whose name
	// matches the DriverName set in the ResourceClaimStatus this
	// ResourceHandle is embedded in. It is set at allocation time and is
	// intended for processing by the kubelet plugin whose name matches
	// the DriverName set in this ResourceHandle.
	//
	// The maximum size of this field is 16KiB. This may get increased in the
	// future, but not reduced.
	// +optional
	Data string `json:"data,omitempty" protobuf:"bytes,2,opt,name=data"`

	// If StructuredData is set, then it needs to be used instead of Data.
	//
	// +optional
	StructuredData *StructuredResourceHandle `json:"structuredData,omitempty" protobuf:"bytes,5,opt,name=structuredData"`
}

// ResourceHandleDataMaxSize represents the maximum size of resourceHandle.data.
const ResourceHandleDataMaxSize = 16 * 1024

// StructuredResourceHandle is the in-tree representation of the allocation result.
type StructuredResourceHandle struct {
	// VendorClassParameters are the per-claim configuration parameters
	// from the resource class at the time that the claim was allocated.
	//
	// +optional
	VendorClassParameters runtime.RawExtension `json:"vendorClassParameters,omitempty" protobuf:"bytes,1,opt,name=vendorClassParameters"`

	// VendorClaimParameters are the per-claim configuration parameters
	// from the resource claim parameters at the time that the claim was
	// allocated.
	//
	// +optional
	VendorClaimParameters runtime.RawExtension `json:"vendorClaimParameters,omitempty" protobuf:"bytes,2,opt,name=vendorClaimParameters"`

	// NodeName is the name of the node providing the necessary resources
	// if the resources are local to a node.
	//
	// +optional
	NodeName string `json:"nodeName,omitempty" protobuf:"bytes,4,name=nodeName"`

	// Results lists all allocated driver resources.
	//
	// +listType=atomic
	Results []DriverAllocationResult `json:"results" protobuf:"bytes,5,name=results"`
}

// DriverAllocationResult contains vendor parameters and the allocation result for
// one request.
type DriverAllocationResult struct {
	// VendorRequestParameters are the per-request configuration parameters
	// from the time that the claim was allocated.
	//
	// +optional
	VendorRequestParameters runtime.RawExtension `json:"vendorRequestParameters,omitempty" protobuf:"bytes,1,opt,name=vendorRequestParameters"`

	AllocationResultModel `json:",inline" protobuf:"bytes,2,name=allocationResultModel"`
}

// AllocationResultModel must have one and only one field set.
type AllocationResultModel struct {
	// NamedResources describes the allocation result when using the named resources model.
	//
	// +optional
	NamedResources *NamedResourcesAllocationResult `json:"namedResources,omitempty" protobuf:"bytes,1,opt,name=namedResources"`
}

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

// PodSchedulingContext objects hold information that is needed to schedule
// a Pod with ResourceClaims that use "WaitForFirstConsumer" allocation
// mode.
//
// This is an alpha type and requires enabling the DynamicResourceAllocation
// feature gate.
type PodSchedulingContext struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec describes where resources for the Pod are needed.
	Spec PodSchedulingContextSpec `json:"spec" protobuf:"bytes,2,name=spec"`

	// Status describes where resources for the Pod can be allocated.
	// +optional
	Status PodSchedulingContextStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// PodSchedulingContextSpec describes where resources for the Pod are needed.
type PodSchedulingContextSpec struct {
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
	// +listType=atomic
	// +optional
	PotentialNodes []string `json:"potentialNodes,omitempty" protobuf:"bytes,2,opt,name=potentialNodes"`
}

// PodSchedulingContextStatus describes where resources for the Pod can be allocated.
type PodSchedulingContextStatus struct {
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
	// +listType=atomic
	// +optional
	UnsuitableNodes []string `json:"unsuitableNodes,omitempty" protobuf:"bytes,2,opt,name=unsuitableNodes"`
}

// PodSchedulingNodeListMaxSize defines the maximum number of entries in the
// node lists that are stored in PodSchedulingContext objects. This limit is part
// of the API.
const PodSchedulingNodeListMaxSize = 128

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// PodSchedulingContextList is a collection of Pod scheduling objects.
type PodSchedulingContextList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of PodSchedulingContext objects.
	Items []PodSchedulingContext `json:"items" protobuf:"bytes,2,rep,name=items"`
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

	// If and only if allocation of claims using this class is handled
	// via structured parameters, then StructuredParameters must be set to true.
	// +optional
	StructuredParameters *bool `json:"structuredParameters,omitempty" protobuf:"bytes,5,opt,name=structuredParameters"`
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

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// ResourceSlice provides information about available
// resources on individual nodes.
type ResourceSlice struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// NodeName identifies the node which provides the resources
	// if they are local to a node.
	//
	// A field selector can be used to list only ResourceSlice
	// objects with a certain node name.
	//
	// +optional
	NodeName string `json:"nodeName,omitempty" protobuf:"bytes,2,opt,name=nodeName"`

	// DriverName identifies the DRA driver providing the capacity information.
	// A field selector can be used to list only ResourceSlice
	// objects with a certain driver name.
	DriverName string `json:"driverName" protobuf:"bytes,3,name=driverName"`

	ResourceModel `json:",inline" protobuf:"bytes,4,name=resourceModel"`
}

// ResourceModel must have one and only one field set.
type ResourceModel struct {
	// NamedResources describes available resources using the named resources model.
	//
	// +optional
	NamedResources *NamedResourcesResources `json:"namedResources,omitempty" protobuf:"bytes,1,opt,name=namedResources"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// ResourceSliceList is a collection of ResourceSlices.
type ResourceSliceList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of node resource capacity objects.
	Items []ResourceSlice `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// ResourceClaimParameters defines resource requests for a ResourceClaim in an
// in-tree format understood by Kubernetes.
type ResourceClaimParameters struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// If this object was created from some other resource, then this links
	// back to that resource. This field is used to find the in-tree representation
	// of the claim parameters when the parameter reference of the claim refers
	// to some unknown type.
	// +optional
	GeneratedFrom *ResourceClaimParametersReference `json:"generatedFrom,omitempty" protobuf:"bytes,2,opt,name=generatedFrom"`

	// Shareable indicates whether the allocated claim is meant to be shareable
	// by multiple consumers at the same time.
	// +optional
	Shareable bool `json:"shareable,omitempty" protobuf:"bytes,3,opt,name=shareable"`

	// DriverRequests describes all resources that are needed for the
	// allocated claim. A single claim may use resources coming from
	// different drivers. For each driver, this array has at most one
	// entry which then may have one or more per-driver requests.
	//
	// May be empty, in which case the claim can always be allocated.
	//
	// +listType=atomic
	DriverRequests []DriverRequests `json:"driverRequests,omitempty" protobuf:"bytes,4,opt,name=driverRequests"`
}

// DriverRequests describes all resources that are needed from one particular driver.
type DriverRequests struct {
	// DriverName is the name used by the DRA driver kubelet plugin.
	DriverName string `json:"driverName,omitempty" protobuf:"bytes,1,opt,name=driverName"`

	// VendorParameters are arbitrary setup parameters for all requests of the
	// claim. They are ignored while allocating the claim.
	//
	// +optional
	VendorParameters runtime.RawExtension `json:"vendorParameters,omitempty" protobuf:"bytes,2,opt,name=vendorParameters"`

	// Requests describes all resources that are needed from the driver.
	// +listType=atomic
	Requests []ResourceRequest `json:"requests,omitempty" protobuf:"bytes,3,opt,name=requests"`
}

// ResourceRequest is a request for resources from one particular driver.
type ResourceRequest struct {
	// VendorParameters are arbitrary setup parameters for the requested
	// resource. They are ignored while allocating a claim.
	//
	// +optional
	VendorParameters runtime.RawExtension `json:"vendorParameters,omitempty" protobuf:"bytes,1,opt,name=vendorParameters"`

	ResourceRequestModel `json:",inline" protobuf:"bytes,2,name=resourceRequestModel"`
}

// ResourceRequestModel must have one and only one field set.
type ResourceRequestModel struct {
	// NamedResources describes a request for resources with the named resources model.
	//
	// +optional
	NamedResources *NamedResourcesRequest `json:"namedResources,omitempty" protobuf:"bytes,1,opt,name=namedResources"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// ResourceClaimParametersList is a collection of ResourceClaimParameters.
type ResourceClaimParametersList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of node resource capacity objects.
	Items []ResourceClaimParameters `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// ResourceClassParameters defines resource requests for a ResourceClass in an
// in-tree format understood by Kubernetes.
type ResourceClassParameters struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// If this object was created from some other resource, then this links
	// back to that resource. This field is used to find the in-tree representation
	// of the class parameters when the parameter reference of the class refers
	// to some unknown type.
	// +optional
	GeneratedFrom *ResourceClassParametersReference `json:"generatedFrom,omitempty" protobuf:"bytes,2,opt,name=generatedFrom"`

	// VendorParameters are arbitrary setup parameters for all claims using
	// this class. They are ignored while allocating the claim. There must
	// not be more than one entry per driver.
	//
	// +listType=atomic
	// +optional
	VendorParameters []VendorParameters `json:"vendorParameters,omitempty" protobuf:"bytes,3,opt,name=vendorParameters"`

	// Filters describes additional contraints that must be met when using the class.
	//
	// +listType=atomic
	Filters []ResourceFilter `json:"filters,omitempty" protobuf:"bytes,4,opt,name=filters"`
}

// ResourceFilter is a filter for resources from one particular driver.
type ResourceFilter struct {
	// DriverName is the name used by the DRA driver kubelet plugin.
	DriverName string `json:"driverName,omitempty" protobuf:"bytes,1,opt,name=driverName"`

	ResourceFilterModel `json:",inline" protobuf:"bytes,2,name=resourceFilterModel"`
}

// ResourceFilterModel must have one and only one field set.
type ResourceFilterModel struct {
	// NamedResources describes a resource filter using the named resources model.
	//
	// +optional
	NamedResources *NamedResourcesFilter `json:"namedResources,omitempty" protobuf:"bytes,1,opt,name=namedResources"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// ResourceClassParametersList is a collection of ResourceClassParameters.
type ResourceClassParametersList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of node resource capacity objects.
	Items []ResourceClassParameters `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// VendorParameters are opaque parameters for one particular driver.
type VendorParameters struct {
	// DriverName is the name used by the DRA driver kubelet plugin.
	DriverName string `json:"driverName,omitempty" protobuf:"bytes,1,opt,name=driverName"`

	// Parameters can be arbitrary setup parameters. They are ignored while
	// allocating a claim.
	//
	// +optional
	Parameters runtime.RawExtension `json:"parameters,omitempty" protobuf:"bytes,2,opt,name=parameters"`
}
