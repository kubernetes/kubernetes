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

package v1alpha3

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
)

const (
	// Finalizer is the finalizer that gets set for claims
	// which were allocated through a builtin controller.
	// Reserved for use by Kubernetes, DRA driver controllers must
	// use their own finalizer.
	Finalizer = "dra.k8s.io/delete-protection"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.31

// One or more slices represent a pool of devices managed by a given driver.
// How many slices the driver uses to publish that pool is driver-specific.
// Each device in a given pool must have a unique name.
//
// The slice in which a device gets published may change over time. The unique identifier
// for a device is the tuple `<driver name>/<pool name>/<device name>`. Driver name
// and device name don't contain slashes, so it is okay to concatenate them
// like this in a string with a slash as separator. The pool name itself may contain
// additional slashes.
//
// Whenever a driver needs to update a pool, it bumps the pool generation number
// and updates all slices with that new number and any new device definitions. A consumer
// must only use device definitions from slices with the highest generation number
// and ignore all others.
//
// If necessary, a consumer can check the number of total devices in a pool (included
// in each slice) to determine whether its view of a pool is complete.
//
// For devices that are not local to a node, the node name is not set. Instead,
// the driver may use a node selector to specify where the devices are available.
type ResourceSlice struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Contains the information published by the driver.
	//
	// Changing the spec bumps up the generation number.
	Spec ResourceSliceSpec `json:"spec" protobuf:"bytes,2,name=spec"`

	// Future extension: status.
}

// ResourceSliceSpec contains the information published by the driver in one ResourceSlice.
type ResourceSliceSpec struct {
	// DriverName identifies the DRA driver providing the capacity information.
	// A field selector can be used to list only ResourceSlice
	// objects with a certain driver name.
	//
	// Must be a DNS subdomain and should end with a DNS domain owned by the
	// vendor of the driver.
	//
	// +required
	DriverName string `json:"driverName" protobuf:"bytes,1,name=driverName"`

	// PoolName is used to identify devices. For node-local devices, this
	// is often the node name, but this is not required.
	//
	// It must not be longer than 253 and must consist of one or more DNS sub-domains
	// separated by slashes.
	//
	// +required
	PoolName string `json:"poolName" protobuf:"bytes,2,name=poolName"`

	// NodeName identifies the node which provides the devices.
	// A field selector can be used to list only ResourceSlice
	// objects belonging to a certain node.
	//
	// This field can be used to limit access from nodes to slices with
	// the same node name. It also indicates to autoscalers that adding
	// new nodes of the same type as some old node might also make new
	// devices available.
	//
	// NodeName and NodeSelector are mutually exclusive. One of them
	// must be set.
	//
	// +optional
	NodeName *string `json:"nodeName,omitempty" protobuf:"bytes,3,opt,name=nodeName"`

	// Defines which nodes have access to the devices in the pool.
	// If the node selector is empty, all nodes have access.
	//
	// NodeName and NodeSelector are mutually exclusive. One of them
	// must be set.
	//
	// +optional
	NodeSelector *v1.NodeSelector `json:"nodeSelector,omitempty" protobuf:"bytes,4,opt,name=nodeSelector"`

	// The generation gets bumped in all slices of a pool whenever device
	// definitions change. A consumer must only use device definitions from slices
	// with the highest generation number and ignore all others.
	PoolGeneration int64 `json:"poolGeneration" protobuf:"bytes,5,name=poolGeneration"`

	// The total number of slices in the pool.
	// Consumers can use this to check whether they have
	// seen all slices.
	PoolSliceCount int64 `json:"poolSliceCount" protobuf:"bytes,6,name=poolSliceCount"`

	// Devices lists all available devices in this pool.
	//
	// Must not have more than 128 entries.
	//
	// +required
	// +listType=atomic
	Devices []Device `json:"devices" protobuf:"bytes,7,name=devices"`

	// FUTURE EXTENSION: some other kind of list, should we ever need it.
	// Old clients seeing an empty Devices field can safely ignore the (to
	// them) empty pool.
}

const ResourceSliceMaxSharedCapacity = 128
const ResourceSliceMaxDevices = 128
const PoolNameMaxLength = validation.DNS1123SubdomainMaxLength // Same as for a single node name.

// Device represents one individual hardware instance that can be selected based
// on its attributes.
type Device struct {
	// Name is unique identifier among all devices managed by
	// the driver in the pool. It must be a DNS label.
	//
	// +required
	Name string `json:"name" protobuf:"bytes,1,name=name"`

	// Attributes defines the set of attributes for this device.
	// The name of each attribute must be unique in that set.
	//
	// The maximum number of attributes and capacities is 32.
	//
	// +optional
	// +listType=atomic
	Attributes []DeviceAttribute `json:"attributes,omitempty" protobuf:"bytes,2,rep,name=attributes"`

	// Capacities defines the set of capacities for this device.
	// The name of each capacity must be unique in that set.
	//
	// The maximum number of attributes and capacities is 32.
	//
	// +optional
	// +listType=atomic
	Capacities []DeviceCapacity `json:"capacities,omitempty" protobuf:"bytes,3,rep,name=capacities"`
}

// Limit for the sum of the number of entries in both slices.
const ResourceSliceMaxAttributesAndCapacitiesPerDevice = 32

// DeviceAttribute is a combination of an attribute name and its value.
// Exactly one value must be set.
type DeviceAttribute struct {
	// Name is a unique identifier for this attribute, which will be
	// referenced when selecting devices.
	//
	// Attributes are defined either by the owner of the specific driver
	// (usually the vendor) or by some 3rd party (e.g. the Kubernetes
	// project). Because attributes are sometimes compared across devices,
	// a given name is expected to mean the same thing and have the same
	// type on all devices.
	//
	// Attribute names must be either a DNS label
	// (e.g. "theName") or a DNS subdomain followed by a slash ("/")
	// followed by a DNS label
	// (e.g. "example.com/theName"). Attributes whose name do not
	// include the domain prefix are assumed to be part of the driver's
	// domain. Attributes defined by 3rd parties must include the domain
	// prefix.
	//
	// The maximum length for the DNS subdomain is 63 characters (same as
	// for driver names) and the maximum length of the DNS label identifier
	// is 32.
	//
	// +required
	Name string `json:"name" protobuf:"bytes,1,name=name"`

	// The Go field names below have a Value suffix to avoid a conflict between the
	// field "String" and the corresponding method. That method is required.
	// The Kubernetes API is defined without that suffix to keep it more natural.

	// IntValue is a number.
	//
	// +optional
	IntValue *int64 `json:"int,omitempty" protobuf:"varint,2,opt,name=int"`

	// BoolValue is a true/false value.
	//
	// +optional
	BoolValue *bool `json:"bool,omitempty" protobuf:"varint,3,opt,name=bool"`

	// StringValue is a string. Must not be longer than 64 characters.
	//
	// +optional
	StringValue *string `json:"string,omitempty" protobuf:"bytes,4,opt,name=string"`

	// VersionValue is a semantic version according to semver.org spec 2.0.0.
	// Must not be longer than 64 characters.
	//
	// +optional
	VersionValue *string `json:"version,omitempty" protobuf:"bytes,5,opt,name=version"`
}

// DeviceCapacity is a combination of a capacity name and its value.
// Exactly one value must be set.
type DeviceCapacity struct {
	// Name is a unique identifier for this capacity, which will be
	// referenced when selecting devices.
	//
	// Capacities are defined either by the owner of the specific driver
	// (usually the vendor) or by some 3rd party (e.g. the Kubernetes
	// project). Because capacities are sometimes compared across devices,
	// a given name is expected to mean the same thing and have the same
	// type on all devices.
	//
	// Capacity names must be either a DNS label
	// (e.g. "theName") or a DNS subdomain followed by a slash ("/")
	// followed by a DNS label
	// (e.g. "example.com/theName"). Capacities whose name do not
	// include the domain prefix are assumed to be part of the driver's
	// domain. Capacities defined by 3rd parties must include the domain
	// prefix.
	//
	// The maximum length for the DNS subdomain is 63 characters (same as
	// for driver names) and the maximum length of the DNS label identifier
	// is 32.
	//
	// +required
	Name string `json:"name" protobuf:"bytes,1,name=name"`

	// Quantity determines the size of the capacity.
	//
	// +required
	Quantity *resource.Quantity `json:"quantity,omitempty" protobuf:"bytes,2,opt,name=quantity"`
}

// DeviceMaxIDLength is the maximum length of the identifier in a device attribute or capacity name (`<domain>/<ID>`).
const DeviceMaxIDLength = 32

// DeviceAttributeMaxValueLength is the maximum length of a string or version attribute value.
const DeviceAttributeMaxValueLength = 64

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.31

// ResourceSliceList is a collection of slices.
type ResourceSliceList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// +optional
	metav1.ListMeta `json:"listMeta" protobuf:"bytes,1,opt,name=listMeta"`

	// Items is the list of resource slices.
	Items []ResourceSlice `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// ResourceClaim describes which resources (typically one or more devices)
// are needed by a claim consumer.
// Its status tracks whether the claim has been allocated and what the
// resulting attributes are.
//
// This is an alpha type and requires enabling the DynamicResourceAllocation
// feature gate.
type ResourceClaim struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec defines what to allocated and how to configure it.
	// The spec is immutable.
	Spec ResourceClaimSpec `json:"spec" protobuf:"bytes,2,name=spec"`

	// Status describes whether the claim is ready for use.
	// +optional
	Status ResourceClaimStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// ResourceClaimSpec defines how a resource is to be allocated.
type ResourceClaimSpec struct {
	// Requests are individual requests for separate resources for the claim.
	// An empty list is valid and means that the claim can always be allocated
	// without needing anything. A class can be referenced to use the default
	// requests from that class.
	//
	// +required
	// +listType=atomic
	Requests []Request `json:"requests" protobuf:"bytes,1,name=requests"`

	// These constraints must be satisfied by the set of devices that get
	// allocated for the claim.
	//
	// +optional
	// +listType=atomic
	Constraints []Constraint `json:"constraints,omitempty" protobuf:"bytes,2,opt,name=constraints"`

	// This field holds configuration for multiple potential drivers which
	// could satisfy requests in this claim. It is ignored while allocating
	// the claim.
	//
	// +optional
	// +listType=atomic
	Config []ClaimConfiguration `json:"config,omitempty" protobuf:"bytes,3,opt,name=config"`

	// ControllerName defines the name of the DRA driver that is meant
	// to handle allocation of this claim. If empty, allocation is handled
	// by the scheduler while scheduling a pod.
	//
	// Must be a DNS subdomain and should end with a DNS domain owned by the
	// vendor of the driver.
	//
	// This is an alpha field and requires enabling the DRAControlPlaneController
	// feature gate.
	//
	// +optional
	ControllerName *string `json:"controllerName,omitempty" protobuf:"bytes,4,opt,name=controllerName"`

	// Future extension, ignored by older schedulers. This is fine because
	// scoring allows users to define a preference, without making it a
	// hard requirement.
	//
	// Score *SomeScoringStruct
}

// Request is a request for one of many resources required for a claim.
// This is typically a request for a single resource like a device, but can
// also ask for several identical devices. It might get extended to support
// asking for one of several different alternatives.
type Request struct {
	// The name can be used to reference this request in a pod.spec.containers[].resources.claims
	// entry and in a constraint of the claim.
	//
	// Must be a DNS label.
	Name string `json:"name" protobuf:"bytes,1,name=name"`

	*RequestDetail `json:",inline" protobuf:"bytes,2,name=requestDetail"`

	// FUTURE EXTENSION:
	//
	// OneOf contains a list of requests, only one of which must be satisfied.
	// Requests are listed in order of priority.
	//
	// +optional
	// +listType=atomic
	// OneOf []RequestDetail
}

// RequestDetail is embedded inside Request. Exactly one field must be set.
type RequestDetail struct {
	// Device requests one or more devices.
	//
	// +required
	Device *DeviceRequest `json:"device,omitempty" protobuf:"bytes,1,opt,name=device"`
}

// DeviceRequest is currently the only permitted alternative in RequestDetail.
type DeviceRequest struct {
	// By referencing a DeviceClass, a request inherits additional
	// configuration parameters and selectors.
	//
	// A class is required. Which classes are available depends on the cluster.
	//
	// Administrators may use this to restrict which devices may get
	// requested by only installing classes with selectors for permitted
	// devices. If users are free to request anything without restrictions,
	// then an empty class called "none" can get created to permit
	// `deviceClassName: none`.
	//
	// +required
	DeviceClassName string `json:"deviceClassName" protobuf:"bytes,1,name=deviceClassName"`

	// Each selector must be satisfied by a device which is requested.
	//
	// +optional
	// +listType=atomic
	Selectors []Selector `json:"selectors,omitempty" protobuf:"bytes,2,name=selectors"`

	// The count mode together with, for some modes, additional fields
	// determines how many devices to allocate for the request.
	//
	// The default if unset is exactly one device:
	//     countMode: Exact
	//     count: 1
	//
	// "countMode: All" asks for all devices matching the selectors.
	// Allocation fails if not all of them are available, unless admin
	// access is requested. Admin access is granted also for
	// devices which are in use.
	//
	// More modes may get added in the future.
	//
	// +default
	CountMode string `json:"countMode,omitempty" protobuf:"bytes,3,opt,name=countMode"`

	// Count is used only when the count mode is "Exact". Must be larger than zero.
	//
	// +optional
	Count *int64 `json:"count,omitempty" protobuf:"bytes,4,opt,name=count"`

	// AdminAccess indicates that this is a claim for administrative access
	// to the device(s). Claims with AdminAccess are expected to be used for
	// monitoring or other management services for a device.  They ignore
	// all ordinary claims to the device with respect to access modes and
	// any resource allocations. Ability to request this kind of access is
	// controlled via ResourceQuota in the resource.k8s.io API.
	//
	// Default is false.
	//
	// +optional
	AdminAccess *bool `json:"adminAccess,omitempty" protobuf:"bytes,5,opt,name=adminAccess"`
}

// Valid [DeviceRequest.CountMode] values.
const (
	CountModeExact = "Exact"
	CountModeAll   = "All"
)

// Exactly one field must be set.
type Selector struct {
	// CEL contains a CEL expression for selecting a device.
	//
	// +required
	CEL *CELSelector `json:"cel,omitempty" protobuf:"bytes,1,opt,name=cel"`
}

// CELSelector contains a CEL expression for selecting a device.
type CELSelector struct {
	// This CEL expression must evaluate to true if a device is suitable.
	// This covers qualitative aspects of device selection.
	//
	// The language is as defined in
	// https://kubernetes.io/docs/reference/using-api/cel/
	// with several additions that are specific to device selectors.
	//
	// Attributes of a device are made available through a nested
	// `device.attributes` map with the domain part of the attribute name
	// as key in the outer map and the identifier as key in the inner
	// map. All identifiers can be used in a field lookup:
	//
	//    device.attributes["dra.example.com"].driverVersion
	//
	// The type of each entry varies, depending on the attribute
	// that is being looked up. The domain lookup returns an empty
	// map if there is no attribute with that domain. However,
	// unknown identifiers then trigger a runtime error.
	//
	// The `cel.bind` function is enabled and can be used to simplify
	// expressions that access multiple attributes with the same domain:
	//
	//    cel.bind(dra, device.attributes["dra.example.com"], dra.someBool && dra.anotherBool)
	//
	// Capacities associated with a device are made available through a
	// nested `device.capacities` map the same way as attributes.
	//
	// The `device.driverName` string variable can be used to check for a specific
	// driver explicitly in a filter that is meant to work for devices from
	// different vendors. It is provided by Kubernetes and matches the
	// `driverName` from the ResourceSlice which provides the device.
	//
	// The CEL expression is applied to *all* available devices from any driver.
	// The expression has to check for existence of an attribute when it is not
	// certain that it is provided because runtime errors are not automatically
	// treated as "don't select device". Instead, device selection fails completely
	// and reports the error.
	//
	// Some more examples:
	//
	//    "memory" in device.capacities["dra.example.com"] && # Is the capacity available?
	//       device.capacities["dra.example.com"].memory.isGreaterThan(quantity("1Gi")) # >= 1Gi
	//
	//    device.attributes["dra.example.com"].driverVersion.isGreaterThan(semver("1.0.0")) # >= v1.0.0, runtime error if not available
	//
	//    device.driverName == "dra.example.com" # any device from that driver
	//
	// +required
	Expression string `json:"expression" protobuf:"bytes,1,name=expression"`
}

// Besides the request name slice, constraint must have exactly one field set.
type Constraint struct {
	// The constraint applies to devices in these requests. A single entry is okay
	// and used when that request is for multiple devices.
	//
	// If empty, the constrain applies to all devices in the claim.
	//
	// +optional
	// +listType=atomic
	RequestNames []string `json:"requestNames,omitempty" protobuf:"bytes,1,opt,name=requestNames"`

	// The devices must have this attribute and its value must be the same.
	//
	// For example, if you specified "dra.example.com/numa" (a hypothetical example!),
	// then only devices in the same NUMA node will be chosen.
	//
	// +required
	MatchAttribute *string `json:"matchAttribute,omitempty" protobuf:"bytes,2,opt,name=matchAttribute"`

	// TODO (?)
	// MatchQuantity *string

	// Future extension, not part of the current design:
	// A CEL expression which compares different devices and returns
	// true if they match.
	//
	// Because it would be part of a one-of, old schedulers will not
	// accidentally ignore this additional, for them unknown match
	// criteria.
	//
	// matcher string
}

// ClaimConfiguration is used for configuration parameters in ResourcClaimSpec.
type ClaimConfiguration struct {
	// The configuration applies to devices in these requests.
	//
	// If empty, the configuration applies to all devices in the claim.
	//
	// +optional
	// +listType=atomic
	RequestNames []string `json:"requestNames,omitempty" protobuf:"bytes,1,opt,name=requestNames"`

	Configuration `json:",inline" protobuf:"bytes,2,name=configuration"`
}

// Configuration must have exactly one field set. It gets embedded
// inline in some other structs which have other fields, so field names must
// not conflict with those.
type Configuration struct {
	// Opaque provides driver-specific configuration parameters.
	//
	// +required
	Opaque *OpaqueConfiguration `json:"opaque,omitempty" protobuf:"bytes,1,opt,name=opaque"`
}

// OpaqueConfiguration contains configuration parameters for a driver
// in a format defined by the driver vendor.
type OpaqueConfiguration struct {
	// DriverName is used to determine which kubelet plugin needs
	// to be passed these configuration parameters.
	//
	// An admission webhook provided by the driver developer could use this
	// to decide whether it needs to validate them.
	//
	// Must be a DNS subdomain and should end with a DNS domain owned by the
	// vendor of the driver.
	//
	// +required
	DriverName string `json:"driverName" protobuf:"bytes,1,name=driverName"`

	// Parameters can contain arbitrary data. It is the responsibility of
	// the driver developer to handle validation and versioning. Typically this
	// includes self-identification and a version ("kind" + "apiVersion" for
	// Kubernetes types), with conversion between different versions.
	//
	// +required
	Parameters runtime.RawExtension `json:"parameters" protobuf:"bytes,2,name=parameters"`
}

// ResourceClaimStatus tracks whether the resource has been allocated and what
// the result of that was.
type ResourceClaimStatus struct {
	// Allocation is set once the claim has been allocated successfully.
	//
	// +optional
	Allocation *AllocationResult `json:"allocation,omitempty" protobuf:"bytes,1,opt,name=allocation"`

	// ReservedFor indicates which entities are currently allowed to use
	// the claim. A Pod which references a ResourceClaim which is not
	// reserved for that Pod will not be started. A claim that is in
	// use or might be in use because it has been reserved must not get
	// deallocated.
	//
	// In a cluster with multiple scheduler instances, two pods might get
	// scheduled concurrently by different schedulers. When they reference
	// the same ResourceClaim which already has reached its maximum number
	// of consumers, only one pod can be scheduled.
	//
	// Both schedulers try to add their pod to the claim.status.reservedFor
	// field, but only the update that reaches the API server first gets
	// stored. The other one fails with an error and the scheduler
	// which issued it knows that it must put the pod back into the queue,
	// waiting for the ResourceClaim to become usable again.
	//
	// There can be at most 32 such reservations. This may get increased in
	// the future, but not reduced.
	//
	// +optional
	// +listType=map
	// +listMapKey=uid
	// +patchStrategy=merge
	// +patchMergeKey=uid
	ReservedFor []ResourceClaimConsumerReference `json:"reservedFor,omitempty" protobuf:"bytes,2,opt,name=reservedFor" patchStrategy:"merge" patchMergeKey:"uid"`

	// Indicates that a claim is to be deallocated. While this is set,
	// no new consumers may be added to ReservedFor.
	//
	// This is only used if the claim needs to be deallocated by a DRA driver.
	// That driver then must deallocate this claim and reset the field
	// together with clearing the Allocation field.
	//
	// This is an alpha field and requires enabling the DRAControlPlaneController
	// feature gate.
	//
	// +optional
	DeallocationRequested bool `json:"deallocationRequested,omitempty" protobuf:"bytes,3,opt,name=deallocationRequested"`
}

// ReservedForMaxSize is the maximum number of entries in
// claim.status.reservedFor.
const ResourceClaimReservedForMaxSize = 32

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

// AllocationResult contains attributes of an allocated resource.
type AllocationResult struct {
	// Results lists all allocated devices.
	//
	// +optional
	// +listType=atomic
	Results []RequestAllocationResult `json:"results,omitempty" protobuf:"bytes,1,opt,name=results"`

	// This field is a combination of all the claim and class configuration parameters.
	// Drivers can distinguish between those based on a flag.
	//
	// This includes configuration parameters for drivers which have no allocated
	// devices in the result because it is up to the drivers which configuration
	// parameters they support. They can silently ignore unknown configuration
	// parameters.
	//
	// +optional
	// +listType=atomic
	Config []AllocationConfiguration `json:"config,omitempty" protobuf:"bytes,2,opt,name=config"`

	// Setting this field is optional. If unset, the allocated devices are available everywhere.
	//
	// +optional
	AvailableOnNodes *v1.NodeSelector `json:"availableOnNodes,omitempty" protobuf:"bytes,3,opt,name=availableOnNodes"`

	// ControllerName is the name of the DRA driver which handled the
	// allocation. That driver is also responsible for deallocating the
	// claim. It is empty when the claim can be deallocated without
	// involving a driver.
	//
	// A driver may allocate devices provided by other drivers, so this
	// driver name here can be different from the driver names listed for
	// the results.
	//
	// This is an alpha field and requires enabling the DRAControlPlaneController
	// feature gate.
	//
	// +optional
	ControllerName *string `json:"controllerName,omitempty" protobuf:"bytes,4,opt,name=controllerName"`
}

// AllocationResultsMaxSize represents the maximum number of
// entries in allocation.results.
const AllocationResultsMaxSize = 32

// RequestAllocationResult contains the allocation result for one request.
type RequestAllocationResult struct {
	// RequestName identifies the request in the claim which caused this
	// device to be allocated. Multiple devices may have been allocated
	// per request.
	//
	// +required
	RequestName string `json:"requestName" protobuf:"bytes,1,name=requestName"`

	// DriverName specifies the name of the DRA driver whose kubelet
	// plugin should be invoked to process the allocation once the claim is
	// needed on a node.
	//
	// Must be a DNS subdomain and should end with a DNS domain owned by the
	// vendor of the driver.
	//
	// +required
	DriverName string `json:"driverName" protobuf:"bytes,2,name=driverName"`

	// This name together with the driver name and the device name field
	// identify which device was allocated (`<driver name>/<pool name>/<device name>`).
	//
	// Must not be longer than 253 characters and may contain one or more
	// DNS sub-domains separated by slashes.
	//
	// +required
	PoolName string `json:"poolName" protobuf:"bytes,3,name=poolName"`

	// DeviceName references one device instance via its name in the driver's
	// resource pool. It must be a DNS label.
	//
	// +required
	DeviceName string `json:"deviceName" protobuf:"bytes,4,name=deviceName"`
}

// AllocationConfiguration gets embedded in an AllocationResult.
type AllocationConfiguration struct {
	// Admins is true if the source of the configuration was a class and thus
	// not something that a normal user would have been able to set.
	Admin bool `json:"admin,omitempty" protobuf:"bytes,1,opt,name=admin"`

	// The configuration applies to devices in these requests.
	//
	// If empty, the configuration applies to all devices in the claim.
	//
	// +optional
	// +listType=atomic
	RequestNames []string `json:"requestNames,omitempty" protobuf:"bytes,2,opt,name=requestNames"`

	Configuration `json:",inline" protobuf:"bytes,3,name=configuration"`
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
	//
	// +optional
	Status PodSchedulingContextStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// PodSchedulingContextSpec describes where resources for the Pod are needed.
type PodSchedulingContextSpec struct {
	// SelectedNode is the node for which allocation of ResourceClaims that
	// are referenced by the Pod and that use "WaitForFirstConsumer"
	// allocation is to be attempted.
	//
	// +optional
	SelectedNode string `json:"selectedNode,omitempty" protobuf:"bytes,1,opt,name=selectedNode"`

	// PotentialNodes lists nodes where the Pod might be able to run.
	//
	// The size of this field is limited to 128. This is large enough for
	// many clusters. Larger clusters may need more attempts to find a node
	// that suits all pending resources. This may get increased in the
	// future, but not reduced.
	//
	// +optional
	// +listType=atomic
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
	//
	// +required
	Name string `json:"name" protobuf:"bytes,1,name=name"`

	// UnsuitableNodes lists nodes that the ResourceClaim cannot be
	// allocated for.
	//
	// The size of this field is limited to 128, the same as for
	// PodSchedulingSpec.PotentialNodes. This may get increased in the
	// future, but not reduced.
	//
	// +optional
	// +listType=atomic
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
// +k8s:prerelease-lifecycle-gen:introduced=1.31

// DeviceClass is a vendor or admin-provided resource that contains
// device configuration and selectors. It can be referenced in
// the device requests of a claim to apply these presets.
// Cluster scoped.
type DeviceClass struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec defines what can be allocated and how to configure it.
	//
	// This is mutable. Consumers have to be prepared for classes changing
	// at any time, either because they get updated or replaced. Claim
	// allocations are done once based on whatever was set in classes at
	// the time of allocation.
	//
	// Changing the spec bumps up the generation number.
	Spec DeviceClassSpec `json:"spec" protobuf:"bytes,2,name=spec"`

	// Future extension: status with information about errors in CEL expressions.
}

type DeviceClassSpec struct {
	// Each selector must be satisfied by a device which is claimed via this class.
	//
	// +optional
	// +listType=atomic
	Selectors []Selector `json:"selectors,omitempty" protobuf:"bytes,1,opt,name=selectors"`

	// Config defines configuration parameters that apply to each device that is claimed via this class.
	// Some classses may potentially be satisfied by multiple drivers, so each instance of a vendor
	// configuration applies to exactly one driver.
	//
	// They are passed to the driver, but are not considered while allocating the claim.
	//
	// +optional
	// +listType=atomic
	Config []ClassConfiguration `json:"config,omitempty" protobuf:"bytes,2,opt,name=config"`

	// Only nodes matching the selector will be considered by the scheduler
	// when trying to find a Node that fits a Pod when that Pod uses
	// a claim that has not been allocated yet *and* that claim
	// gets allocated through a control plane controller. It is ignored
	// when the claim does not use a control plane controller
	// for allocation.
	//
	// Setting this field is optional. If unset, all Nodes are candidates.
	//
	// This is an alpha field and requires enabling the DRAControlPlaneController
	// feature gate.
	//
	// +optional
	SuitableNodes *v1.NodeSelector `json:"suitableNodes,omitempty" protobuf:"bytes,3,opt,name=suitableNodes"`
}

// ClassConfiguration is used in DeviceClass.
type ClassConfiguration struct {
	Configuration `json:",inline" protobuf:"bytes,1,opt,name=configuration"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// DeviceClassList is a collection of classes.
type DeviceClassList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of resource classes.
	Items []DeviceClass `json:"items" protobuf:"bytes,2,rep,name=items"`
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
