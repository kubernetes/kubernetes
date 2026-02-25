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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.36
// +k8s:supportsSubresource=/status

// ResourcePoolStatusRequest triggers a one-time calculation of resource pool status
// based on the provided filters. The request follows a request/response pattern similar
// to CertificateSigningRequest - create a request, and the controller populates the status.
//
// Once status.observationTime is set, the request is considered complete and will not
// be reprocessed. Users should delete and recreate requests to get updated information.
type ResourcePoolStatusRequest struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec defines the filters for which pools to include in the status.
	// The spec is immutable once created.
	//
	// +required
	// +k8s:immutable
	Spec ResourcePoolStatusRequestSpec `json:"spec,omitzero" protobuf:"bytes,2,name=spec"`

	// Status is populated by the controller with the calculated pool status.
	// Once observationTime is set, the status is considered complete and immutable.
	//
	// +optional
	Status ResourcePoolStatusRequestStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// ResourcePoolStatusRequestSpec defines the filters for the pool status request.
type ResourcePoolStatusRequestSpec struct {
	// Driver specifies the DRA driver name to filter pools.
	// Only pools from ResourceSlices with this driver will be included.
	// Must be a DNS subdomain (e.g., "gpu.example.com").
	// This field is immutable.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-long-name
	Driver string `json:"driver" protobuf:"bytes,1,name=driver"`

	// PoolName optionally filters to a specific pool name.
	// If not specified, all pools from the specified driver are included.
	// When specified, must be a non-empty valid resource pool name
	// (DNS subdomains separated by "/").
	//
	// +optional
	// +k8s:optional
	// +k8s:format=k8s-resource-pool-name
	PoolName *string `json:"poolName,omitempty" protobuf:"bytes,2,opt,name=poolName"`

	// Limit optionally specifies the maximum number of pools to return in the status.
	// If more pools match the filter criteria, the response will be truncated
	// and status.truncation will be set to "Truncated".
	//
	// Default: 100
	// Minimum: 1
	// Maximum: 1000
	//
	// +optional
	// +k8s:optional
	// +default=100
	// +k8s:minimum=1
	Limit *int32 `json:"limit,omitempty" protobuf:"varint,3,opt,name=limit"`
}

// ResourcePoolStatusRequestLimitDefault is the default value for spec.limit.
const ResourcePoolStatusRequestLimitDefault int32 = 100

// ResourcePoolStatusRequestLimitMax is the maximum allowed value for spec.limit.
const ResourcePoolStatusRequestLimitMax int32 = 1000

// ResourcePoolStatusRequestStatus contains the calculated pool status information.
type ResourcePoolStatusRequestStatus struct {
	// ObservationTime is the timestamp when the controller calculated this status.
	// Once set, the request is considered complete and will not be reprocessed.
	// Users should delete and recreate the request to get updated information.
	//
	// +optional
	// +k8s:optional
	ObservationTime *metav1.Time `json:"observationTime,omitempty" protobuf:"bytes,1,opt,name=observationTime"`

	// Pools contains the status of each pool matching the request filters.
	// The list is sorted by driver, then pool name.
	//
	// +optional
	// +k8s:optional
	// +listType=atomic
	// +k8s:listType=atomic
	Pools []PoolStatus `json:"pools,omitempty" protobuf:"bytes,2,rep,name=pools"`

	// Conditions provide information about the state of the request.
	//
	// Known condition types:
	// - "Complete": True when the request has been processed successfully
	// - "Failed": True when the request could not be processed
	//
	// +optional
	// +k8s:optional
	// +listType=map
	// +k8s:listType=map
	// +listMapKey=type
	// +k8s:listMapKey=type
	// +patchStrategy=merge
	// +patchMergeKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,3,rep,name=conditions"`

	// ValidationErrors contains any validation errors encountered while processing
	// the request. If present, the request may have partial or no results.
	//
	// +optional
	// +k8s:optional
	// +listType=atomic
	// +k8s:listType=atomic
	// +k8s:maxItems=10
	ValidationErrors []string `json:"validationErrors,omitempty" protobuf:"bytes,4,rep,name=validationErrors"`

	// Truncation indicates whether the response was truncated due to the limit.
	// When set to "Truncated", there are more pools matching the filter criteria
	// than were returned. When omitted, the response was not truncated.
	//
	// +optional
	// +k8s:optional
	Truncation *TruncationStatus `json:"truncation,omitempty" protobuf:"bytes,5,opt,name=truncation"`

	// TotalMatchingPools is the total number of pools that matched the filter criteria,
	// regardless of truncation. This helps users understand how many pools exist
	// even when the response is truncated. When nil, the status has not yet been
	// populated. A value of 0 means no pools matched the filter criteria.
	//
	// +optional
	// +k8s:optional
	// +k8s:minimum=0
	TotalMatchingPools *int32 `json:"totalMatchingPools,omitempty" protobuf:"varint,6,opt,name=totalMatchingPools"`
}

// PoolStatus contains status information for a single resource pool.
type PoolStatus struct {
	// Driver is the DRA driver name for this pool.
	// Must be a DNS subdomain (e.g., "gpu.example.com").
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-long-name
	Driver string `json:"driver" protobuf:"bytes,1,name=driver"`

	// PoolName is the name of the pool.
	// Must be a valid resource pool name (DNS subdomains separated by "/").
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-resource-pool-name
	PoolName string `json:"poolName" protobuf:"bytes,2,name=poolName"`

	// NodeName is the node this pool is associated with.
	// When omitted, the pool is not associated with a specific node.
	//
	// +optional
	// +k8s:optional
	NodeName *string `json:"nodeName,omitempty" protobuf:"bytes,3,opt,name=nodeName"`

	// TotalDevices is the total number of devices in the pool across all slices.
	// A value of 0 means the pool has no devices.
	//
	// +required
	// +k8s:required
	// +k8s:minimum=0
	TotalDevices *int32 `json:"totalDevices,omitempty" protobuf:"varint,4,opt,name=totalDevices"`

	// AllocatedDevices is the number of devices currently allocated to claims.
	// A value of 0 means no devices are allocated.
	//
	// +required
	// +k8s:required
	// +k8s:minimum=0
	AllocatedDevices *int32 `json:"allocatedDevices,omitempty" protobuf:"varint,5,opt,name=allocatedDevices"`

	// AvailableDevices is the number of devices available for allocation.
	// This equals TotalDevices - AllocatedDevices - UnavailableDevices.
	// A value of 0 means no devices are currently available.
	//
	// +required
	// +k8s:required
	// +k8s:minimum=0
	AvailableDevices *int32 `json:"availableDevices,omitempty" protobuf:"varint,6,opt,name=availableDevices"`

	// UnavailableDevices is the number of devices that are not available
	// due to taints or other conditions, but are not allocated.
	//
	// +optional
	// +k8s:optional
	// +k8s:minimum=0
	UnavailableDevices int32 `json:"unavailableDevices,omitempty" protobuf:"varint,7,opt,name=unavailableDevices"`

	// SliceCount is the number of ResourceSlices that make up this pool.
	//
	// +required
	// +k8s:required
	// +k8s:minimum=1
	SliceCount int32 `json:"sliceCount,omitempty" protobuf:"varint,8,opt,name=sliceCount"`

	// Generation is the maximum metadata.generation observed across all
	// ResourceSlices in this pool. Can be used to detect changes.
	//
	// +optional
	// +k8s:optional
	// +k8s:minimum=0
	Generation int64 `json:"generation,omitempty" protobuf:"varint,9,opt,name=generation"`
}

// TruncationStatus is a string enum indicating whether the response was truncated.
//
// +enum
// +k8s:enum
type TruncationStatus string

const (
	// TruncationStatusTruncated indicates the response was truncated due to the limit.
	TruncationStatusTruncated TruncationStatus = "Truncated"
)

// ResourcePoolStatusRequestConditionComplete is the condition type for completed requests.
const ResourcePoolStatusRequestConditionComplete = "Complete"

// ResourcePoolStatusRequestConditionFailed is the condition type for failed requests.
const ResourcePoolStatusRequestConditionFailed = "Failed"

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.36

// ResourcePoolStatusRequestList is a collection of ResourcePoolStatusRequests.
type ResourcePoolStatusRequestList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of ResourcePoolStatusRequests.
	Items []ResourcePoolStatusRequest `json:"items" protobuf:"bytes,2,rep,name=items"`
}
