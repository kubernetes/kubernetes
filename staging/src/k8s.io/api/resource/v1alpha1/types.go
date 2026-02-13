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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.36

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
	//
	// +required
	Spec ResourcePoolStatusRequestSpec `json:"spec" protobuf:"bytes,2,name=spec"`

	// Status is populated by the controller with the calculated pool status.
	//
	// +optional
	Status ResourcePoolStatusRequestStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// ResourcePoolStatusRequestSpec defines the filters for the pool status request.
type ResourcePoolStatusRequestSpec struct {
	// Driver specifies the DRA driver name to filter pools.
	// Only pools from ResourceSlices with this driver will be included.
	// This field is required to bound the scope of the request.
	//
	// +required
	Driver string `json:"driver" protobuf:"bytes,1,name=driver"`

	// PoolName optionally filters to a specific pool name.
	// If not specified, all pools from the specified driver are included.
	//
	// +optional
	PoolName string `json:"poolName,omitempty" protobuf:"bytes,2,opt,name=poolName"`

	// Limit optionally specifies the maximum number of pools to return in the status.
	// If more pools match the filter criteria, the response will be truncated
	// and status.truncated will be set to true.
	//
	// Default: 100
	// Maximum: 1000
	//
	// +optional
	// +kubebuilder:default=100
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=1000
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
	ObservationTime *metav1.Time `json:"observationTime,omitempty" protobuf:"bytes,1,opt,name=observationTime"`

	// Pools contains the status of each pool matching the request filters.
	// The list is sorted by driver, then pool name.
	//
	// +optional
	// +listType=atomic
	Pools []PoolStatus `json:"pools,omitempty" protobuf:"bytes,2,rep,name=pools"`

	// Conditions provide information about the state of the request.
	//
	// Known condition types:
	// - "Complete": True when the request has been processed successfully
	// - "Failed": True when the request could not be processed
	//
	// +optional
	// +listType=map
	// +listMapKey=type
	// +patchStrategy=merge
	// +patchMergeKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,3,rep,name=conditions"`

	// ValidationErrors contains any validation errors encountered while processing
	// the request. If present, the request may have partial or no results.
	//
	// +optional
	// +listType=atomic
	ValidationErrors []string `json:"validationErrors,omitempty" protobuf:"bytes,4,rep,name=validationErrors"`

	// Truncated indicates whether the response was truncated due to the limit.
	// If true, there are more pools matching the filter criteria than were returned.
	//
	// +optional
	Truncated bool `json:"truncated,omitempty" protobuf:"varint,5,opt,name=truncated"`

	// TotalMatchingPools is the total number of pools that matched the filter criteria,
	// regardless of truncation. This helps users understand how many pools exist
	// even when the response is truncated.
	//
	// +optional
	TotalMatchingPools int32 `json:"totalMatchingPools,omitempty" protobuf:"varint,6,opt,name=totalMatchingPools"`
}

// PoolStatus contains status information for a single resource pool.
type PoolStatus struct {
	// Driver is the DRA driver name for this pool.
	//
	// +required
	Driver string `json:"driver" protobuf:"bytes,1,name=driver"`

	// PoolName is the name of the pool.
	//
	// +required
	PoolName string `json:"poolName" protobuf:"bytes,2,name=poolName"`

	// NodeName is the node this pool is associated with.
	// Empty for non-node-local pools.
	//
	// +optional
	NodeName string `json:"nodeName,omitempty" protobuf:"bytes,3,opt,name=nodeName"`

	// TotalDevices is the total number of devices in the pool across all slices.
	//
	// +required
	TotalDevices int32 `json:"totalDevices" protobuf:"varint,4,name=totalDevices"`

	// AllocatedDevices is the number of devices currently allocated to claims.
	//
	// +required
	AllocatedDevices int32 `json:"allocatedDevices" protobuf:"varint,5,name=allocatedDevices"`

	// AvailableDevices is the number of devices available for allocation.
	// This equals TotalDevices - AllocatedDevices - UnavailableDevices.
	//
	// +required
	AvailableDevices int32 `json:"availableDevices" protobuf:"varint,6,name=availableDevices"`

	// UnavailableDevices is the number of devices that are not available
	// due to taints or other conditions, but are not allocated.
	//
	// +optional
	UnavailableDevices int32 `json:"unavailableDevices,omitempty" protobuf:"varint,7,opt,name=unavailableDevices"`

	// SliceCount is the number of ResourceSlices that make up this pool.
	//
	// +required
	SliceCount int32 `json:"sliceCount" protobuf:"varint,8,name=sliceCount"`

	// Generation is the maximum metadata.generation observed across all
	// ResourceSlices in this pool. Can be used to detect changes.
	//
	// +optional
	Generation int64 `json:"generation,omitempty" protobuf:"varint,9,opt,name=generation"`
}

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
