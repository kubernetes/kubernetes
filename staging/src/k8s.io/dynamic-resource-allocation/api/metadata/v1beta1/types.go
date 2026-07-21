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

package v1beta1

import (
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// DeviceMetadata contains metadata about devices allocated to a ResourceClaim.
// It is serialized to versioned JSON files that can be mounted into containers.
//
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type DeviceMetadata struct {
	metav1.TypeMeta   `json:""`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// PodClaimName is the name used to reference this claim in the pod spec
	// (i.e., the entry in pod.spec.resourceClaims[].name). For claims
	// created from a ResourceClaimTemplate, this differs from the actual
	// ResourceClaim name and is needed to resolve the mapping.
	//
	// +optional
	// +k8s:optional
	// +k8s:format=k8s-short-name
	PodClaimName *string `json:"podClaimName,omitempty"`

	// Requests contains the device allocation information for each request
	// in the ResourceClaim.
	//
	// +optional
	// +k8s:optional
	Requests []DeviceMetadataRequest `json:"requests,omitempty"`
}

// DeviceMetadataRequest contains metadata for a single request within a ResourceClaim.
type DeviceMetadataRequest struct {
	// Name is the request name from the ResourceClaim spec. For requests
	// using FirstAvailable with subrequests, this includes the subrequest
	// name in the format "<request>/<subrequest>", matching the format
	// used in DeviceRequestAllocationResult.Request
	//
	// +required
	// +k8s:required
	Name string `json:"name"`

	// Devices contains metadata for each device allocated to this request.
	//
	// +optional
	// +k8s:optional
	Devices []Device `json:"devices,omitempty"`
}

// Device contains metadata about a single allocated device.
type Device struct {
	// Driver is the name of the DRA driver that manages this device.
	// It must be a DNS subdomain no longer than 63 characters.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-long-name-caseless
	// +k8s:maxLength=63
	Driver string `json:"driver"`

	// Pool is the name of the resource pool this device belongs to.
	// It must be one or more DNS subdomains separated by slashes and must
	// not be longer than 253 characters.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-resource-pool-name
	Pool string `json:"pool"`

	// Name is the name of the device within the pool.
	// It must be a DNS label.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-short-name
	Name string `json:"name"`

	// Attributes contains device attributes in the same format as a ResourceSlice.
	// Keys are qualified attribute names (e.g., "model", "resource.k8s.io/pciBusID").
	// There can be at most 32 attributes.
	//
	// +optional
	// +k8s:optional
	// +k8s:maxProperties=32
	// +k8s:customValidation
	// +k8s:eachVal=+k8s:opaqueType
	Attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute `json:"attributes,omitempty"`

	// NetworkData contains network-specific device data (e.g., interface name,
	// addresses, hardware address). This is populated for network devices,
	// typically during the CRI RPC RunPodSandbox, before the containers are
	// started and after the network namespace is created.
	//
	// +optional
	// +k8s:optional
	// +k8s:customValidation
	// +k8s:opaqueType
	NetworkData *resourceapi.NetworkDeviceData `json:"networkData,omitempty"`
}
