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

package metadata

import (
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// The following constants define the in-container directory layout for device
// metadata files. They form part of the consumer-facing API: changing any of
// these values is a breaking change for applications that read metadata files.
const (
	// ContainerDir is the well-known directory inside containers where
	// device metadata files are mounted.
	ContainerDir = "/var/run/kubernetes.io/dra-device-attributes"

	// ResourceClaimsSubDir is the subdirectory under ContainerDir for
	// directly referenced ResourceClaims (i.e., those listed by name in
	// pod.spec.resourceClaims[].resourceClaimName). The claim name is
	// used as the next path segment.
	ResourceClaimsSubDir = "resourceclaims"

	// ResourceClaimTemplatesSubDir is the subdirectory under ContainerDir
	// for claims created from a ResourceClaimTemplate. The podClaimName
	// (pod.spec.resourceClaims[].name) is used as the next path segment.
	ResourceClaimTemplatesSubDir = "resourceclaimtemplates"

	// MetadataFileSuffix is appended to the driver name to form the
	// metadata file name in each request directory. The full file name
	// is "{driverName}-metadata.json"; using the driver name avoids
	// collisions when multiple drivers serve the same request.
	MetadataFileSuffix = "-metadata.json"
)

// DeviceMetadata contains metadata about devices allocated to a ResourceClaim.
// It is serialized to versioned JSON files mounted into containers under
// [ContainerDir].
//
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type DeviceMetadata struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// PodClaimName is the name used to reference this claim in the pod spec
	// (i.e., the entry in pod.spec.resourceClaims[].name). It is set for claims
	// created from a ResourceClaimTemplate, otherwise it is nil. When set, the
	// metadata file is placed under [ResourceClaimTemplatesSubDir]; otherwise
	// under [ResourceClaimsSubDir].
	PodClaimName *string

	// Requests contains the device allocation information for each request
	// in the ResourceClaim.
	Requests []DeviceMetadataRequest
}

// DeviceMetadataRequest contains metadata for a single request within a ResourceClaim.
type DeviceMetadataRequest struct {
	// Name is the request name from the ResourceClaim spec. For requests
	// using FirstAvailable with subrequests, this includes the subrequest
	// name in the format "<request>/<subrequest>", matching the format
	// used in DeviceRequestAllocationResult.Request.
	Name string

	// Devices contains metadata for each device allocated to this request.
	Devices []Device
}

// Device contains metadata about a single allocated device.
type Device struct {
	// Driver is the name of the DRA driver that manages this device.
	Driver string

	// Pool is the name of the resource pool this device belongs to.
	Pool string

	// Name is the name of the device within the pool.
	Name string

	// Attributes contains device attributes in the same format as a ResourceSlice.
	// Keys are qualified attribute names (e.g., "model", "resource.k8s.io/pciBusID").
	Attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute

	// NetworkData contains network-specific device data (e.g., interface name,
	// addresses, hardware address). This is populated for network devices,
	// typically during the CRI RPC RunPodSandbox, before the containers are
	// started and after the network namespace is created.
	NetworkData *resourceapi.NetworkDeviceData
}
