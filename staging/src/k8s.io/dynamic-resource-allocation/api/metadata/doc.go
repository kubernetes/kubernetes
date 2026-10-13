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

// +k8s:deepcopy-gen=package

// Package metadata contains internal (unversioned) types for DRA device
// metadata. This metadata consists of device attributes that are similar, but
// not necessarily identical to the ones found in ResourceSlices. DRA drivers
// publish this information for consumption inside the containers which use
// devices.
//
// These types are the canonical in-memory representation that Go
// consumers program against. Versioned types (e.g. v1alpha1) are converted
// to/from these internal types via the scheme.
//
// # In-container metadata API
//
// Device metadata files are mounted read-only into every container that
// references one or more ResourceClaims, provided that the DRA drivers which
// allocate the devices support the feature.
//
// The directory layout distinguishes between two kinds of claims because
// consumers may have to be configured at the time when the Pod spec is
// created, i.e. without access to the Pod status where the mapping from
// ResourceClaimTemplate to ResourceClaim is stored at runtime:
//
//   - Directly referenced claims (pod.spec.resourceClaims[].resourceClaimName
//     is set) are stored under the [ResourceClaimsSubDir] subdirectory,
//     keyed by the ResourceClaim name, which the consumer can read from
//     the pod spec:
//
//     /var/run/kubernetes.io/dra-device-attributes/resourceclaims/<claimName>/<requestName>/<driverName>-metadata.json
//
//   - Template-generated claims (pod.spec.resourceClaims[].resourceClaimTemplateName
//     was set) are stored under the [ResourceClaimTemplatesSubDir] subdirectory,
//     keyed by the pod-local claim name (pod.spec.resourceClaims[].name).
//     The actual ResourceClaim name is generated and not predictable from
//     the pod spec, so the pod-local name is used instead:
//
//     /var/run/kubernetes.io/dra-device-attributes/resourceclaimtemplates/<podClaimName>/<requestName>/<driverName>-metadata.json
//
// DRA drivers can derive these file names from the information which is available
// to them through the ResourceClaims, without relying on the Pod.
//
// The <driverName> in the file name is the DRA driver name (e.g.,
// "gpu.example.com"). Each driver writes its own file, so multiple drivers
// serving the same request do not collide.
//
// The <requestName> segment corresponds to an entry in the claim's
// spec.devices.requests[] list. For requests that use FirstAvailable with
// subrequests, the subrequest name will be removed and only request name will be used
//
// # Accessing metadata files
//
// Individual metadata files are bind-mounted read-only. Consumers should
// construct the exact file paths from known claim names and request names
// rather than enumerating directories.
//
// # File format
//
// Each metadata.json file is a concatenated JSON stream containing the same
// [DeviceMetadata] object encoded once per supported API version, newest
// first. This ensures forward and backward compatibility: older consumers
// skip versions they do not recognise and newer consumers can always find a
// version they support. Standard JSON decoders can read the objects one by
// one; see the k8s.io/dynamic-resource-allocation/devicemetadata package
// for a ready-made decoder.
package metadata
