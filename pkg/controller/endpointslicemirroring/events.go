/*
Copyright 2020 The Kubernetes Authors.

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

package endpointslicemirroring

const (
	// FailedToListEndpointSlices indicates the controller has failed to list
	// EndpointSlices.
	FailedToListEndpointSlices = "FailedToListEndpointSlices"
	// FailedToUpdateEndpointSlices indicates the controller has failed to
	// update EndpointSlices.
	FailedToUpdateEndpointSlices = "FailedToUpdateEndpointSlices"
	// InvalidIPAddress indicates that an IP address found in an Endpoints
	// resource is invalid.
	InvalidIPAddress = "InvalidIPAddress"
	// TooManyAddressesToMirror indicates that some addresses were not mirrored
	// due to an EndpointSubset containing more addresses to mirror than
	// MaxEndpointsPerSubset allows.
	TooManyAddressesToMirror = "TooManyAddressesToMirror"
)
