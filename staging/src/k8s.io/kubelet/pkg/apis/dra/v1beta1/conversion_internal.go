/*
Copyright 2025 The Kubernetes Authors.

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
	conversion "k8s.io/apimachinery/pkg/conversion"
	v1 "k8s.io/kubelet/pkg/apis/dra/v1"
)

// Convert_v1beta1_Claim_To_v1_Claim is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1beta1_Claim_To_v1_Claim(in *Claim, out *v1.Claim, s conversion.Scope) error {
	return autoConvert_v1beta1_Claim_To_v1_Claim(in, out, s)
}

// Convert_v1_Claim_To_v1beta1_Claim is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1_Claim_To_v1beta1_Claim(in *v1.Claim, out *Claim, s conversion.Scope) error {
	return autoConvert_v1_Claim_To_v1beta1_Claim(in, out, s)
}

// Convert_v1beta1_Device_To_v1_Device is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1beta1_Device_To_v1_Device(in *Device, out *v1.Device, s conversion.Scope) error {
	return autoConvert_v1beta1_Device_To_v1_Device(in, out, s)
}

// Convert_v1_Device_To_v1beta1_Device is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1_Device_To_v1beta1_Device(in *v1.Device, out *Device, s conversion.Scope) error {
	return autoConvert_v1_Device_To_v1beta1_Device(in, out, s)
}

// Convert_v1beta1_NodePrepareResourceResponse_To_v1_NodePrepareResourceResponse is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1beta1_NodePrepareResourceResponse_To_v1_NodePrepareResourceResponse(in *NodePrepareResourceResponse, out *v1.NodePrepareResourceResponse, s conversion.Scope) error {
	return autoConvert_v1beta1_NodePrepareResourceResponse_To_v1_NodePrepareResourceResponse(in, out, s)
}

// Convert_v1_NodePrepareResourceResponse_To_v1beta1_NodePrepareResourceResponse is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1_NodePrepareResourceResponse_To_v1beta1_NodePrepareResourceResponse(in *v1.NodePrepareResourceResponse, out *NodePrepareResourceResponse, s conversion.Scope) error {
	return autoConvert_v1_NodePrepareResourceResponse_To_v1beta1_NodePrepareResourceResponse(in, out, s)
}

// Convert_v1beta1_NodePrepareResourcesRequest_To_v1_NodePrepareResourcesRequest is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1beta1_NodePrepareResourcesRequest_To_v1_NodePrepareResourcesRequest(in *NodePrepareResourcesRequest, out *v1.NodePrepareResourcesRequest, s conversion.Scope) error {
	return autoConvert_v1beta1_NodePrepareResourcesRequest_To_v1_NodePrepareResourcesRequest(in, out, s)
}

// Convert_v1_NodePrepareResourcesRequest_To_v1beta1_NodePrepareResourcesRequest is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1_NodePrepareResourcesRequest_To_v1beta1_NodePrepareResourcesRequest(in *v1.NodePrepareResourcesRequest, out *NodePrepareResourcesRequest, s conversion.Scope) error {
	return autoConvert_v1_NodePrepareResourcesRequest_To_v1beta1_NodePrepareResourcesRequest(in, out, s)
}

// Convert_v1beta1_NodePrepareResourcesResponse_To_v1_NodePrepareResourcesResponse is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1beta1_NodePrepareResourcesResponse_To_v1_NodePrepareResourcesResponse(in *NodePrepareResourcesResponse, out *v1.NodePrepareResourcesResponse, s conversion.Scope) error {
	return autoConvert_v1beta1_NodePrepareResourcesResponse_To_v1_NodePrepareResourcesResponse(in, out, s)
}

// Convert_v1_NodePrepareResourcesResponse_To_v1beta1_NodePrepareResourcesResponse is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1_NodePrepareResourcesResponse_To_v1beta1_NodePrepareResourcesResponse(in *v1.NodePrepareResourcesResponse, out *NodePrepareResourcesResponse, s conversion.Scope) error {
	return autoConvert_v1_NodePrepareResourcesResponse_To_v1beta1_NodePrepareResourcesResponse(in, out, s)
}

// Convert_v1beta1_NodeUnprepareResourceResponse_To_v1_NodeUnprepareResourceResponse is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1beta1_NodeUnprepareResourceResponse_To_v1_NodeUnprepareResourceResponse(in *NodeUnprepareResourceResponse, out *v1.NodeUnprepareResourceResponse, s conversion.Scope) error {
	return autoConvert_v1beta1_NodeUnprepareResourceResponse_To_v1_NodeUnprepareResourceResponse(in, out, s)
}

// Convert_v1_NodeUnprepareResourceResponse_To_v1beta1_NodeUnprepareResourceResponse is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1_NodeUnprepareResourceResponse_To_v1beta1_NodeUnprepareResourceResponse(in *v1.NodeUnprepareResourceResponse, out *NodeUnprepareResourceResponse, s conversion.Scope) error {
	return autoConvert_v1_NodeUnprepareResourceResponse_To_v1beta1_NodeUnprepareResourceResponse(in, out, s)
}

// Convert_v1beta1_NodeUnprepareResourcesRequest_To_v1_NodeUnprepareResourcesRequest is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1beta1_NodeUnprepareResourcesRequest_To_v1_NodeUnprepareResourcesRequest(in *NodeUnprepareResourcesRequest, out *v1.NodeUnprepareResourcesRequest, s conversion.Scope) error {
	return autoConvert_v1beta1_NodeUnprepareResourcesRequest_To_v1_NodeUnprepareResourcesRequest(in, out, s)
}

// Convert_v1_NodeUnprepareResourcesRequest_To_v1beta1_NodeUnprepareResourcesRequest is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1_NodeUnprepareResourcesRequest_To_v1beta1_NodeUnprepareResourcesRequest(in *v1.NodeUnprepareResourcesRequest, out *NodeUnprepareResourcesRequest, s conversion.Scope) error {
	return autoConvert_v1_NodeUnprepareResourcesRequest_To_v1beta1_NodeUnprepareResourcesRequest(in, out, s)
}

// Convert_v1beta1_NodeUnprepareResourcesResponse_To_v1_NodeUnprepareResourcesResponse is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1beta1_NodeUnprepareResourcesResponse_To_v1_NodeUnprepareResourcesResponse(in *NodeUnprepareResourcesResponse, out *v1.NodeUnprepareResourcesResponse, s conversion.Scope) error {
	return autoConvert_v1beta1_NodeUnprepareResourcesResponse_To_v1_NodeUnprepareResourcesResponse(in, out, s)
}

// Convert_v1_NodeUnprepareResourcesResponse_To_v1beta1_NodeUnprepareResourcesResponse is a manually created conversion function because the object contains private fields that cannot be auto-converted.
func Convert_v1_NodeUnprepareResourcesResponse_To_v1beta1_NodeUnprepareResourcesResponse(in *v1.NodeUnprepareResourcesResponse, out *NodeUnprepareResourcesResponse, s conversion.Scope) error {
	return autoConvert_v1_NodeUnprepareResourcesResponse_To_v1beta1_NodeUnprepareResourcesResponse(in, out, s)
}
