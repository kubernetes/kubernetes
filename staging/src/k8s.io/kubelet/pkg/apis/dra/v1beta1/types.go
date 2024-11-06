/*
Copyright 2024 The Kubernetes Authors.

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

// Package v1beta1 has the same Go API as v1alpha4. A DRA driver implementing
// [v1beta1.DRAPluginServer] also implements [v1alpha4.NodeServer] and vice versa.
//
// The [k8s.io/dynamic-resource-allocation/kubeletplugin] helper will
// automatically register both API versions unless explicitly configured
// otherwise.
package v1beta1

import (
	"k8s.io/kubelet/pkg/apis/dra/v1alpha4"
)

type (
	NodePrepareResourcesRequest    = v1alpha4.NodePrepareResourcesRequest
	NodePrepareResourcesResponse   = v1alpha4.NodePrepareResourcesResponse
	NodePrepareResourceResponse    = v1alpha4.NodePrepareResourceResponse
	NodeUnprepareResourcesRequest  = v1alpha4.NodeUnprepareResourcesRequest
	NodeUnprepareResourcesResponse = v1alpha4.NodeUnprepareResourcesResponse
	NodeUnprepareResourceResponse  = v1alpha4.NodeUnprepareResourceResponse
	Device                         = v1alpha4.Device
	Claim                          = v1alpha4.Claim
)

const (
	// DRAPluginService needs to be listed in the "supported versions"
	// array during plugin registration by a DRA plugin which provides
	// an implementation of the v1beta1 DRAPlugin service.
	DRAPluginService = "v1beta1.DRAPlugin"
)

// Ensure that the interfaces are equivalent.
var _ DRAPluginServer = v1alpha4.NodeServer(nil)
