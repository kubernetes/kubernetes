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

package v1beta1

import (
	context "context"
	fmt "fmt"

	grpc "google.golang.org/grpc"
	"k8s.io/apimachinery/pkg/runtime"

	v1 "k8s.io/kubelet/pkg/apis/dra/v1"
)

var (
	localSchemeBuilder runtime.SchemeBuilder
	AddToScheme        = localSchemeBuilder.AddToScheme
)

// V1ServerWrapper implements the [NodeServer] interface by wrapping a [v1.DRAPluginServer].
type V1ServerWrapper struct {
	UnsafeDRAPluginServer
	v1.DRAPluginServer
}

var _ DRAPluginServer = V1ServerWrapper{}

func (w V1ServerWrapper) NodePrepareResources(ctx context.Context, req *NodePrepareResourcesRequest) (*NodePrepareResourcesResponse, error) {
	var convertedReq v1.NodePrepareResourcesRequest
	if err := Convert_v1beta1_NodePrepareResourcesRequest_To_v1_NodePrepareResourcesRequest(req, &convertedReq, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodePrepareResourcesRequest from v1beta1 to v1: %w", err)
	}
	resp, err := w.DRAPluginServer.NodePrepareResources(ctx, &convertedReq)
	if err != nil {
		return nil, err
	}
	var convertedResp NodePrepareResourcesResponse
	if err := Convert_v1_NodePrepareResourcesResponse_To_v1beta1_NodePrepareResourcesResponse(resp, &convertedResp, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodePrepareResourcesResponse from v1 to v1beta1: %w", err)
	}
	return &convertedResp, nil
}

func (w V1ServerWrapper) NodeUnprepareResources(ctx context.Context, req *NodeUnprepareResourcesRequest) (*NodeUnprepareResourcesResponse, error) {
	var convertedReq v1.NodeUnprepareResourcesRequest
	if err := Convert_v1beta1_NodeUnprepareResourcesRequest_To_v1_NodeUnprepareResourcesRequest(req, &convertedReq, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodeUnprepareResourcesRequest from v1beta1 to v1: %w", err)
	}
	resp, err := w.DRAPluginServer.NodeUnprepareResources(ctx, &convertedReq)
	if err != nil {
		return nil, err
	}
	var convertedResp NodeUnprepareResourcesResponse
	if err := Convert_v1_NodeUnprepareResourcesResponse_To_v1beta1_NodeUnprepareResourcesResponse(resp, &convertedResp, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodeUnprepareResourcesResponse from v1 to v1beta1: %w", err)
	}
	return &convertedResp, nil
}

// V1Beta1ServerWrapper implements the [v1.DRAPluginServer] interface by wrapping a [NodeServer].
type V1Beta1ServerWrapper struct {
	v1.UnsafeDRAPluginServer
	DRAPluginServer
}

var _ v1.DRAPluginServer = V1Beta1ServerWrapper{}

func (w V1Beta1ServerWrapper) NodePrepareResources(ctx context.Context, req *v1.NodePrepareResourcesRequest) (*v1.NodePrepareResourcesResponse, error) {
	var convertedReq NodePrepareResourcesRequest
	if err := Convert_v1_NodePrepareResourcesRequest_To_v1beta1_NodePrepareResourcesRequest(req, &convertedReq, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodePrepareResourcesRequest from v1 to v1beta1: %w", err)
	}
	resp, err := w.DRAPluginServer.NodePrepareResources(ctx, &convertedReq)
	if err != nil {
		return nil, err
	}
	var convertedResp v1.NodePrepareResourcesResponse
	if err := Convert_v1beta1_NodePrepareResourcesResponse_To_v1_NodePrepareResourcesResponse(resp, &convertedResp, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodePrepareResourcesResponse from v1beta1 to v1: %w", err)
	}
	return &convertedResp, nil
}

func (w V1Beta1ServerWrapper) NodeUnprepareResources(ctx context.Context, req *v1.NodeUnprepareResourcesRequest) (*v1.NodeUnprepareResourcesResponse, error) {
	var convertedReq NodeUnprepareResourcesRequest
	if err := Convert_v1_NodeUnprepareResourcesRequest_To_v1beta1_NodeUnprepareResourcesRequest(req, &convertedReq, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodeUnprepareResourcesRequest from v1 to v1beta1: %w", err)
	}
	resp, err := w.DRAPluginServer.NodeUnprepareResources(ctx, &convertedReq)
	if err != nil {
		return nil, err
	}
	var convertedResp v1.NodeUnprepareResourcesResponse
	if err := Convert_v1beta1_NodeUnprepareResourcesResponse_To_v1_NodeUnprepareResourcesResponse(resp, &convertedResp, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodeUnprepareResourcesResponse from v1beta1 to v1: %w", err)
	}
	return &convertedResp, nil
}

// V1ClientWrapper implements the [NodeClient] interface by wrapping a [v1.DRAPluginClient].
type V1ClientWrapper struct {
	v1.DRAPluginClient
}

var _ DRAPluginClient = V1ClientWrapper{}

func (w V1ClientWrapper) NodePrepareResources(ctx context.Context, req *NodePrepareResourcesRequest, options ...grpc.CallOption) (*NodePrepareResourcesResponse, error) {
	var convertedReq v1.NodePrepareResourcesRequest
	if err := Convert_v1beta1_NodePrepareResourcesRequest_To_v1_NodePrepareResourcesRequest(req, &convertedReq, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodePrepareResourcesRequest from v1beta1 to v1: %w", err)
	}
	resp, err := w.DRAPluginClient.NodePrepareResources(ctx, &convertedReq, options...)
	if err != nil {
		return nil, err
	}
	var convertedResp NodePrepareResourcesResponse
	if err := Convert_v1_NodePrepareResourcesResponse_To_v1beta1_NodePrepareResourcesResponse(resp, &convertedResp, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodePrepareResourcesResponse from v1 to v1beta1: %w", err)
	}
	return &convertedResp, nil
}

func (w V1ClientWrapper) NodeUnprepareResources(ctx context.Context, req *NodeUnprepareResourcesRequest, options ...grpc.CallOption) (*NodeUnprepareResourcesResponse, error) {
	var convertedReq v1.NodeUnprepareResourcesRequest
	if err := Convert_v1beta1_NodeUnprepareResourcesRequest_To_v1_NodeUnprepareResourcesRequest(req, &convertedReq, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodeUnprepareResourcesRequest from v1beta1 to v1: %w", err)
	}
	resp, err := w.DRAPluginClient.NodeUnprepareResources(ctx, &convertedReq, options...)
	if err != nil {
		return nil, err
	}
	var convertedResp NodeUnprepareResourcesResponse
	if err := Convert_v1_NodeUnprepareResourcesResponse_To_v1beta1_NodeUnprepareResourcesResponse(resp, &convertedResp, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodeUnprepareResourcesResponse from v1 to v1beta1: %w", err)
	}
	return &convertedResp, nil
}

// V1Beta1ClientWrapper implements the [v1.DRAPluginClient] interface by wrapping a [NodeClient].
type V1Beta1ClientWrapper struct {
	DRAPluginClient
}

var _ v1.DRAPluginClient = V1Beta1ClientWrapper{}

func (w V1Beta1ClientWrapper) NodePrepareResources(ctx context.Context, req *v1.NodePrepareResourcesRequest, options ...grpc.CallOption) (*v1.NodePrepareResourcesResponse, error) {
	var convertedReq NodePrepareResourcesRequest
	if err := Convert_v1_NodePrepareResourcesRequest_To_v1beta1_NodePrepareResourcesRequest(req, &convertedReq, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodePrepareResourcesRequest from v1 to v1beta1: %w", err)
	}
	resp, err := w.DRAPluginClient.NodePrepareResources(ctx, &convertedReq, options...)
	if err != nil {
		return nil, err
	}
	var convertedResp v1.NodePrepareResourcesResponse
	if err := Convert_v1beta1_NodePrepareResourcesResponse_To_v1_NodePrepareResourcesResponse(resp, &convertedResp, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodePrepareResourcesResponse from v1beta1 to v1: %w", err)
	}
	return &convertedResp, nil
}

func (w V1Beta1ClientWrapper) NodeUnprepareResources(ctx context.Context, req *v1.NodeUnprepareResourcesRequest, options ...grpc.CallOption) (*v1.NodeUnprepareResourcesResponse, error) {
	var convertedReq NodeUnprepareResourcesRequest
	if err := Convert_v1_NodeUnprepareResourcesRequest_To_v1beta1_NodeUnprepareResourcesRequest(req, &convertedReq, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodeUnprepareResourcesRequest from v1 to v1beta1: %w", err)
	}
	resp, err := w.DRAPluginClient.NodeUnprepareResources(ctx, &convertedReq, options...)
	if err != nil {
		return nil, err
	}
	var convertedResp v1.NodeUnprepareResourcesResponse
	if err := Convert_v1beta1_NodeUnprepareResourcesResponse_To_v1_NodeUnprepareResourcesResponse(resp, &convertedResp, nil); err != nil {
		return nil, fmt.Errorf("internal error converting NodeUnprepareResourcesResponse from v1beta1 to v1: %w", err)
	}
	return &convertedResp, nil
}
