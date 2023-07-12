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

package plugin

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc"
	grpccodes "google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	drapbv1alpha2 "k8s.io/kubelet/pkg/apis/dra/v1alpha2"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1alpha3"
)

const PluginClientTimeout = 45 * time.Second

// draPluginClient encapsulates all dra plugin methods.
type draPluginClient struct {
	pluginName string
	plugin     *Plugin
}

func NewDRAPluginClient(pluginName string) (drapb.NodeClient, error) {
	if pluginName == "" {
		return nil, fmt.Errorf("plugin name is empty")
	}

	existingPlugin := draPlugins.Get(pluginName)
	if existingPlugin == nil {
		return nil, fmt.Errorf("plugin name %s not found in the list of registered DRA plugins", pluginName)
	}

	return &draPluginClient{
		pluginName: pluginName,
		plugin:     existingPlugin,
	}, nil
}

func (r *draPluginClient) NodePrepareResources(
	ctx context.Context,
	req *drapb.NodePrepareResourcesRequest,
	opts ...grpc.CallOption,
) (resp *drapb.NodePrepareResourcesResponse, err error) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info(log("calling NodePrepareResources rpc"), "request", req)
	defer logger.V(4).Info(log("done calling NodePrepareResources rpc"), "response", resp, "err", err)

	conn, err := r.plugin.GetOrCreateGRPCConn()
	if err != nil {
		return nil, err
	}
	nodeClient := drapb.NewNodeClient(conn)
	nodeClientOld := drapbv1alpha2.NewNodeClient(conn)

	ctx, cancel := context.WithTimeout(ctx, PluginClientTimeout)
	defer cancel()

	resp, err = nodeClient.NodePrepareResources(ctx, req)
	if err != nil {
		status, _ := grpcstatus.FromError(err)
		if status.Code() == grpccodes.Unimplemented {
			// Fall back to the older gRPC API.
			resp = &drapb.NodePrepareResourcesResponse{
				Claims: make(map[string]*drapb.NodePrepareResourceResponse),
			}
			err = nil
			for _, claim := range req.Claims {
				respOld, errOld := nodeClientOld.NodePrepareResource(ctx,
					&drapbv1alpha2.NodePrepareResourceRequest{
						Namespace:      claim.Namespace,
						ClaimUid:       claim.Uid,
						ClaimName:      claim.Name,
						ResourceHandle: claim.ResourceHandle,
					})
				result := &drapb.NodePrepareResourceResponse{}
				if errOld != nil {
					result.Error = errOld.Error()
				} else {
					result.CDIDevices = respOld.CdiDevices
				}
				resp.Claims[claim.Uid] = result
			}
		}
	}

	return
}

func (r *draPluginClient) NodeUnprepareResources(
	ctx context.Context,
	req *drapb.NodeUnprepareResourcesRequest,
	opts ...grpc.CallOption,
) (resp *drapb.NodeUnprepareResourcesResponse, err error) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info(log("calling NodeUnprepareResource rpc"), "request", req)
	defer logger.V(4).Info(log("done calling NodeUnprepareResources rpc"), "response", resp, "err", err)

	conn, err := r.plugin.GetOrCreateGRPCConn()
	if err != nil {
		return nil, err
	}
	nodeClient := drapb.NewNodeClient(conn)
	nodeClientOld := drapbv1alpha2.NewNodeClient(conn)

	ctx, cancel := context.WithTimeout(ctx, PluginClientTimeout)
	defer cancel()

	resp, err = nodeClient.NodeUnprepareResources(ctx, req)
	if err != nil {
		status, _ := grpcstatus.FromError(err)
		if status.Code() == grpccodes.Unimplemented {
			// Fall back to the older gRPC API.
			resp = &drapb.NodeUnprepareResourcesResponse{
				Claims: make(map[string]*drapb.NodeUnprepareResourceResponse),
			}
			err = nil
			for _, claim := range req.Claims {
				_, errOld := nodeClientOld.NodeUnprepareResource(ctx,
					&drapbv1alpha2.NodeUnprepareResourceRequest{
						Namespace:      claim.Namespace,
						ClaimUid:       claim.Uid,
						ClaimName:      claim.Name,
						ResourceHandle: claim.ResourceHandle,
					})
				result := &drapb.NodeUnprepareResourceResponse{}
				if errOld != nil {
					result.Error = errOld.Error()
				}
				resp.Claims[claim.Uid] = result
			}
		}
	}

	return
}
