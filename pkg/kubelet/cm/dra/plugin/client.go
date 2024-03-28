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

type (
	nodeResourceManager interface {
		Prepare(context.Context, *grpc.ClientConn, *plugin, *drapb.NodePrepareResourcesRequest) (*drapb.NodePrepareResourcesResponse, error)
		Unprepare(context.Context, *grpc.ClientConn, *plugin, *drapb.NodeUnprepareResourcesRequest) (*drapb.NodeUnprepareResourcesResponse, error)
	}

	v1alpha2NodeResourceManager struct{}
	v1alpha3NodeResourceManager struct{}
)

var nodeResourceManagers = map[string]nodeResourceManager{
	v1alpha2Version: v1alpha2NodeResourceManager{},
	v1alpha3Version: v1alpha3NodeResourceManager{},
}

func (v1alpha2rm v1alpha2NodeResourceManager) Prepare(ctx context.Context, conn *grpc.ClientConn, _ *plugin, req *drapb.NodePrepareResourcesRequest) (*drapb.NodePrepareResourcesResponse, error) {
	nodeClient := drapbv1alpha2.NewNodeClient(conn)
	response := &drapb.NodePrepareResourcesResponse{
		Claims: make(map[string]*drapb.NodePrepareResourceResponse),
	}

	for _, claim := range req.Claims {
		req := &drapbv1alpha2.NodePrepareResourceRequest{
			Namespace:                claim.Namespace,
			ClaimUid:                 claim.Uid,
			ClaimName:                claim.Name,
			ResourceHandle:           claim.ResourceHandle,
			StructuredResourceHandle: claim.StructuredResourceHandle,
		}
		res, err := nodeClient.NodePrepareResource(ctx, req)
		result := &drapb.NodePrepareResourceResponse{}
		if err != nil {
			result.Error = err.Error()
		} else {
			result.CDIDevices = res.CdiDevices
		}
		response.Claims[claim.Uid] = result
	}

	return response, nil
}

func (v1alpha2rm v1alpha2NodeResourceManager) Unprepare(ctx context.Context, conn *grpc.ClientConn, _ *plugin, req *drapb.NodeUnprepareResourcesRequest) (*drapb.NodeUnprepareResourcesResponse, error) {
	nodeClient := drapbv1alpha2.NewNodeClient(conn)
	response := &drapb.NodeUnprepareResourcesResponse{
		Claims: make(map[string]*drapb.NodeUnprepareResourceResponse),
	}

	for _, claim := range req.Claims {
		_, err := nodeClient.NodeUnprepareResource(ctx,
			&drapbv1alpha2.NodeUnprepareResourceRequest{
				Namespace:      claim.Namespace,
				ClaimUid:       claim.Uid,
				ClaimName:      claim.Name,
				ResourceHandle: claim.ResourceHandle,
			})
		result := &drapb.NodeUnprepareResourceResponse{}
		if err != nil {
			result.Error = err.Error()
		}
		response.Claims[claim.Uid] = result
	}

	return response, nil
}

func (v1alpha3rm v1alpha3NodeResourceManager) Prepare(ctx context.Context, conn *grpc.ClientConn, p *plugin, req *drapb.NodePrepareResourcesRequest) (*drapb.NodePrepareResourcesResponse, error) {
	nodeClient := drapb.NewNodeClient(conn)
	response, err := nodeClient.NodePrepareResources(ctx, req)
	if err != nil {
		status, _ := grpcstatus.FromError(err)
		if status.Code() == grpccodes.Unimplemented {
			p.setVersion(v1alpha2Version)
			return nodeResourceManagers[v1alpha2Version].Prepare(ctx, conn, p, req)
		}
		return nil, err
	}

	return response, nil
}

func (v1alpha3rm v1alpha3NodeResourceManager) Unprepare(ctx context.Context, conn *grpc.ClientConn, p *plugin, req *drapb.NodeUnprepareResourcesRequest) (*drapb.NodeUnprepareResourcesResponse, error) {
	nodeClient := drapb.NewNodeClient(conn)
	response, err := nodeClient.NodeUnprepareResources(ctx, req)
	if err != nil {
		status, _ := grpcstatus.FromError(err)
		if status.Code() == grpccodes.Unimplemented {
			p.setVersion(v1alpha2Version)
			return nodeResourceManagers[v1alpha2Version].Unprepare(ctx, conn, p, req)
		}
		return nil, err
	}

	return response, nil
}

func NewDRAPluginClient(pluginName string) (drapb.NodeClient, error) {
	if pluginName == "" {
		return nil, fmt.Errorf("plugin name is empty")
	}

	existingPlugin := draPlugins.get(pluginName)
	if existingPlugin == nil {
		return nil, fmt.Errorf("plugin name %s not found in the list of registered DRA plugins", pluginName)
	}

	return existingPlugin, nil
}

func (p *plugin) NodePrepareResources(
	ctx context.Context,
	req *drapb.NodePrepareResourcesRequest,
	opts ...grpc.CallOption,
) (*drapb.NodePrepareResourcesResponse, error) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info(log("calling NodePrepareResources rpc"), "request", req)

	conn, err := p.getOrCreateGRPCConn()
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, p.clientTimeout)
	defer cancel()

	version := p.getVersion()
	resourceManager, exists := nodeResourceManagers[version]
	if !exists {
		err := fmt.Errorf("unsupported plugin version: %s", version)
		logger.V(4).Info(log("done calling NodePrepareResources rpc"), "response", nil, "err", err)
		return nil, err
	}

	response, err := resourceManager.Prepare(ctx, conn, p, req)
	logger.V(4).Info(log("done calling NodePrepareResources rpc"), "response", response, "err", err)
	return response, err
}

func (p *plugin) NodeUnprepareResources(
	ctx context.Context,
	req *drapb.NodeUnprepareResourcesRequest,
	opts ...grpc.CallOption,
) (*drapb.NodeUnprepareResourcesResponse, error) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info(log("calling NodeUnprepareResource rpc"), "request", req)

	conn, err := p.getOrCreateGRPCConn()
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, p.clientTimeout)
	defer cancel()

	version := p.getVersion()
	resourceManager, exists := nodeResourceManagers[version]
	if !exists {
		err := fmt.Errorf("unsupported plugin version: %s", version)
		logger.V(4).Info(log("done calling NodeUnprepareResources rpc"), "response", nil, "err", err)
		return nil, err
	}

	response, err := resourceManager.Unprepare(ctx, conn, p, req)
	logger.V(4).Info(log("done calling NodeUnprepareResources rpc"), "response", response, "err", err)
	return response, err
}

func (p *plugin) NodeListAndWatchResources(
	ctx context.Context,
	req *drapb.NodeListAndWatchResourcesRequest,
	opts ...grpc.CallOption,
) (drapb.Node_NodeListAndWatchResourcesClient, error) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info(log("calling NodeListAndWatchResources rpc"), "request", req)

	conn, err := p.getOrCreateGRPCConn()
	if err != nil {
		return nil, err
	}

	nodeClient := drapb.NewNodeClient(conn)
	return nodeClient.NodeListAndWatchResources(ctx, req, opts...)
}
