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

	"k8s.io/klog/v2"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1alpha3"
)

const PluginClientTimeout = 45 * time.Second

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

	nodeClient := drapb.NewNodeClient(conn)
	response, err := nodeClient.NodePrepareResources(ctx, req)
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

	nodeClient := drapb.NewNodeClient(conn)
	response, err := nodeClient.NodeUnprepareResources(ctx, req)
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
