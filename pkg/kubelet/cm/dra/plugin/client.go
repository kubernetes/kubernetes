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
	"errors"
	"fmt"
	"io"
	"net"
	"time"

	"google.golang.org/grpc"
	grpccodes "google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	grpcstatus "google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	drapbv1alpha2 "k8s.io/kubelet/pkg/apis/dra/v1alpha2"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1alpha3"
)

const PluginClientTimeout = 45 * time.Second

// Strongly typed address.
type draAddr string

// draPluginClient encapsulates all dra plugin methods.
type draPluginClient struct {
	pluginName        string
	addr              draAddr
	nodeClientCreator nodeClientCreator
}

var _ drapb.NodeClient = &draPluginClient{}

type nodeClientCreator func(addr draAddr) (
	nodeClient drapb.NodeClient,
	nodeClientOld drapbv1alpha2.NodeClient,
	closer io.Closer,
	err error,
)

// newNodeClient creates a new NodeClient with the internally used gRPC
// connection set up. It also returns a closer which must be called to close
// the gRPC connection when the NodeClient is not used anymore.
// This is the default implementation for the nodeClientCreator, used in
// newDRAPluginClient.
func newNodeClient(addr draAddr) (nodeClient drapb.NodeClient, nodeClientOld drapbv1alpha2.NodeClient, closer io.Closer, err error) {
	var conn *grpc.ClientConn

	conn, err = newGrpcConn(addr)
	if err != nil {
		return nil, nil, nil, err
	}

	return drapb.NewNodeClient(conn), drapbv1alpha2.NewNodeClient(conn), conn, nil
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
		pluginName:        pluginName,
		addr:              draAddr(existingPlugin.endpoint),
		nodeClientCreator: newNodeClient,
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

	if r.nodeClientCreator == nil {
		return nil, errors.New("failed to call NodePrepareResources. nodeClientCreator is nil")
	}

	nodeClient, nodeClientOld, closer, err := r.nodeClientCreator(r.addr)
	if err != nil {
		return nil, err
	}
	defer closer.Close()

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

	if r.nodeClientCreator == nil {
		return nil, errors.New("failed to call NodeUnprepareResources. nodeClientCreator is nil")
	}

	nodeClient, nodeClientOld, closer, err := r.nodeClientCreator(r.addr)
	if err != nil {
		return nil, err
	}
	defer closer.Close()

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

func newGrpcConn(addr draAddr) (*grpc.ClientConn, error) {
	network := "unix"
	klog.V(4).InfoS(log("creating new gRPC connection"), "protocol", network, "endpoint", addr)

	return grpc.Dial(
		string(addr),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, target string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, target)
		}),
	)
}
