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

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"

	drapbv1 "k8s.io/kubelet/pkg/apis/dra/v1alpha1"
)

type Client interface {
	NodePrepareResource(
		ctx context.Context,
		namespace string,
		claimUID types.UID,
		claimName string,
		resourceHandle string,
	) (*drapbv1.NodePrepareResourceResponse, error)

	NodeUnprepareResource(
		ctx context.Context,
		namespace string,
		claimUID types.UID,
		claimName string,
		cdiDevice []string,
	) (*drapbv1.NodeUnprepareResourceResponse, error)
}

// Strongly typed address.
type draAddr string

// draPluginClient encapsulates all dra plugin methods.
type draPluginClient struct {
	pluginName          string
	addr                draAddr
	nodeV1ClientCreator nodeV1ClientCreator
}

var _ Client = &draPluginClient{}

type nodeV1ClientCreator func(addr draAddr) (
	nodeClient drapbv1.NodeClient,
	closer io.Closer,
	err error,
)

// newV1NodeClient creates a new NodeClient with the internally used gRPC
// connection set up. It also returns a closer which must be called to close
// the gRPC connection when the NodeClient is not used anymore.
// This is the default implementation for the nodeV1ClientCreator, used in
// newDRAPluginClient.
func newV1NodeClient(addr draAddr) (nodeClient drapbv1.NodeClient, closer io.Closer, err error) {
	var conn *grpc.ClientConn

	conn, err = newGrpcConn(addr)
	if err != nil {
		return nil, nil, err
	}

	return drapbv1.NewNodeClient(conn), conn, nil
}

func NewDRAPluginClient(pluginName string) (Client, error) {
	if pluginName == "" {
		return nil, fmt.Errorf("plugin name is empty")
	}

	existingPlugin := draPlugins.Get(pluginName)
	if existingPlugin == nil {
		return nil, fmt.Errorf("plugin name %s not found in the list of registered DRA plugins", pluginName)
	}

	return &draPluginClient{
		pluginName:          pluginName,
		addr:                draAddr(existingPlugin.endpoint),
		nodeV1ClientCreator: newV1NodeClient,
	}, nil
}

func (r *draPluginClient) NodePrepareResource(
	ctx context.Context,
	namespace string,
	claimUID types.UID,
	claimName string,
	resourceHandle string,
) (*drapbv1.NodePrepareResourceResponse, error) {
	klog.V(4).InfoS(
		log("calling NodePrepareResource rpc"),
		"namespace", namespace,
		"claim UID", claimUID,
		"claim name", claimName,
		"resource handle", resourceHandle)

	if r.nodeV1ClientCreator == nil {
		return nil, errors.New("failed to call NodePrepareResource. nodeV1ClientCreator is nil")
	}

	nodeClient, closer, err := r.nodeV1ClientCreator(r.addr)
	if err != nil {
		return nil, err
	}
	defer closer.Close()

	req := &drapbv1.NodePrepareResourceRequest{
		Namespace:      namespace,
		ClaimUid:       string(claimUID),
		ClaimName:      claimName,
		ResourceHandle: resourceHandle,
	}

	return nodeClient.NodePrepareResource(ctx, req)
}

func (r *draPluginClient) NodeUnprepareResource(
	ctx context.Context,
	namespace string,
	claimUID types.UID,
	claimName string,
	cdiDevices []string,
) (*drapbv1.NodeUnprepareResourceResponse, error) {
	klog.V(4).InfoS(
		log("calling NodeUnprepareResource rpc"),
		"namespace", namespace,
		"claim UID", claimUID,
		"claim name", claimName,
		"cdi devices", cdiDevices)

	if r.nodeV1ClientCreator == nil {
		return nil, errors.New("nodeV1ClientCreate is nil")
	}

	nodeClient, closer, err := r.nodeV1ClientCreator(r.addr)
	if err != nil {
		return nil, err
	}
	defer closer.Close()

	req := &drapbv1.NodeUnprepareResourceRequest{
		Namespace:  namespace,
		ClaimUid:   string(claimUID),
		ClaimName:  claimName,
		CdiDevices: cdiDevices,
	}

	return nodeClient.NodeUnprepareResource(ctx, req)
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
