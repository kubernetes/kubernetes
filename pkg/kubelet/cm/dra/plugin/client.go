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

package dra

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"

	drapbv1 "k8s.io/kubernetes/pkg/kubelet/apis/dra/v1alpha1"
)

type DRAClient interface {
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

// Strongly typed address
type draAddr string

// draPluginClient encapsulates all dra plugin methods
type draPluginClient struct {
	pluginName          string
	addr                draAddr
	nodeV1ClientCreator nodeV1ClientCreator
}

var _ DRAClient = &draPluginClient{}

type nodeV1ClientCreator func(addr draAddr) (
	nodeClient drapbv1.NodeClient,
	closer io.Closer,
	err error,
)

// UncertainProgressError indicates operation failed with a non-final error
// and operation may be in-progress in background.
type UncertainProgressError struct {
	msg string
}

func (err *UncertainProgressError) Error() string {
	return err.msg
}

// NewUncertainProgressError creates an instance of UncertainProgressError type
func NewUncertainProgressError(msg string) *UncertainProgressError {
	return &UncertainProgressError{msg: msg}
}

// newV1NodeClient creates a new NodeClient with the internally used gRPC
// connection set up. It also returns a closer which must be called to close
// the gRPC connection when the NodeClient is not used anymore.
// This is the default implementation for the nodeV1ClientCreator, used in
// newCsiDriverClient.
func newV1NodeClient(addr draAddr) (nodeClient drapbv1.NodeClient, closer io.Closer, err error) {
	var conn *grpc.ClientConn
	conn, err = newGrpcConn(addr)
	if err != nil {
		return nil, nil, err
	}

	nodeClient = drapbv1.NewNodeClient(conn)
	return nodeClient, conn, nil
}

func NewDRAPluginClient(pluginName string) (DRAClient, error) {
	if pluginName == "" {
		return nil, fmt.Errorf("plugin name is empty")
	}

	existingDriver, driverExists := draPlugins.Get(string(pluginName))
	if !driverExists {
		return nil, fmt.Errorf("plugin name %s not found in the list of registered DRA plugins", pluginName)
	}

	nodeV1ClientCreator := newV1NodeClient
	return &draPluginClient{
		pluginName:          pluginName,
		addr:                draAddr(existingDriver.endpoint),
		nodeV1ClientCreator: nodeV1ClientCreator,
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
		ClaimName:      string(claimName),
		ResourceHandle: resourceHandle,
	}

	response, err := nodeClient.NodePrepareResource(ctx, req)
	if err != nil && !isFinalError(err) {
		return nil, NewUncertainProgressError(err.Error())
	}
	return response, err
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
		Namespace: namespace,
		ClaimUid:  string(claimUID),
		ClaimName: string(claimName),
		CdiDevice: cdiDevices,
	}

	response, err := nodeClient.NodeUnprepareResource(ctx, req)
	if err != nil && !isFinalError(err) {
		return nil, NewUncertainProgressError(err.Error())
	}

	return response, err
}

func newGrpcConn(addr draAddr) (*grpc.ClientConn, error) {
	network := "unix"
	klog.V(4).InfoS(log("creating new gRPC connection"), "protocol", network, "endpoint", addr)

	return grpc.Dial(
		string(addr),
		grpc.WithInsecure(),
		grpc.WithContextDialer(func(ctx context.Context, target string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, target)
		}),
	)
}

func isFinalError(err error) bool {
	st, ok := status.FromError(err)
	if !ok {
		// This is not gRPC error. The operation must have failed before gRPC
		// method was called, otherwise we would get gRPC error.
		// We don't know if any previous volume operation is in progress, be on the safe side.
		return false
	}
	switch st.Code() {
	case codes.Canceled, // gRPC: Client Application cancelled the request
		codes.DeadlineExceeded,  // gRPC: Timeout
		codes.Unavailable,       // gRPC: Server shutting down, TCP connection broken - previous volume operation may be still in progress.
		codes.ResourceExhausted, // gRPC: Server temporarily out of resources - previous volume operation may be still in progress.
		codes.Aborted:           // CSI: Operation pending for volume
		return false
	}
	// All other errors mean that operation either did not
	// even start or failed. It is for sure not in progress.
	return true
}
