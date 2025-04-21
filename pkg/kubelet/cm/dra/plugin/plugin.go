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
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	"k8s.io/klog/v2"
	drahealthv1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
	drapbv1alpha4 "k8s.io/kubelet/pkg/apis/dra/v1alpha4"
	drapbv1beta1 "k8s.io/kubelet/pkg/apis/dra/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

// NewDRAPluginClient returns a wrapper around those gRPC methods of a DRA
// driver kubelet plugin which need to be called by kubelet. The wrapper
// handles gRPC connection management and logging. Connections are reused
// across different NewDRAPluginClient calls.
func NewDRAPluginClient(pluginName string) (*Plugin, error) {
	if pluginName == "" {
		return nil, fmt.Errorf("plugin name is empty")
	}

	existingPlugin := draPlugins.get(pluginName)
	if existingPlugin == nil {
		return nil, fmt.Errorf("plugin name %s not found in the list of registered DRA plugins", pluginName)
	}

	return existingPlugin, nil
}

type Plugin struct {
	name          string
	backgroundCtx context.Context
	cancel        func(cause error)

	mutex             sync.Mutex
	conn              *grpc.ClientConn
	endpoint          string
	chosenService     string // e.g. drapbv1beta1.DRAPluginService
	clientCallTimeout time.Duration

	healthClient       drahealthv1alpha1.NodeHealthClient
	healthStreamCtx    context.Context
	healthStreamCancel context.CancelFunc
}

func (p *Plugin) getOrCreateGRPCConn() (*grpc.ClientConn, error) {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	// If connection exists and is ready, return it.
	if p.conn != nil && p.conn.GetState() != connectivity.Shutdown {
		// Initialize health client if connection exists but client is nil
		// This allows lazy init if connection was established before health was added.
		if p.healthClient == nil {
			p.healthClient = drahealthv1alpha1.NewNodeHealthClient(p.conn)
			klog.FromContext(p.backgroundCtx).V(4).Info("Initialized NodeHealthClient lazily")
		}
		return p.conn, nil
	}

	if p.conn != nil {
		p.conn.Close()
		p.conn = nil
		p.healthClient = nil
	}

	ctx := p.backgroundCtx
	logger := klog.FromContext(ctx)

	network := "unix"
	logger.V(4).Info("Creating new gRPC connection", "protocol", network, "endpoint", p.endpoint)
	// grpc.Dial is deprecated. grpc.NewClient should be used instead.
	// For now this gets ignored because this function is meant to establish
	// the connection, with the one second timeout below. Perhaps that
	// approach should be reconsidered?
	//nolint:staticcheck
	conn, err := grpc.Dial(
		p.endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, target string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, target)
		}),
		grpc.WithChainUnaryInterceptor(newMetricsInterceptor(p.name)),
		grpc.WithBlock(),
	)
	if err != nil {
		// Error could be context deadline exceeded or connection refused etc.
		logger.Error(err, "Failed to dial gRPC connection")
		return nil, fmt.Errorf("failed to dial %s: %w", p.endpoint, err)
	}

	logger.V(4).Info("Successfully established gRPC connection", "endpoint", p.endpoint)
	p.conn = conn
	// Initialize clients now that connection is established
	p.healthClient = drahealthv1alpha1.NewNodeHealthClient(p.conn)

	return p.conn, nil
}

// SetHealthStream stores the context and cancel function for the active health stream.
func (p *Plugin) SetHealthStream(ctx context.Context, cancel context.CancelFunc) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.healthStreamCtx = ctx
	p.healthStreamCancel = cancel
}

// HealthStreamCancel returns the cancel function for the current health stream, if any.
func (p *Plugin) HealthStreamCancel() context.CancelFunc {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.healthStreamCancel
}

// HealthClient returns the NodeHealthClient, ensuring it's initialized.
// Returns nil if connection cannot be established.
func (p *Plugin) HealthClient() drahealthv1alpha1.NodeHealthClient {
	// Ensure connection and client are initialized
	_, err := p.getOrCreateGRPCConn()
	if err != nil {
		klog.FromContext(p.backgroundCtx).Error(err, "Failed to get gRPC connection for health client")
		return nil
	}
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.healthClient
}

func (p *Plugin) NodePrepareResources(
	ctx context.Context,
	req *drapbv1beta1.NodePrepareResourcesRequest,
	opts ...grpc.CallOption,
) (*drapbv1beta1.NodePrepareResourcesResponse, error) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Calling NodePrepareResources rpc", "request", req)

	conn, err := p.getOrCreateGRPCConn()
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, p.clientCallTimeout)
	defer cancel()

	var response *drapbv1beta1.NodePrepareResourcesResponse
	switch p.chosenService {
	case drapbv1beta1.DRAPluginService:
		nodeClient := drapbv1beta1.NewDRAPluginClient(conn)
		response, err = nodeClient.NodePrepareResources(ctx, req)
	case drapbv1alpha4.NodeService:
		nodeClient := drapbv1alpha4.NewNodeClient(conn)
		response, err = drapbv1alpha4.V1Alpha4ClientWrapper{NodeClient: nodeClient}.NodePrepareResources(ctx, req)
	default:
		// Shouldn't happen, validateSupportedServices should only
		// return services we support here.
		return nil, fmt.Errorf("internal error: unsupported chosen service: %q", p.chosenService)
	}
	logger.V(4).Info("Done calling NodePrepareResources rpc", "response", response, "err", err)
	return response, err
}

func (p *Plugin) NodeUnprepareResources(
	ctx context.Context,
	req *drapbv1beta1.NodeUnprepareResourcesRequest,
	opts ...grpc.CallOption,
) (*drapbv1beta1.NodeUnprepareResourcesResponse, error) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Calling NodeUnprepareResource rpc", "request", req)

	conn, err := p.getOrCreateGRPCConn()
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, p.clientCallTimeout)
	defer cancel()

	var response *drapbv1beta1.NodeUnprepareResourcesResponse
	switch p.chosenService {
	case drapbv1beta1.DRAPluginService:
		nodeClient := drapbv1beta1.NewDRAPluginClient(conn)
		response, err = nodeClient.NodeUnprepareResources(ctx, req)
	case drapbv1alpha4.NodeService:
		nodeClient := drapbv1alpha4.NewNodeClient(conn)
		response, err = drapbv1alpha4.V1Alpha4ClientWrapper{NodeClient: nodeClient}.NodeUnprepareResources(ctx, req)
	default:
		// Shouldn't happen, validateSupportedServices should only
		// return services we support here.
		return nil, fmt.Errorf("internal error: unsupported chosen service: %q", p.chosenService)
	}
	logger.V(4).Info("Done calling NodeUnprepareResources rpc", "response", response, "err", err)
	return response, err
}

func newMetricsInterceptor(pluginName string) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply any, conn *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		start := time.Now()
		err := invoker(ctx, method, req, reply, conn, opts...)
		metrics.DRAGRPCOperationsDuration.WithLabelValues(pluginName, method, status.Code(err).String()).Observe(time.Since(start).Seconds())
		return err
	}
}

// WatchResources establishes a stream to receive health updates from the DRA plugin.
func (p *Plugin) WatchResources(ctx context.Context) (drahealthv1alpha1.NodeHealth_WatchResourcesClient, error) {
	if p.healthClient == nil {
		return nil, fmt.Errorf("health client not initialized for plugin %s", p.name)
	}
	logger := klog.FromContext(ctx).V(4).WithValues("pluginName", p.name)
	logger.Info("Starting WatchResources stream")
	stream, err := p.healthClient.WatchResources(ctx, &drahealthv1alpha1.WatchResourcesRequest{})
	if err != nil {
		logger.Error(err, "WatchResources RPC call failed")
		return nil, err // Return original gRPC error
	}

	logger.V(4).Info("WatchResources stream initiated successfully")
	return stream, nil
}
