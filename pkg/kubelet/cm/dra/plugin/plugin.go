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
	"net/url"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"

	"k8s.io/klog/v2"
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

	connectedPlugin := draPlugins.get(pluginName)
	if connectedPlugin == nil {
		return nil, fmt.Errorf("plugin %s not found in the list of registered DRA plugins or not connected", pluginName)
	}

	return connectedPlugin, nil
}

type Plugin struct {
	name          string
	backgroundCtx context.Context
	cancel        func(cause error)

	mutex               sync.Mutex
	conn                *grpc.ClientConn
	connected           bool
	endpoint            string
	chosenService       string // e.g. drapbv1beta1.DRAPluginService
	registrationHandler *RegistrationHandler
	clientCallTimeout   time.Duration
	cancelCleanup       *context.CancelCauseFunc
}

// getOrCreateGRPCConn creates and triggers a gRPC client connection to the
// plugin's endpoint. If a connection already exists, it returns the
// existing connection. Otherwise, it creates a new connection using the
// specified endpoint and configuration, including transport credentials,
// context dialer for Unix sockets, unary interceptor for metrics, and a stats
// handler. The method is thread-safe and ensures only one connection is
// created per plugin instance.
// Returns the gRPC client connection or an error if the connection could
// not be created.
// NOTE: This method doesn't wait for the connection to be established.
// It only creates the connection and returns it immediately.
func (p *Plugin) getOrCreateGRPCConn() (*grpc.ClientConn, error) {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	if p.conn != nil {
		return p.conn, nil
	}

	ctx := p.backgroundCtx
	logger := klog.FromContext(ctx)

	network := "unix"
	target := (&url.URL{Scheme: network, Path: p.endpoint}).String()
	logger.V(4).Info("Creating new gRPC connection", "protocol", network, "endpoint", p.endpoint)

	conn, err := grpc.NewClient(target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithStatsHandler(p),
		grpc.WithChainUnaryInterceptor(newMetricsInterceptor(p.name)),
	)

	if err != nil {
		logger.V(4).Info("failed to create gRPC connection", "plugin", p.name, "endpoint", p.endpoint)
		return nil, fmt.Errorf("failed to create gRPC connection to plugin %s at endpoint %s: %w", p.name, p.endpoint, err)
	}

	conn.Connect() // Trigger the connection immediately

	logger.V(4).Info("gRPC connection created", "plugin", p.name, "endpoint", p.endpoint, "state", conn.GetState())

	p.conn = conn
	return p.conn, nil
}

// isConnected returns true if the plugin is currently connected.
func (p *Plugin) isConnected() bool {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.connected
}

// setConnected updates the connection status of the plugin.
// Depending on the new connection state, it either cancels any pending
// resource wipes (if connected) or initiates cleanup of resource slices
// (if disconnected).
func (p *Plugin) setConnected(connected bool) {
	p.mutex.Lock()
	p.connected = connected
	p.mutex.Unlock()
	if p.registrationHandler != nil {
		if connected {
			p.registrationHandler.cancelPendingWipes(p.name, "connection established")
		} else {
			p.setCancelCleanup(p.registrationHandler.cleanupResourceSlices(p.name))
		}
	}
}

// setCancelCleanup sets the cancelCleanup function for the Plugin instance.
// The cancelCleanup parameter is a pointer to a context.CancelCauseFunc,
// which can be used to cancel and clean up resources associated with the Plugin.
func (p *Plugin) setCancelCleanup(cancelClenup *context.CancelCauseFunc) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	if p.cancelCleanup != nil {
		// If cancelCleanup is already set, we don't need to do anything.
		return
	}
	p.cancelCleanup = cancelClenup
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

// HandleConn implements the grpc stats.Handler interface for connection events.
// It is called by gRPC when a connection begins or ends. On connection begin,
// it marks the plugin as connected and cancels any pending cleanup. On
// connection end, it marks the plugin as disconnected, triggers cleanup and
// initiates gRPC re-connection.
func (p *Plugin) HandleConn(ctx context.Context, connStats stats.ConnStats) {
	logger := klog.FromContext(ctx)

	if !draPlugins.pluginRegistered(p.name, p.endpoint) {
		logger.V(2).Info("Plugin not registered, skipping connection stats handling", "plugin", p.name, "endpoint", p.endpoint)
		return
	}

	switch connStats.(type) {
	case *stats.ConnBegin:
		logger.V(2).Info("Connection begin", "plugin", p.name, "endpoint", p.endpoint, "stats", connStats)
		p.setConnected(true)
	case *stats.ConnEnd:
		logger.V(2).Info("Connection end", "plugin", p.name, "endpoint", p.endpoint, "stats", connStats)
		p.setConnected(false)
		if p.conn != nil {
			logger.V(2).Info("Trigger reconnecting to plugin", "plugin", p.name, "endpoint", p.endpoint)
			p.conn.Connect()
		}
	}
}

func (p *Plugin) HandleRPC(ctx context.Context, stats stats.RPCStats)                  {}
func (p *Plugin) TagConn(ctx context.Context, info *stats.ConnTagInfo) context.Context { return ctx }
func (p *Plugin) TagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context   { return ctx }
