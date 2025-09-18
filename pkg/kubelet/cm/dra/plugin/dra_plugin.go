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
	drapbv1 "k8s.io/kubelet/pkg/apis/dra/v1"
	drapbv1beta1 "k8s.io/kubelet/pkg/apis/dra/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

// defaultClientCallTimeout is the default amount of time that a DRA driver has
// to respond to any of the gRPC calls. kubelet uses this value by passing nil
// to RegisterPlugin. Some tests use a different, usually shorter timeout to
// speed up testing.
//
// This is half of the kubelet retry period (according to
// https://github.com/kubernetes/kubernetes/commit/0449cef8fd5217d394c5cd331d852bd50983e6b3).
const defaultClientCallTimeout = 45 * time.Second

// All API versions supported by this wrapper.
// Sorted by most recent first, oldest last.
var servicesSupportedByKubelet = []string{
	drapbv1.DRAPluginService,
	drapbv1beta1.DRAPluginService,
}

// DRAPlugin contains information about one registered plugin of a DRA driver.
// It implements the kubelet operations for preparing/unpreparing by calling
// a gRPC interface that is implemented by the plugin.
//
// This implementation uses a unified connection approach where both DRA APIs
// and health monitoring share the same gRPC connection and management logic,
// completely replacing the previous dual-connection pattern and eliminating
// all lazy initialization to address issues identified in #133943.
type DRAPlugin struct {
	driverName        string
	endpoint          string
	chosenService     string // e.g. drapbv1.DRAPluginService
	clientCallTimeout time.Duration

	// Unified connection management - all clients use same connection
	mutex         sync.Mutex
	backgroundCtx context.Context
	conn          *grpc.ClientConn

	// Pre-created clients - no lazy initialization
	draV1Client      drapbv1.DRAPluginClient
	draV1Beta1Client drapbv1beta1.DRAPluginClient
	healthClient     drahealthv1alpha1.DRAResourceHealthClient

	// Health stream management
	healthStreamCtx    context.Context
	healthStreamCancel context.CancelFunc
}

// ensureConnection ensures a healthy gRPC connection exists and all clients are initialized.
// This unified approach replaces both the previous getOrCreateGRPCConn() function and
// separate health connection logic, addressing the lazy initialization issues in #133943.
// All clients are created immediately when connection is established.
func (p *DRAPlugin) ensureConnection() error {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	// If connection exists and is healthy, all clients should already exist
	if p.conn != nil && p.conn.GetState() != connectivity.Shutdown {
		// Verify all clients exist (defensive check)
		if p.draV1Client == nil || p.draV1Beta1Client == nil || p.healthClient == nil {
			// This should not happen with proper initialization, but recover if needed
			p.createAllClients()
		}
		return nil
	}

	// Clean up any existing connection and clients
	if p.conn != nil {
		if err := p.conn.Close(); err != nil {
			klog.FromContext(p.backgroundCtx).Error(err, "Failed to close stale gRPC connection", "endpoint", p.endpoint)
		}
		p.clearAllClients()
	}

	// Create new connection
	conn, err := p.createConnection()
	if err != nil {
		return fmt.Errorf("failed to create gRPC connection to %s: %w", p.endpoint, err)
	}

	p.conn = conn
	// Immediately create all clients - no lazy initialization
	p.createAllClients()

	klog.FromContext(p.backgroundCtx).V(4).Info("Successfully established unified gRPC connection with all clients",
		"endpoint", p.endpoint, "driverName", p.driverName)

	return nil
}

// createConnection creates a new gRPC connection using modern APIs.
// Replaces deprecated grpc.Dial with grpc.NewClient as requested in review feedback.
func (p *DRAPlugin) createConnection() (*grpc.ClientConn, error) {
	logger := klog.FromContext(p.backgroundCtx)
	logger.V(4).Info("Creating new unified gRPC connection", "protocol", "unix", "endpoint", p.endpoint)

	network := "unix"

	// Use modern grpc.NewClient instead of deprecated grpc.Dial
	conn, err := grpc.NewClient(
		"unix://"+p.endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithChainUnaryInterceptor(newMetricsInterceptor(p.driverName)),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create connection: %w", err)
	}

	// Test connection readiness with a simple connectivity check
	ctx, cancel := context.WithTimeout(p.backgroundCtx, time.Second)
	defer cancel()

	// For unix sockets with grpc.NewClient, we need to test actual connectivity differently
	// since WaitForStateChange behavior differs. Instead, we'll test with a quick dial check.
	testConn, err := (&net.Dialer{}).DialContext(ctx, network, p.endpoint)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to verify socket connectivity: %w", err)
	}
	testConn.Close()

	return conn, nil
}

// createAllClients creates all necessary gRPC clients using the unified connection.
// Eliminates lazy initialization by creating all clients immediately upon connection establishment.
func (p *DRAPlugin) createAllClients() {
	if p.conn == nil {
		return
	}

	// Always create all clients immediately
	p.draV1Client = drapbv1.NewDRAPluginClient(p.conn)
	p.draV1Beta1Client = drapbv1beta1.NewDRAPluginClient(p.conn)
	p.healthClient = drahealthv1alpha1.NewDRAResourceHealthClient(p.conn)
}

// clearAllClients resets all client references when connection is closed.
func (p *DRAPlugin) clearAllClients() {
	p.conn = nil
	p.draV1Client = nil
	p.draV1Beta1Client = nil
	p.healthClient = nil
}

func newMetricsInterceptor(driverName string) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply any, conn *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		start := time.Now()
		err := invoker(ctx, method, req, reply, conn, opts...)
		metrics.DRAGRPCOperationsDuration.WithLabelValues(driverName, method, status.Code(err).String()).Observe(time.Since(start).Seconds())
		return err
	}
}

func (p *DRAPlugin) DriverName() string {
	return p.driverName
}

func (p *DRAPlugin) NodePrepareResources(
	ctx context.Context,
	req *drapbv1.NodePrepareResourcesRequest,
	opts ...grpc.CallOption,
) (*drapbv1.NodePrepareResourcesResponse, error) {
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "driverName", p.driverName, "endpoint", p.endpoint)
	ctx = klog.NewContext(ctx, logger)
	logger.V(4).Info("Calling NodePrepareResources rpc", "request", req)

	// Ensure unified connection and all clients are available
	if err := p.ensureConnection(); err != nil {
		return nil, fmt.Errorf("failed to ensure gRPC connection: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, p.clientCallTimeout)
	defer cancel()

	var response *drapbv1.NodePrepareResourcesResponse
	var err error

	// Use pre-created clients from unified connection - no more inline client creation
	switch p.chosenService {
	case drapbv1beta1.DRAPluginService:
		response, err = drapbv1beta1.V1Beta1ClientWrapper{DRAPluginClient: p.draV1Beta1Client}.NodePrepareResources(ctx, req)
	case drapbv1.DRAPluginService:
		response, err = p.draV1Client.NodePrepareResources(ctx, req)
	default:
		return nil, fmt.Errorf("internal error: unsupported chosen service: %q", p.chosenService)
	}

	logger.V(4).Info("Done calling NodePrepareResources rpc", "response", response, "err", err)
	return response, err
}

func (p *DRAPlugin) NodeUnprepareResources(
	ctx context.Context,
	req *drapbv1.NodeUnprepareResourcesRequest,
	opts ...grpc.CallOption,
) (*drapbv1.NodeUnprepareResourcesResponse, error) {
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "driverName", p.driverName, "endpoint", p.endpoint)
	ctx = klog.NewContext(ctx, logger)
	logger.V(4).Info("Calling NodeUnprepareResource rpc", "request", req)

	// Ensure unified connection and all clients are available
	if err := p.ensureConnection(); err != nil {
		return nil, fmt.Errorf("failed to ensure gRPC connection: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, p.clientCallTimeout)
	defer cancel()

	var response *drapbv1.NodeUnprepareResourcesResponse
	var err error

	// Use pre-created clients from unified connection - no more inline client creation
	switch p.chosenService {
	case drapbv1beta1.DRAPluginService:
		response, err = drapbv1beta1.V1Beta1ClientWrapper{DRAPluginClient: p.draV1Beta1Client}.NodeUnprepareResources(ctx, req)
	case drapbv1.DRAPluginService:
		response, err = p.draV1Client.NodeUnprepareResources(ctx, req)
	default:
		return nil, fmt.Errorf("internal error: unsupported chosen service: %q", p.chosenService)
	}

	logger.V(4).Info("Done calling NodeUnprepareResources rpc", "response", response, "err", err)
	return response, err
}

// SetHealthStream stores the context and cancel function for the active health stream.
func (p *DRAPlugin) SetHealthStream(ctx context.Context, cancel context.CancelFunc) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.healthStreamCtx = ctx
	p.healthStreamCancel = cancel
}

// HealthStreamCancel returns the cancel function for the current health stream, if any.
func (p *DRAPlugin) HealthStreamCancel() context.CancelFunc {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.healthStreamCancel
}

// NodeWatchResources establishes a stream to receive health updates from the DRA plugin.
// Now uses the unified connection management, completely eliminating separate health connection logic.
func (p *DRAPlugin) NodeWatchResources(ctx context.Context) (drahealthv1alpha1.DRAResourceHealth_NodeWatchResourcesClient, error) {
	// Ensure unified connection and all clients (including health client) are available
	if err := p.ensureConnection(); err != nil {
		klog.FromContext(p.backgroundCtx).Error(err, "Failed to ensure gRPC connection for health client")
		return nil, err
	}

	logger := klog.FromContext(ctx).WithValues("driverName", p.driverName)
	logger.V(4).Info("Starting WatchResources stream using unified connection")

	// Health client is guaranteed to exist after successful ensureConnection call
	stream, err := p.healthClient.NodeWatchResources(ctx, &drahealthv1alpha1.NodeWatchResourcesRequest{})
	if err != nil {
		logger.Error(err, "NodeWatchResources RPC call failed")
		return nil, err
	}

	logger.V(4).Info("NodeWatchResources stream initiated successfully")
	return stream, nil
}

// Close gracefully shuts down the plugin connection and cleans up all resources.
func (p *DRAPlugin) Close() error {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	// Clean up health stream
	if p.healthStreamCancel != nil {
		p.healthStreamCancel()
		p.healthStreamCancel = nil
		p.healthStreamCtx = nil
	}

	// Clean up connection and all clients
	if p.conn != nil {
		err := p.conn.Close()
		p.clearAllClients()
		return err
	}

	return nil
}
