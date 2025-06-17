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
type DRAPlugin struct {
	driverName        string
	conn              *grpc.ClientConn
	endpoint          string
	chosenService     string // e.g. drapbv1.DRAPluginService
	clientCallTimeout time.Duration

	mutex         sync.Mutex
	backgroundCtx context.Context

	healthClient       drahealthv1alpha1.DRAResourceHealthClient
	healthStreamCtx    context.Context
	healthStreamCancel context.CancelFunc
}

func (p *DRAPlugin) getOrCreateGRPCConn() (*grpc.ClientConn, error) {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	// If connection exists and is ready, return it.
	if p.conn != nil && p.conn.GetState() != connectivity.Shutdown {
		// Initialize health client if connection exists but client is nil
		// This allows lazy init if connection was established before health was added.
		if p.healthClient == nil {
			p.healthClient = drahealthv1alpha1.NewDRAResourceHealthClient(p.conn)
			klog.FromContext(p.backgroundCtx).V(4).Info("Initialized DRAResourceHealthClient lazily")
		}
		return p.conn, nil
	}

	// If the connection is dead, clean it up before creating a new one.
	if p.conn != nil {
		if err := p.conn.Close(); err != nil {
			return nil, fmt.Errorf("failed to close stale gRPC connection to %s: %w", p.endpoint, err)
		}
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
		grpc.WithChainUnaryInterceptor(newMetricsInterceptor(p.driverName)),
	)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	if ok := conn.WaitForStateChange(ctx, connectivity.Connecting); !ok {
		return nil, errors.New("timed out waiting for gRPC connection to be ready")
	}

	p.conn = conn
	p.healthClient = drahealthv1alpha1.NewDRAResourceHealthClient(p.conn)

	return p.conn, nil
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

	ctx, cancel := context.WithTimeout(ctx, p.clientCallTimeout)
	defer cancel()

	var err error
	var response *drapbv1.NodePrepareResourcesResponse
	switch p.chosenService {
	case drapbv1beta1.DRAPluginService:
		client := drapbv1beta1.NewDRAPluginClient(p.conn)
		response, err = drapbv1beta1.V1Beta1ClientWrapper{DRAPluginClient: client}.NodePrepareResources(ctx, req)
	case drapbv1.DRAPluginService:
		client := drapbv1.NewDRAPluginClient(p.conn)
		response, err = client.NodePrepareResources(ctx, req)
	default:
		// Shouldn't happen, validateSupportedServices should only
		// return services we support here.
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
	logger.V(4).Info("Calling NodeUnprepareResource rpc", "request", req)
	logger = klog.LoggerWithValues(logger, "driverName", p.driverName, "endpoint", p.endpoint)
	ctx = klog.NewContext(ctx, logger)

	ctx, cancel := context.WithTimeout(ctx, p.clientCallTimeout)
	defer cancel()

	var err error
	var response *drapbv1.NodeUnprepareResourcesResponse
	switch p.chosenService {
	case drapbv1beta1.DRAPluginService:
		client := drapbv1beta1.NewDRAPluginClient(p.conn)
		response, err = drapbv1beta1.V1Beta1ClientWrapper{DRAPluginClient: client}.NodeUnprepareResources(ctx, req)
	case drapbv1.DRAPluginService:
		client := drapbv1.NewDRAPluginClient(p.conn)
		response, err = client.NodeUnprepareResources(ctx, req)
	default:
		// Shouldn't happen, validateSupportedServices should only
		// return services we support here.
		return nil, fmt.Errorf("internal error: unsupported chosen service: %q", p.chosenService)
	}
	logger.V(4).Info("Done calling NodeUnprepareResources rpc", "response", response, "err", err)
	return response, err
}

func newMetricsInterceptor(driverName string) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply any, conn *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		start := time.Now()
		err := invoker(ctx, method, req, reply, conn, opts...)
		metrics.DRAGRPCOperationsDuration.WithLabelValues(driverName, method, status.Code(err).String()).Observe(time.Since(start).Seconds())
		return err
	}
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
func (p *DRAPlugin) NodeWatchResources(ctx context.Context) (drahealthv1alpha1.DRAResourceHealth_NodeWatchResourcesClient, error) {
	// Ensure a connection and the health client exist before proceeding.
	// This call is idempotent and will create them if they don't exist.
	_, err := p.getOrCreateGRPCConn()
	if err != nil {
		klog.FromContext(p.backgroundCtx).Error(err, "Failed to get gRPC connection for health client")
		return nil, err
	}

	logger := klog.FromContext(ctx).WithValues("pluginName", p.driverName)
	logger.V(4).Info("Starting WatchResources stream")
	stream, err := p.healthClient.NodeWatchResources(ctx, &drahealthv1alpha1.NodeWatchResourcesRequest{})
	if err != nil {
		logger.Error(err, "NodeWatchResources RPC call failed")
		return nil, err
	}

	logger.V(4).Info("NodeWatchResources stream initiated successfully")
	return stream, nil
}
