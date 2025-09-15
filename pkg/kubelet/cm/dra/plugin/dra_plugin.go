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
	"google.golang.org/grpc/status"

	"k8s.io/klog/v2"
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
	drapbv1beta1.DRAPluginService,
}

// DRAPlugin contains information about one registered plugin of a DRA driver.
// It implements the kubelet operations for preparing/unpreparing by calling
// a gRPC interface that is implemented by the plugin.
type DRAPlugin struct {
	driverName        string
	conn              *grpc.ClientConn
	endpoint          string
	chosenService     string // e.g. drapbv1beta1.DRAPluginService
	clientCallTimeout time.Duration
}

func (p *DRAPlugin) DriverName() string {
	return p.driverName
}

func (p *DRAPlugin) NodePrepareResources(
	ctx context.Context,
	req *drapbv1beta1.NodePrepareResourcesRequest,
	opts ...grpc.CallOption,
) (*drapbv1beta1.NodePrepareResourcesResponse, error) {
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "driverName", p.driverName, "endpoint", p.endpoint)
	ctx = klog.NewContext(ctx, logger)
	logger.V(4).Info("Calling NodePrepareResources rpc", "request", req)

	ctx, cancel := context.WithTimeout(ctx, p.clientCallTimeout)
	defer cancel()

	var err error
	var response *drapbv1beta1.NodePrepareResourcesResponse
	switch p.chosenService {
	case drapbv1beta1.DRAPluginService:
		nodeClient := drapbv1beta1.NewDRAPluginClient(p.conn)
		response, err = nodeClient.NodePrepareResources(ctx, req)
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
	req *drapbv1beta1.NodeUnprepareResourcesRequest,
	opts ...grpc.CallOption,
) (*drapbv1beta1.NodeUnprepareResourcesResponse, error) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Calling NodeUnprepareResource rpc", "request", req)
	logger = klog.LoggerWithValues(logger, "driverName", p.driverName, "endpoint", p.endpoint)
	ctx = klog.NewContext(ctx, logger)

	ctx, cancel := context.WithTimeout(ctx, p.clientCallTimeout)
	defer cancel()

	var err error
	var response *drapbv1beta1.NodeUnprepareResourcesResponse
	switch p.chosenService {
	case drapbv1beta1.DRAPluginService:
		nodeClient := drapbv1beta1.NewDRAPluginClient(p.conn)
		response, err = nodeClient.NodeUnprepareResources(ctx, req)
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
