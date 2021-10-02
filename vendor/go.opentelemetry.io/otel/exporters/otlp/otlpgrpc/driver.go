// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package otlpgrpc // import "go.opentelemetry.io/otel/exporters/otlp/otlpgrpc"

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"go.opentelemetry.io/otel/exporters/otlp/internal/otlpconfig"

	"google.golang.org/grpc"

	"go.opentelemetry.io/otel/exporters/otlp"
	"go.opentelemetry.io/otel/exporters/otlp/internal/transform"
	metricsdk "go.opentelemetry.io/otel/sdk/export/metric"
	tracesdk "go.opentelemetry.io/otel/sdk/trace"
	colmetricpb "go.opentelemetry.io/proto/otlp/collector/metrics/v1"
	coltracepb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	metricpb "go.opentelemetry.io/proto/otlp/metrics/v1"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
)

type driver struct {
	metricsDriver metricsDriver
	tracesDriver  tracesDriver
}

type metricsDriver struct {
	connection *connection

	lock          sync.Mutex
	metricsClient colmetricpb.MetricsServiceClient
}

type tracesDriver struct {
	connection *connection

	lock         sync.Mutex
	tracesClient coltracepb.TraceServiceClient
}

var (
	errNoClient = errors.New("no client")
)

// NewDriver creates a new gRPC protocol driver.
func NewDriver(opts ...Option) otlp.ProtocolDriver {
	cfg := otlpconfig.NewDefaultConfig()
	otlpconfig.ApplyGRPCEnvConfigs(&cfg)
	for _, opt := range opts {
		opt.ApplyGRPCOption(&cfg)
	}

	d := &driver{}

	d.tracesDriver = tracesDriver{
		connection: newConnection(cfg, cfg.Traces, d.tracesDriver.handleNewConnection),
	}

	d.metricsDriver = metricsDriver{
		connection: newConnection(cfg, cfg.Metrics, d.metricsDriver.handleNewConnection),
	}
	return d
}

func (md *metricsDriver) handleNewConnection(cc *grpc.ClientConn) {
	md.lock.Lock()
	defer md.lock.Unlock()
	if cc != nil {
		md.metricsClient = colmetricpb.NewMetricsServiceClient(cc)
	} else {
		md.metricsClient = nil
	}
}

func (td *tracesDriver) handleNewConnection(cc *grpc.ClientConn) {
	td.lock.Lock()
	defer td.lock.Unlock()
	if cc != nil {
		td.tracesClient = coltracepb.NewTraceServiceClient(cc)
	} else {
		td.tracesClient = nil
	}
}

// Start implements otlp.ProtocolDriver. It establishes a connection
// to the collector.
func (d *driver) Start(ctx context.Context) error {
	d.tracesDriver.connection.startConnection(ctx)
	d.metricsDriver.connection.startConnection(ctx)
	return nil
}

// Stop implements otlp.ProtocolDriver. It shuts down the connection
// to the collector.
func (d *driver) Stop(ctx context.Context) error {
	if err := d.tracesDriver.connection.shutdown(ctx); err != nil {
		return err
	}

	return d.metricsDriver.connection.shutdown(ctx)
}

// ExportMetrics implements otlp.ProtocolDriver. It transforms metrics
// to protobuf binary format and sends the result to the collector.
func (d *driver) ExportMetrics(ctx context.Context, cps metricsdk.CheckpointSet, selector metricsdk.ExportKindSelector) error {
	if !d.metricsDriver.connection.connected() {
		return fmt.Errorf("metrics exporter is disconnected from the server %s: %w", d.metricsDriver.connection.sCfg.Endpoint, d.metricsDriver.connection.lastConnectError())
	}
	ctx, cancel := d.metricsDriver.connection.contextWithStop(ctx)
	defer cancel()
	ctx, tCancel := context.WithTimeout(ctx, d.metricsDriver.connection.sCfg.Timeout)
	defer tCancel()

	rms, err := transform.CheckpointSet(ctx, selector, cps, 1)
	if err != nil {
		return err
	}
	if len(rms) == 0 {
		return nil
	}

	return d.metricsDriver.uploadMetrics(ctx, rms)
}

func (md *metricsDriver) uploadMetrics(ctx context.Context, protoMetrics []*metricpb.ResourceMetrics) error {
	ctx = md.connection.contextWithMetadata(ctx)
	err := func() error {
		md.lock.Lock()
		defer md.lock.Unlock()
		if md.metricsClient == nil {
			return errNoClient
		}
		_, err := md.metricsClient.Export(ctx, &colmetricpb.ExportMetricsServiceRequest{
			ResourceMetrics: protoMetrics,
		})
		return err
	}()
	if err != nil {
		md.connection.setStateDisconnected(err)
	}
	return err
}

// ExportTraces implements otlp.ProtocolDriver. It transforms spans to
// protobuf binary format and sends the result to the collector.
func (d *driver) ExportTraces(ctx context.Context, ss []*tracesdk.SpanSnapshot) error {
	if !d.tracesDriver.connection.connected() {
		return fmt.Errorf("traces exporter is disconnected from the server %s: %w", d.tracesDriver.connection.sCfg.Endpoint, d.tracesDriver.connection.lastConnectError())
	}
	ctx, cancel := d.tracesDriver.connection.contextWithStop(ctx)
	defer cancel()
	ctx, tCancel := context.WithTimeout(ctx, d.tracesDriver.connection.sCfg.Timeout)
	defer tCancel()

	protoSpans := transform.SpanData(ss)
	if len(protoSpans) == 0 {
		return nil
	}

	return d.tracesDriver.uploadTraces(ctx, protoSpans)
}

func (td *tracesDriver) uploadTraces(ctx context.Context, protoSpans []*tracepb.ResourceSpans) error {
	ctx = td.connection.contextWithMetadata(ctx)
	err := func() error {
		td.lock.Lock()
		defer td.lock.Unlock()
		if td.tracesClient == nil {
			return errNoClient
		}
		_, err := td.tracesClient.Export(ctx, &coltracepb.ExportTraceServiceRequest{
			ResourceSpans: protoSpans,
		})
		return err
	}()
	if err != nil {
		td.connection.setStateDisconnected(err)
	}
	return err
}
