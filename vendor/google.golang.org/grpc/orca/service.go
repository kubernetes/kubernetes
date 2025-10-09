/*
 *
 * Copyright 2022 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package orca

import (
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/internal"
	ointernal "google.golang.org/grpc/orca/internal"
	"google.golang.org/grpc/status"

	v3orcaservicegrpc "github.com/cncf/xds/go/xds/service/orca/v3"
	v3orcaservicepb "github.com/cncf/xds/go/xds/service/orca/v3"
)

func init() {
	ointernal.AllowAnyMinReportingInterval = func(so *ServiceOptions) {
		so.allowAnyMinReportingInterval = true
	}
	internal.ORCAAllowAnyMinReportingInterval = ointernal.AllowAnyMinReportingInterval
}

// minReportingInterval is the absolute minimum value supported for
// out-of-band metrics reporting from the ORCA service implementation
// provided by the orca package.
const minReportingInterval = 30 * time.Second

// Service provides an implementation of the OpenRcaService as defined in the
// [ORCA] service protos. Instances of this type must be created via calls to
// Register() or NewService().
//
// Server applications can use the SetXxx() and DeleteXxx() methods to record
// measurements corresponding to backend metrics, which eventually get pushed to
// clients who have initiated the SteamCoreMetrics streaming RPC.
//
// [ORCA]: https://github.com/cncf/xds/blob/main/xds/service/orca/v3/orca.proto
type Service struct {
	v3orcaservicegrpc.UnimplementedOpenRcaServiceServer

	// Minimum reporting interval, as configured by the user, or the default.
	minReportingInterval time.Duration

	smProvider ServerMetricsProvider
}

// ServiceOptions contains options to configure the ORCA service implementation.
type ServiceOptions struct {
	// ServerMetricsProvider is the provider to be used by the service for
	// reporting OOB server metrics to clients.  Typically obtained via
	// NewServerMetricsRecorder.  This field is required.
	ServerMetricsProvider ServerMetricsProvider

	// MinReportingInterval sets the lower bound for how often out-of-band
	// metrics are reported on the streaming RPC initiated by the client. If
	// unspecified, negative or less than the default value of 30s, the default
	// is used. Clients may request a higher value as part of the
	// StreamCoreMetrics streaming RPC.
	MinReportingInterval time.Duration

	// Allow a minReportingInterval which is less than the default of 30s.
	// Used for testing purposes only.
	allowAnyMinReportingInterval bool
}

// A ServerMetricsProvider provides ServerMetrics upon request.
type ServerMetricsProvider interface {
	// ServerMetrics returns the current set of server metrics.  It should
	// return a read-only, immutable copy of the data that is active at the
	// time of the call.
	ServerMetrics() *ServerMetrics
}

// NewService creates a new ORCA service implementation configured using the
// provided options.
func NewService(opts ServiceOptions) (*Service, error) {
	// The default minimum supported reporting interval value can be overridden
	// for testing purposes through the orca internal package.
	if opts.ServerMetricsProvider == nil {
		return nil, fmt.Errorf("ServerMetricsProvider not specified")
	}
	if !opts.allowAnyMinReportingInterval {
		if opts.MinReportingInterval < 0 || opts.MinReportingInterval < minReportingInterval {
			opts.MinReportingInterval = minReportingInterval
		}
	}
	service := &Service{
		minReportingInterval: opts.MinReportingInterval,
		smProvider:           opts.ServerMetricsProvider,
	}
	return service, nil
}

// Register creates a new ORCA service implementation configured using the
// provided options and registers the same on the provided grpc Server.
func Register(s grpc.ServiceRegistrar, opts ServiceOptions) error {
	service, err := NewService(opts)
	if err != nil {
		return err
	}
	v3orcaservicegrpc.RegisterOpenRcaServiceServer(s, service)
	return nil
}

// determineReportingInterval determines the reporting interval for out-of-band
// metrics. If the reporting interval is not specified in the request, or is
// negative or is less than the configured minimum (via
// ServiceOptions.MinReportingInterval), the latter is used. Else the value from
// the incoming request is used.
func (s *Service) determineReportingInterval(req *v3orcaservicepb.OrcaLoadReportRequest) time.Duration {
	if req.GetReportInterval() == nil {
		return s.minReportingInterval
	}
	dur := req.GetReportInterval().AsDuration()
	if dur < s.minReportingInterval {
		logger.Warningf("Received reporting interval %q is less than configured minimum: %v. Using minimum", dur, s.minReportingInterval)
		return s.minReportingInterval
	}
	return dur
}

func (s *Service) sendMetricsResponse(stream v3orcaservicegrpc.OpenRcaService_StreamCoreMetricsServer) error {
	return stream.Send(s.smProvider.ServerMetrics().toLoadReportProto())
}

// StreamCoreMetrics streams custom backend metrics injected by the server
// application.
func (s *Service) StreamCoreMetrics(req *v3orcaservicepb.OrcaLoadReportRequest, stream v3orcaservicegrpc.OpenRcaService_StreamCoreMetricsServer) error {
	ticker := time.NewTicker(s.determineReportingInterval(req))
	defer ticker.Stop()

	for {
		if err := s.sendMetricsResponse(stream); err != nil {
			return err
		}
		// Send a response containing the currently recorded metrics
		select {
		case <-stream.Context().Done():
			return status.Error(codes.Canceled, "Stream has ended.")
		case <-ticker.C:
		}
	}
}
