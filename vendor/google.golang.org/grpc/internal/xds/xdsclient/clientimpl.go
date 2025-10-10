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

package xdsclient

import (
	"fmt"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	estats "google.golang.org/grpc/experimental/stats"
	"google.golang.org/grpc/internal/backoff"
	"google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/xds/bootstrap"

	"google.golang.org/grpc/internal/xds/clients"
	"google.golang.org/grpc/internal/xds/clients/grpctransport"
	"google.golang.org/grpc/internal/xds/clients/lrsclient"
	"google.golang.org/grpc/internal/xds/clients/xdsclient"
	"google.golang.org/grpc/internal/xds/clients/xdsclient/metrics"
	xdsbootstrap "google.golang.org/grpc/xds/bootstrap"
)

const (
	// NameForServer represents the value to be passed as name when creating an xDS
	// client from xDS-enabled gRPC servers. This is a well-known dedicated key
	// value, and is defined in gRFC A71.
	NameForServer = "#server"

	defaultWatchExpiryTimeout = 15 * time.Second
)

var (
	// The following functions are no-ops in the actual code, but can be
	// overridden in tests to give them visibility into certain events.
	xdsClientImplCreateHook = func(string) {}
	xdsClientImplCloseHook  = func(string) {}

	defaultExponentialBackoff = backoff.DefaultExponential.Backoff

	xdsClientResourceUpdatesValidMetric = estats.RegisterInt64Count(estats.MetricDescriptor{
		Name:        "grpc.xds_client.resource_updates_valid",
		Description: "A counter of resources received that were considered valid. The counter will be incremented even for resources that have not changed.",
		Unit:        "{resource}",
		Labels:      []string{"grpc.target", "grpc.xds.server", "grpc.xds.resource_type"},
		Default:     false,
	})
	xdsClientResourceUpdatesInvalidMetric = estats.RegisterInt64Count(estats.MetricDescriptor{
		Name:        "grpc.xds_client.resource_updates_invalid",
		Description: "A counter of resources received that were considered invalid.",
		Unit:        "{resource}",
		Labels:      []string{"grpc.target", "grpc.xds.server", "grpc.xds.resource_type"},
		Default:     false,
	})
	xdsClientServerFailureMetric = estats.RegisterInt64Count(estats.MetricDescriptor{
		Name:        "grpc.xds_client.server_failure",
		Description: "A counter of xDS servers going from healthy to unhealthy. A server goes unhealthy when we have a connectivity failure or when the ADS stream fails without seeing a response message, as per gRFC A57.",
		Unit:        "{failure}",
		Labels:      []string{"grpc.target", "grpc.xds.server"},
		Default:     false,
	})
)

// clientImpl embed xdsclient.XDSClient and implement internal XDSClient
// interface with ref counting so that it can be shared by the xds resolver and
// balancer implementations, across multiple ClientConns and Servers.
type clientImpl struct {
	*xdsclient.XDSClient // TODO: #8313 - get rid of embedding, if possible.

	// The following fields are initialized at creation time and are read-only
	// after that.
	xdsClientConfig xdsclient.Config
	bootstrapConfig *bootstrap.Config
	logger          *grpclog.PrefixLogger
	target          string
	lrsClient       *lrsclient.LRSClient

	// Accessed atomically
	refCount int32
}

// metricsReporter implements the clients.MetricsReporter interface and uses an
// underlying stats.MetricsRecorderList to record metrics.
type metricsReporter struct {
	recorder estats.MetricsRecorder
	target   string
}

// ReportMetric implements the clients.MetricsReporter interface.
// It receives metric data, determines the appropriate metric based on the type
// of the data, and records it using the embedded MetricsRecorderList.
func (mr *metricsReporter) ReportMetric(metric any) {
	if mr.recorder == nil {
		return
	}

	switch m := metric.(type) {
	case *metrics.ResourceUpdateValid:
		xdsClientResourceUpdatesValidMetric.Record(mr.recorder, 1, mr.target, m.ServerURI, m.ResourceType)
	case *metrics.ResourceUpdateInvalid:
		xdsClientResourceUpdatesInvalidMetric.Record(mr.recorder, 1, mr.target, m.ServerURI, m.ResourceType)
	case *metrics.ServerFailure:
		xdsClientServerFailureMetric.Record(mr.recorder, 1, mr.target, m.ServerURI)
	}
}

func newClientImpl(config *bootstrap.Config, metricsRecorder estats.MetricsRecorder, target string, watchExpiryTimeout time.Duration) (*clientImpl, error) {
	gConfig, err := buildXDSClientConfig(config, metricsRecorder, target, watchExpiryTimeout)
	if err != nil {
		return nil, err
	}
	client, err := xdsclient.New(gConfig)
	if err != nil {
		return nil, err
	}
	lrsC, err := lrsclient.New(lrsclient.Config{
		Node:             gConfig.Node,
		TransportBuilder: gConfig.TransportBuilder,
	})
	if err != nil {
		return nil, err
	}
	c := &clientImpl{
		XDSClient:       client,
		xdsClientConfig: gConfig,
		bootstrapConfig: config,
		target:          target,
		refCount:        1,
		lrsClient:       lrsC,
	}
	c.logger = prefixLogger(c)
	return c, nil
}

// BootstrapConfig returns the configuration read from the bootstrap file.
// Callers must treat the return value as read-only.
func (c *clientImpl) BootstrapConfig() *bootstrap.Config {
	return c.bootstrapConfig
}

func (c *clientImpl) incrRef() int32 {
	return atomic.AddInt32(&c.refCount, 1)
}

func (c *clientImpl) decrRef() int32 {
	return atomic.AddInt32(&c.refCount, -1)
}

// buildXDSClientConfig builds the xdsclient.Config from the bootstrap.Config.
func buildXDSClientConfig(config *bootstrap.Config, metricsRecorder estats.MetricsRecorder, target string, watchExpiryTimeout time.Duration) (xdsclient.Config, error) {
	grpcTransportConfigs := make(map[string]grpctransport.Config)
	gServerCfgMap := make(map[xdsclient.ServerConfig]*bootstrap.ServerConfig)

	gAuthorities := make(map[string]xdsclient.Authority)
	for name, cfg := range config.Authorities() {
		// If server configs are specified in the authorities map, use that.
		// Else, use the top-level server configs.
		serverCfg := config.XDSServers()
		if len(cfg.XDSServers) >= 1 {
			serverCfg = cfg.XDSServers
		}
		var gServerCfg []xdsclient.ServerConfig
		for _, sc := range serverCfg {
			if err := populateGRPCTransportConfigsFromServerConfig(sc, grpcTransportConfigs); err != nil {
				return xdsclient.Config{}, err
			}
			gsc := xdsclient.ServerConfig{
				ServerIdentifier:       clients.ServerIdentifier{ServerURI: sc.ServerURI(), Extensions: grpctransport.ServerIdentifierExtension{ConfigName: sc.SelectedCreds().Type}},
				IgnoreResourceDeletion: sc.ServerFeaturesIgnoreResourceDeletion()}
			gServerCfg = append(gServerCfg, gsc)
			gServerCfgMap[gsc] = sc
		}
		gAuthorities[name] = xdsclient.Authority{XDSServers: gServerCfg}
	}

	gServerCfgs := make([]xdsclient.ServerConfig, 0, len(config.XDSServers()))
	for _, sc := range config.XDSServers() {
		if err := populateGRPCTransportConfigsFromServerConfig(sc, grpcTransportConfigs); err != nil {
			return xdsclient.Config{}, err
		}
		gsc := xdsclient.ServerConfig{
			ServerIdentifier:       clients.ServerIdentifier{ServerURI: sc.ServerURI(), Extensions: grpctransport.ServerIdentifierExtension{ConfigName: sc.SelectedCreds().Type}},
			IgnoreResourceDeletion: sc.ServerFeaturesIgnoreResourceDeletion()}
		gServerCfgs = append(gServerCfgs, gsc)
		gServerCfgMap[gsc] = sc
	}

	node := config.Node()
	gNode := clients.Node{
		ID:               node.GetId(),
		Cluster:          node.GetCluster(),
		Metadata:         node.Metadata,
		UserAgentName:    node.UserAgentName,
		UserAgentVersion: node.GetUserAgentVersion(),
	}
	if node.Locality != nil {
		gNode.Locality = clients.Locality{
			Region:  node.Locality.Region,
			Zone:    node.Locality.Zone,
			SubZone: node.Locality.SubZone,
		}
	}

	return xdsclient.Config{
		Authorities:        gAuthorities,
		Servers:            gServerCfgs,
		Node:               gNode,
		TransportBuilder:   grpctransport.NewBuilder(grpcTransportConfigs),
		ResourceTypes:      supportedResourceTypes(config, gServerCfgMap),
		MetricsReporter:    &metricsReporter{recorder: metricsRecorder, target: target},
		WatchExpiryTimeout: watchExpiryTimeout,
	}, nil
}

// populateGRPCTransportConfigsFromServerConfig iterates through the channel
// credentials of the provided server configuration, builds credential bundles,
// and populates the grpctransport.Config map.
func populateGRPCTransportConfigsFromServerConfig(sc *bootstrap.ServerConfig, grpcTransportConfigs map[string]grpctransport.Config) error {
	for _, cc := range sc.ChannelCreds() {
		c := xdsbootstrap.GetCredentials(cc.Type)
		if c == nil {
			continue
		}
		bundle, _, err := c.Build(cc.Config)
		if err != nil {
			return fmt.Errorf("xds: failed to build credentials bundle from bootstrap for %q: %v", cc.Type, err)
		}
		grpcTransportConfigs[cc.Type] = grpctransport.Config{
			Credentials: bundle,
			GRPCNewClient: func(target string, opts ...grpc.DialOption) (*grpc.ClientConn, error) {
				opts = append(opts, sc.DialOptions()...)
				return grpc.NewClient(target, opts...)
			},
		}
	}
	return nil
}
