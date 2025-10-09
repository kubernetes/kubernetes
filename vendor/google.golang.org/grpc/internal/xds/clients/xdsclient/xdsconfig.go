/*
 *
 * Copyright 2025 gRPC authors.
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
	"time"

	"google.golang.org/grpc/internal/xds/clients"
)

// Config is used to configure an xDS client. After one has been passed to the
// xDS client's New function, no part of it may be modified. A Config may be
// reused; the xdsclient package will also not modify it.
type Config struct {
	// Servers specifies a list of xDS management servers to connect to. The
	// order of the servers in this list reflects the order of preference of
	// the data returned by those servers. The xDS client uses the first
	// available server from the list.
	//
	// See gRFC A71 for more details on fallback behavior when the primary
	// xDS server is unavailable.
	//
	// gRFC A71: https://github.com/grpc/proposal/blob/master/A71-xds-fallback.md
	Servers []ServerConfig

	// Authorities defines the configuration for each xDS authority.  Federated resources
	// will be fetched from the servers specified by the corresponding Authority.
	Authorities map[string]Authority

	// Node is the identity of the xDS client connecting to the xDS
	// management server.
	Node clients.Node

	// TransportBuilder is used to create connections to xDS management servers.
	TransportBuilder clients.TransportBuilder

	// ResourceTypes is a map from resource type URLs to resource type
	// implementations. Each resource type URL uniquely identifies a specific
	// kind of xDS resource, and the corresponding resource type implementation
	// provides logic for parsing, validating, and processing resources of that
	// type.
	//
	// For example: "type.googleapis.com/envoy.config.listener.v3.Listener"
	ResourceTypes map[string]ResourceType

	// MetricsReporter is used to report registered metrics. If unset, no
	// metrics will be reported.
	MetricsReporter clients.MetricsReporter

	// WatchExpiryTimeout is the duration after which a resource watch expires
	// if the requested resource is not received from the management server.
	// Most users will not need to set this. If zero, a default value of 15
	// seconds is used as specified here:
	// envoyproxy.io/docs/envoy/latest/api-docs/xds_protocol#knowing-when-a-requested-resource-does-not-exist
	WatchExpiryTimeout time.Duration
}

// ServerConfig contains configuration for an xDS management server.
type ServerConfig struct {
	ServerIdentifier clients.ServerIdentifier

	// IgnoreResourceDeletion is a server feature which if set to true,
	// indicates that resource deletion errors from xDS management servers can
	// be ignored and cached resource data can be used.
	//
	// This will be removed in the future once we implement gRFC A88
	// and two new fields FailOnDataErrors and
	// ResourceTimerIsTransientError will be introduced.
	IgnoreResourceDeletion bool

	// TODO: Link to gRFC A88
}

// Authority contains configuration for an xDS control plane authority.
//
// See: https://www.envoyproxy.io/docs/envoy/latest/xds/core/v3/resource_locator.proto#xds-core-v3-resourcelocator
type Authority struct {
	// XDSServers contains the list of server configurations for this authority.
	//
	// See Config.Servers for more details.
	XDSServers []ServerConfig
}

func isServerConfigEqual(a, b *ServerConfig) bool {
	return a.ServerIdentifier == b.ServerIdentifier && a.IgnoreResourceDeletion == b.IgnoreResourceDeletion
}
