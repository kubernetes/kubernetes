/*
 *
 * Copyright 2021 gRPC authors.
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
 */

package xdsresource

import (
	"time"

	"google.golang.org/grpc/internal/xds/httpfilter"
	"google.golang.org/protobuf/types/known/anypb"
)

// ListenerUpdate contains information received in an LDS response, which is of
// interest to the registered LDS watcher.
type ListenerUpdate struct {
	// RouteConfigName is the route configuration name corresponding to the
	// target which is being watched through LDS.
	//
	// Exactly one of RouteConfigName and InlineRouteConfig is set.
	RouteConfigName string
	// InlineRouteConfig is the inline route configuration (RDS response)
	// returned inside LDS.
	//
	// Exactly one of RouteConfigName and InlineRouteConfig is set.
	InlineRouteConfig *RouteConfigUpdate

	// MaxStreamDuration contains the HTTP connection manager's
	// common_http_protocol_options.max_stream_duration field, or zero if
	// unset.
	MaxStreamDuration time.Duration
	// HTTPFilters is a list of HTTP filters (name, config) from the LDS
	// response.
	HTTPFilters []HTTPFilter
	// InboundListenerCfg contains inbound listener configuration.
	InboundListenerCfg *InboundListenerConfig

	// Raw is the resource from the xds response.
	Raw *anypb.Any
}

// HTTPFilter represents one HTTP filter from an LDS response's HTTP connection
// manager field.
type HTTPFilter struct {
	// Name is an arbitrary name of the filter.  Used for applying override
	// settings in virtual host / route / weighted cluster configuration (not
	// yet supported).
	Name string
	// Filter is the HTTP filter found in the registry for the config type.
	Filter httpfilter.Filter
	// Config contains the filter's configuration
	Config httpfilter.FilterConfig
}

// InboundListenerConfig contains information about the inbound listener, i.e
// the server-side listener.
type InboundListenerConfig struct {
	// Address is the local address on which the inbound listener is expected to
	// accept incoming connections.
	Address string
	// Port is the local port on which the inbound listener is expected to
	// accept incoming connections.
	Port string
	// FilterChains is the list of filter chains associated with this listener.
	FilterChains *FilterChainManager
}
