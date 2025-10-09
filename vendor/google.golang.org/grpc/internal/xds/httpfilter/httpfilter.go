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
 *
 */

// Package httpfilter contains the HTTPFilter interface and a registry for
// storing and retrieving their implementations.
package httpfilter

import (
	iresolver "google.golang.org/grpc/internal/resolver"
	"google.golang.org/protobuf/proto"
)

// FilterConfig represents an opaque data structure holding configuration for a
// filter.  Embed this interface to implement it.
type FilterConfig interface {
	isFilterConfig()
}

// Filter defines the parsing functionality of an HTTP filter.  A Filter may
// optionally implement either ClientInterceptorBuilder or
// ServerInterceptorBuilder or both, indicating it is capable of working on the
// client side or server side or both, respectively.
type Filter interface {
	// TypeURLs are the proto message types supported by this filter.  A filter
	// will be registered by each of its supported message types.
	TypeURLs() []string
	// ParseFilterConfig parses the provided configuration proto.Message from
	// the LDS configuration of this filter.  This may be an anypb.Any, a
	// udpa.type.v1.TypedStruct, or an xds.type.v3.TypedStruct for filters that
	// do not accept a custom type. The resulting FilterConfig will later be
	// passed to Build.
	ParseFilterConfig(proto.Message) (FilterConfig, error)
	// ParseFilterConfigOverride parses the provided override configuration
	// proto.Message from the RDS override configuration of this filter.  This
	// may be an anypb.Any, a udpa.type.v1.TypedStruct, or an
	// xds.type.v3.TypedStruct for filters that do not accept a custom type.
	// The resulting FilterConfig will later be passed to Build.
	ParseFilterConfigOverride(proto.Message) (FilterConfig, error)
	// IsTerminal returns whether this Filter is terminal or not (i.e. it must
	// be last filter in the filter chain).
	IsTerminal() bool
}

// ClientInterceptorBuilder constructs a Client Interceptor.  If this type is
// implemented by a Filter, it is capable of working on a client.
type ClientInterceptorBuilder interface {
	// BuildClientInterceptor uses the FilterConfigs produced above to produce
	// an HTTP filter interceptor for clients.  config will always be non-nil,
	// but override may be nil if no override config exists for the filter.  It
	// is valid for Build to return a nil Interceptor and a nil error.  In this
	// case, the RPC will not be intercepted by this filter.
	BuildClientInterceptor(config, override FilterConfig) (iresolver.ClientInterceptor, error)
}

// ServerInterceptorBuilder constructs a Server Interceptor.  If this type is
// implemented by a Filter, it is capable of working on a server.
type ServerInterceptorBuilder interface {
	// BuildServerInterceptor uses the FilterConfigs produced above to produce
	// an HTTP filter interceptor for servers.  config will always be non-nil,
	// but override may be nil if no override config exists for the filter.  It
	// is valid for Build to return a nil Interceptor and a nil error.  In this
	// case, the RPC will not be intercepted by this filter.
	BuildServerInterceptor(config, override FilterConfig) (iresolver.ServerInterceptor, error)
}

var (
	// m is a map from scheme to filter.
	m = make(map[string]Filter)
)

// Register registers the HTTP filter Builder to the filter map. b.TypeURLs()
// will be used as the types for this filter.
//
// NOTE: this function must only be called during initialization time (i.e. in
// an init() function), and is not thread-safe. If multiple filters are
// registered with the same type URL, the one registered last will take effect.
func Register(b Filter) {
	for _, u := range b.TypeURLs() {
		m[u] = b
	}
}

// UnregisterForTesting unregisters the HTTP Filter for testing purposes.
func UnregisterForTesting(typeURL string) {
	delete(m, typeURL)
}

// Get returns the HTTPFilter registered with typeURL.
//
// If no filter is register with typeURL, nil will be returned.
func Get(typeURL string) Filter {
	return m[typeURL]
}
