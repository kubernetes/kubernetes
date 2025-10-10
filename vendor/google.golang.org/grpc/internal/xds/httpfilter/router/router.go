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

// Package router implements the Envoy Router HTTP filter.
package router

import (
	"fmt"

	iresolver "google.golang.org/grpc/internal/resolver"
	"google.golang.org/grpc/internal/xds/httpfilter"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"

	pb "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/router/v3"
)

// TypeURL is the message type for the Router configuration.
const TypeURL = "type.googleapis.com/envoy.extensions.filters.http.router.v3.Router"

func init() {
	httpfilter.Register(builder{})
}

// IsRouterFilter returns true iff a HTTP filter is a Router filter.
func IsRouterFilter(b httpfilter.Filter) bool {
	_, ok := b.(builder)
	return ok
}

type builder struct {
}

func (builder) TypeURLs() []string { return []string{TypeURL} }

func (builder) ParseFilterConfig(cfg proto.Message) (httpfilter.FilterConfig, error) {
	// The gRPC router filter does not currently use any fields from the
	// config.  Verify type only.
	if cfg == nil {
		return nil, fmt.Errorf("router: nil configuration message provided")
	}
	m, ok := cfg.(*anypb.Any)
	if !ok {
		return nil, fmt.Errorf("router: error parsing config %v: unknown type %T", cfg, cfg)
	}
	msg := new(pb.Router)
	if err := m.UnmarshalTo(msg); err != nil {
		return nil, fmt.Errorf("router: error parsing config %v: %v", cfg, err)
	}
	return config{}, nil
}

func (builder) ParseFilterConfigOverride(override proto.Message) (httpfilter.FilterConfig, error) {
	if override != nil {
		return nil, fmt.Errorf("router: unexpected config override specified: %v", override)
	}
	return config{}, nil
}

func (builder) IsTerminal() bool {
	return true
}

var (
	_ httpfilter.ClientInterceptorBuilder = builder{}
	_ httpfilter.ServerInterceptorBuilder = builder{}
)

func (builder) BuildClientInterceptor(cfg, override httpfilter.FilterConfig) (iresolver.ClientInterceptor, error) {
	if _, ok := cfg.(config); !ok {
		return nil, fmt.Errorf("router: incorrect config type provided (%T): %v", cfg, cfg)
	}
	if override != nil {
		return nil, fmt.Errorf("router: unexpected override configuration specified: %v", override)
	}
	// The gRPC router is implemented within the xds resolver's config
	// selector, not as a separate plugin.  So we return a nil HTTPFilter,
	// which will not be invoked.
	return nil, nil
}

func (builder) BuildServerInterceptor(cfg, override httpfilter.FilterConfig) (iresolver.ServerInterceptor, error) {
	if _, ok := cfg.(config); !ok {
		return nil, fmt.Errorf("router: incorrect config type provided (%T): %v", cfg, cfg)
	}
	if override != nil {
		return nil, fmt.Errorf("router: unexpected override configuration specified: %v", override)
	}
	// The gRPC router is currently unimplemented on the server side. So we
	// return a nil HTTPFilter, which will not be invoked.
	return nil, nil
}

// The gRPC router filter does not currently support any configuration.  Verify
// type only.
type config struct {
	httpfilter.FilterConfig
}
