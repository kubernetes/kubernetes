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
	"errors"
	"fmt"
	"strconv"

	v1xdsudpatypepb "github.com/cncf/xds/go/udpa/type/v1"
	v3xdsxdstypepb "github.com/cncf/xds/go/xds/type/v3"
	v3listenerpb "github.com/envoyproxy/go-control-plane/envoy/config/listener/v3"
	v3routepb "github.com/envoyproxy/go-control-plane/envoy/config/route/v3"
	v3httppb "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/network/http_connection_manager/v3"
	"google.golang.org/grpc/internal/xds/httpfilter"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

func unmarshalListenerResource(r *anypb.Any) (string, ListenerUpdate, error) {
	r, err := UnwrapResource(r)
	if err != nil {
		return "", ListenerUpdate{}, fmt.Errorf("failed to unwrap resource: %v", err)
	}

	if !IsListenerResource(r.GetTypeUrl()) {
		return "", ListenerUpdate{}, fmt.Errorf("unexpected listener resource type: %q ", r.GetTypeUrl())
	}
	lis := &v3listenerpb.Listener{}
	if err := proto.Unmarshal(r.GetValue(), lis); err != nil {
		return "", ListenerUpdate{}, fmt.Errorf("failed to unmarshal resource: %v", err)
	}

	lu, err := processListener(lis)
	if err != nil {
		return lis.GetName(), ListenerUpdate{}, err
	}
	lu.Raw = r
	return lis.GetName(), *lu, nil
}

func processListener(lis *v3listenerpb.Listener) (*ListenerUpdate, error) {
	if lis.GetApiListener() != nil {
		return processClientSideListener(lis)
	}
	return processServerSideListener(lis)
}

// processClientSideListener checks if the provided Listener proto meets
// the expected criteria. If so, it returns a non-empty routeConfigName.
func processClientSideListener(lis *v3listenerpb.Listener) (*ListenerUpdate, error) {
	update := &ListenerUpdate{}

	apiLisAny := lis.GetApiListener().GetApiListener()
	if !IsHTTPConnManagerResource(apiLisAny.GetTypeUrl()) {
		return nil, fmt.Errorf("unexpected http connection manager resource type: %q", apiLisAny.GetTypeUrl())
	}
	apiLis := &v3httppb.HttpConnectionManager{}
	if err := proto.Unmarshal(apiLisAny.GetValue(), apiLis); err != nil {
		return nil, fmt.Errorf("failed to unmarshal api_listener: %v", err)
	}
	// "HttpConnectionManager.xff_num_trusted_hops must be unset or zero and
	// HttpConnectionManager.original_ip_detection_extensions must be empty. If
	// either field has an incorrect value, the Listener must be NACKed." - A41
	if apiLis.XffNumTrustedHops != 0 {
		return nil, fmt.Errorf("xff_num_trusted_hops must be unset or zero %+v", apiLis)
	}
	if len(apiLis.OriginalIpDetectionExtensions) != 0 {
		return nil, fmt.Errorf("original_ip_detection_extensions must be empty %+v", apiLis)
	}

	switch apiLis.RouteSpecifier.(type) {
	case *v3httppb.HttpConnectionManager_Rds:
		if configsource := apiLis.GetRds().GetConfigSource(); configsource.GetAds() == nil && configsource.GetSelf() == nil {
			return nil, fmt.Errorf("LDS's RDS configSource is not ADS or Self: %+v", lis)
		}
		name := apiLis.GetRds().GetRouteConfigName()
		if name == "" {
			return nil, fmt.Errorf("empty route_config_name: %+v", lis)
		}
		update.RouteConfigName = name
	case *v3httppb.HttpConnectionManager_RouteConfig:
		routeU, err := generateRDSUpdateFromRouteConfiguration(apiLis.GetRouteConfig())
		if err != nil {
			return nil, fmt.Errorf("failed to parse inline RDS resp: %v", err)
		}
		update.InlineRouteConfig = &routeU
	case nil:
		return nil, fmt.Errorf("no RouteSpecifier: %+v", apiLis)
	default:
		return nil, fmt.Errorf("unsupported type %T for RouteSpecifier", apiLis.RouteSpecifier)
	}

	// The following checks and fields only apply to xDS protocol versions v3+.

	update.MaxStreamDuration = apiLis.GetCommonHttpProtocolOptions().GetMaxStreamDuration().AsDuration()

	var err error
	if update.HTTPFilters, err = processHTTPFilters(apiLis.GetHttpFilters(), false); err != nil {
		return nil, err
	}

	return update, nil
}

func unwrapHTTPFilterConfig(config *anypb.Any) (proto.Message, string, error) {
	switch {
	case config.MessageIs(&v3xdsxdstypepb.TypedStruct{}):
		// The real type name is inside the new TypedStruct message.
		s := new(v3xdsxdstypepb.TypedStruct)
		if err := config.UnmarshalTo(s); err != nil {
			return nil, "", fmt.Errorf("error unmarshalling TypedStruct filter config: %v", err)
		}
		return s, s.GetTypeUrl(), nil
	case config.MessageIs(&v1xdsudpatypepb.TypedStruct{}):
		// The real type name is inside the old TypedStruct message.
		s := new(v1xdsudpatypepb.TypedStruct)
		if err := config.UnmarshalTo(s); err != nil {
			return nil, "", fmt.Errorf("error unmarshalling TypedStruct filter config: %v", err)
		}
		return s, s.GetTypeUrl(), nil
	default:
		return config, config.GetTypeUrl(), nil
	}
}

func validateHTTPFilterConfig(cfg *anypb.Any, lds, optional bool) (httpfilter.Filter, httpfilter.FilterConfig, error) {
	config, typeURL, err := unwrapHTTPFilterConfig(cfg)
	if err != nil {
		return nil, nil, err
	}
	filterBuilder := httpfilter.Get(typeURL)
	if filterBuilder == nil {
		if optional {
			return nil, nil, nil
		}
		return nil, nil, fmt.Errorf("no filter implementation found for %q", typeURL)
	}
	parseFunc := filterBuilder.ParseFilterConfig
	if !lds {
		parseFunc = filterBuilder.ParseFilterConfigOverride
	}
	filterConfig, err := parseFunc(config)
	if err != nil {
		return nil, nil, fmt.Errorf("error parsing config for filter %q: %v", typeURL, err)
	}
	return filterBuilder, filterConfig, nil
}

func processHTTPFilterOverrides(cfgs map[string]*anypb.Any) (map[string]httpfilter.FilterConfig, error) {
	if len(cfgs) == 0 {
		return nil, nil
	}
	m := make(map[string]httpfilter.FilterConfig)
	for name, cfg := range cfgs {
		optional := false
		s := new(v3routepb.FilterConfig)
		if cfg.MessageIs(s) {
			if err := cfg.UnmarshalTo(s); err != nil {
				return nil, fmt.Errorf("filter override %q: error unmarshalling FilterConfig: %v", name, err)
			}
			cfg = s.GetConfig()
			optional = s.GetIsOptional()
		}

		httpFilter, config, err := validateHTTPFilterConfig(cfg, false, optional)
		if err != nil {
			return nil, fmt.Errorf("filter override %q: %v", name, err)
		}
		if httpFilter == nil {
			// Optional configs are ignored.
			continue
		}
		m[name] = config
	}
	return m, nil
}

func processHTTPFilters(filters []*v3httppb.HttpFilter, server bool) ([]HTTPFilter, error) {
	ret := make([]HTTPFilter, 0, len(filters))
	seenNames := make(map[string]bool, len(filters))
	for _, filter := range filters {
		name := filter.GetName()
		if name == "" {
			return nil, errors.New("filter missing name field")
		}
		if seenNames[name] {
			return nil, fmt.Errorf("duplicate filter name %q", name)
		}
		seenNames[name] = true

		httpFilter, config, err := validateHTTPFilterConfig(filter.GetTypedConfig(), true, filter.GetIsOptional())
		if err != nil {
			return nil, err
		}
		if httpFilter == nil {
			// Optional configs are ignored.
			continue
		}
		if server {
			if _, ok := httpFilter.(httpfilter.ServerInterceptorBuilder); !ok {
				if filter.GetIsOptional() {
					continue
				}
				return nil, fmt.Errorf("HTTP filter %q not supported server-side", name)
			}
		} else if _, ok := httpFilter.(httpfilter.ClientInterceptorBuilder); !ok {
			if filter.GetIsOptional() {
				continue
			}
			return nil, fmt.Errorf("HTTP filter %q not supported client-side", name)
		}

		// Save name/config
		ret = append(ret, HTTPFilter{Name: name, Filter: httpFilter, Config: config})
	}
	// "Validation will fail if a terminal filter is not the last filter in the
	// chain or if a non-terminal filter is the last filter in the chain." - A39
	if len(ret) == 0 {
		return nil, fmt.Errorf("http filters list is empty")
	}
	var i int
	for ; i < len(ret)-1; i++ {
		if ret[i].Filter.IsTerminal() {
			return nil, fmt.Errorf("http filter %q is a terminal filter but it is not last in the filter chain", ret[i].Name)
		}
	}
	if !ret[i].Filter.IsTerminal() {
		return nil, fmt.Errorf("http filter %q is not a terminal filter", ret[len(ret)-1].Name)
	}
	return ret, nil
}

func processServerSideListener(lis *v3listenerpb.Listener) (*ListenerUpdate, error) {
	if n := len(lis.ListenerFilters); n != 0 {
		return nil, fmt.Errorf("unsupported field 'listener_filters' contains %d entries", n)
	}
	if lis.GetUseOriginalDst().GetValue() {
		return nil, errors.New("unsupported field 'use_original_dst' is present and set to true")
	}
	addr := lis.GetAddress()
	if addr == nil {
		return nil, fmt.Errorf("no address field in LDS response: %+v", lis)
	}
	sockAddr := addr.GetSocketAddress()
	if sockAddr == nil {
		return nil, fmt.Errorf("no socket_address field in LDS response: %+v", lis)
	}
	lu := &ListenerUpdate{
		InboundListenerCfg: &InboundListenerConfig{
			Address: sockAddr.GetAddress(),
			Port:    strconv.Itoa(int(sockAddr.GetPortValue())),
		},
	}

	fcMgr, err := NewFilterChainManager(lis)
	if err != nil {
		return nil, err
	}
	lu.InboundListenerCfg.FilterChains = fcMgr
	return lu, nil
}
