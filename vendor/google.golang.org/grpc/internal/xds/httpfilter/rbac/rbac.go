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

// Package rbac implements the Envoy RBAC HTTP filter.
package rbac

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"google.golang.org/grpc/internal/resolver"
	"google.golang.org/grpc/internal/xds/httpfilter"
	"google.golang.org/grpc/internal/xds/rbac"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"

	v3rbacpb "github.com/envoyproxy/go-control-plane/envoy/config/rbac/v3"
	rpb "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/rbac/v3"
)

func init() {
	httpfilter.Register(builder{})
}

type builder struct {
}

type config struct {
	httpfilter.FilterConfig
	chainEngine *rbac.ChainEngine
}

func (builder) TypeURLs() []string {
	return []string{
		"type.googleapis.com/envoy.extensions.filters.http.rbac.v3.RBAC",
		"type.googleapis.com/envoy.extensions.filters.http.rbac.v3.RBACPerRoute",
	}
}

// Parsing is the same for the base config and the override config.
func parseConfig(rbacCfg *rpb.RBAC) (httpfilter.FilterConfig, error) {
	// All the validation logic described in A41.
	for _, policy := range rbacCfg.GetRules().GetPolicies() {
		// "Policy.condition and Policy.checked_condition must cause a
		// validation failure if present." - A41
		if policy.Condition != nil {
			return nil, errors.New("rbac: Policy.condition is present")
		}
		if policy.CheckedCondition != nil {
			return nil, errors.New("rbac: policy.CheckedCondition is present")
		}

		// "It is also a validation failure if Permission or Principal has a
		// header matcher for a grpc- prefixed header name or :scheme." - A41
		for _, principal := range policy.Principals {
			name := principal.GetHeader().GetName()
			if name == ":scheme" || strings.HasPrefix(name, "grpc-") {
				return nil, fmt.Errorf("rbac: principal header matcher for %v is :scheme or starts with grpc", name)
			}
		}
		for _, permission := range policy.Permissions {
			name := permission.GetHeader().GetName()
			if name == ":scheme" || strings.HasPrefix(name, "grpc-") {
				return nil, fmt.Errorf("rbac: permission header matcher for %v is :scheme or starts with grpc", name)
			}
		}
	}

	// "Envoy aliases :authority and Host in its header map implementation, so
	// they should be treated equivalent for the RBAC matchers; there must be no
	// behavior change depending on which of the two header names is used in the
	// RBAC policy." - A41. Loop through config's principals and policies, change
	// any header matcher with value "host" to :authority", as that is what
	// grpc-go shifts both headers to in transport layer.
	for _, policy := range rbacCfg.GetRules().GetPolicies() {
		for _, principal := range policy.Principals {
			if principal.GetHeader().GetName() == "host" {
				principal.GetHeader().Name = ":authority"
			}
		}
		for _, permission := range policy.Permissions {
			if permission.GetHeader().GetName() == "host" {
				permission.GetHeader().Name = ":authority"
			}
		}
	}

	// Two cases where this HTTP Filter is a no op:
	// "If absent, no enforcing RBAC policy will be applied" - RBAC
	// Documentation for Rules field.
	// "At this time, if the RBAC.action is Action.LOG then the policy will be
	// completely ignored, as if RBAC was not configured." - A41
	if rbacCfg.Rules == nil || rbacCfg.GetRules().GetAction() == v3rbacpb.RBAC_LOG {
		return config{}, nil
	}

	// TODO(gregorycooke) - change the call chain to here so we have the filter
	// name to input here instead of an empty string. It will come from here:
	// https://github.com/grpc/grpc-go/blob/eff0942e95d93112921414aee758e619ec86f26f/xds/internal/xdsclient/xdsresource/unmarshal_lds.go#L199
	ce, err := rbac.NewChainEngine([]*v3rbacpb.RBAC{rbacCfg.GetRules()}, "")
	if err != nil {
		// "At this time, if the RBAC.action is Action.LOG then the policy will be
		// completely ignored, as if RBAC was not configured." - A41
		if rbacCfg.GetRules().GetAction() != v3rbacpb.RBAC_LOG {
			return nil, fmt.Errorf("rbac: error constructing matching engine: %v", err)
		}
	}

	return config{chainEngine: ce}, nil
}

func (builder) ParseFilterConfig(cfg proto.Message) (httpfilter.FilterConfig, error) {
	if cfg == nil {
		return nil, fmt.Errorf("rbac: nil configuration message provided")
	}
	m, ok := cfg.(*anypb.Any)
	if !ok {
		return nil, fmt.Errorf("rbac: error parsing config %v: unknown type %T", cfg, cfg)
	}
	msg := new(rpb.RBAC)
	if err := m.UnmarshalTo(msg); err != nil {
		return nil, fmt.Errorf("rbac: error parsing config %v: %v", cfg, err)
	}
	return parseConfig(msg)
}

func (builder) ParseFilterConfigOverride(override proto.Message) (httpfilter.FilterConfig, error) {
	if override == nil {
		return nil, fmt.Errorf("rbac: nil configuration message provided")
	}
	m, ok := override.(*anypb.Any)
	if !ok {
		return nil, fmt.Errorf("rbac: error parsing override config %v: unknown type %T", override, override)
	}
	msg := new(rpb.RBACPerRoute)
	if err := m.UnmarshalTo(msg); err != nil {
		return nil, fmt.Errorf("rbac: error parsing override config %v: %v", override, err)
	}
	return parseConfig(msg.Rbac)
}

func (builder) IsTerminal() bool {
	return false
}

var _ httpfilter.ServerInterceptorBuilder = builder{}

// BuildServerInterceptor is an optional interface builder implements in order
// to signify it works server side.
func (builder) BuildServerInterceptor(cfg httpfilter.FilterConfig, override httpfilter.FilterConfig) (resolver.ServerInterceptor, error) {
	if cfg == nil {
		return nil, fmt.Errorf("rbac: nil config provided")
	}

	c, ok := cfg.(config)
	if !ok {
		return nil, fmt.Errorf("rbac: incorrect config type provided (%T): %v", cfg, cfg)
	}

	if override != nil {
		// override completely replaces the listener configuration; but we
		// still validate the listener config type.
		c, ok = override.(config)
		if !ok {
			return nil, fmt.Errorf("rbac: incorrect override config type provided (%T): %v", override, override)
		}
	}

	// RBAC HTTP Filter is a no op from one of these two cases:
	// "If absent, no enforcing RBAC policy will be applied" - RBAC
	// Documentation for Rules field.
	// "At this time, if the RBAC.action is Action.LOG then the policy will be
	// completely ignored, as if RBAC was not configured." - A41
	if c.chainEngine == nil {
		return nil, nil
	}
	return &interceptor{chainEngine: c.chainEngine}, nil
}

type interceptor struct {
	chainEngine *rbac.ChainEngine
}

func (i *interceptor) AllowRPC(ctx context.Context) error {
	return i.chainEngine.IsAuthorized(ctx)
}
