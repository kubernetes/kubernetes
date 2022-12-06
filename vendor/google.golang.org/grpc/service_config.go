/*
 *
 * Copyright 2017 gRPC authors.
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

package grpc

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/internal"
	internalserviceconfig "google.golang.org/grpc/internal/serviceconfig"
	"google.golang.org/grpc/serviceconfig"
)

const maxInt = int(^uint(0) >> 1)

// MethodConfig defines the configuration recommended by the service providers for a
// particular method.
//
// Deprecated: Users should not use this struct. Service config should be received
// through name resolver, as specified here
// https://github.com/grpc/grpc/blob/master/doc/service_config.md
type MethodConfig = internalserviceconfig.MethodConfig

type lbConfig struct {
	name string
	cfg  serviceconfig.LoadBalancingConfig
}

// ServiceConfig is provided by the service provider and contains parameters for how
// clients that connect to the service should behave.
//
// Deprecated: Users should not use this struct. Service config should be received
// through name resolver, as specified here
// https://github.com/grpc/grpc/blob/master/doc/service_config.md
type ServiceConfig struct {
	serviceconfig.Config

	// LB is the load balancer the service providers recommends.  This is
	// deprecated; lbConfigs is preferred.  If lbConfig and LB are both present,
	// lbConfig will be used.
	LB *string

	// lbConfig is the service config's load balancing configuration.  If
	// lbConfig and LB are both present, lbConfig will be used.
	lbConfig *lbConfig

	// Methods contains a map for the methods in this service.  If there is an
	// exact match for a method (i.e. /service/method) in the map, use the
	// corresponding MethodConfig.  If there's no exact match, look for the
	// default config for the service (/service/) and use the corresponding
	// MethodConfig if it exists.  Otherwise, the method has no MethodConfig to
	// use.
	Methods map[string]MethodConfig

	// If a retryThrottlingPolicy is provided, gRPC will automatically throttle
	// retry attempts and hedged RPCs when the client’s ratio of failures to
	// successes exceeds a threshold.
	//
	// For each server name, the gRPC client will maintain a token_count which is
	// initially set to maxTokens, and can take values between 0 and maxTokens.
	//
	// Every outgoing RPC (regardless of service or method invoked) will change
	// token_count as follows:
	//
	//   - Every failed RPC will decrement the token_count by 1.
	//   - Every successful RPC will increment the token_count by tokenRatio.
	//
	// If token_count is less than or equal to maxTokens / 2, then RPCs will not
	// be retried and hedged RPCs will not be sent.
	retryThrottling *retryThrottlingPolicy
	// healthCheckConfig must be set as one of the requirement to enable LB channel
	// health check.
	healthCheckConfig *healthCheckConfig
	// rawJSONString stores service config json string that get parsed into
	// this service config struct.
	rawJSONString string
}

// healthCheckConfig defines the go-native version of the LB channel health check config.
type healthCheckConfig struct {
	// serviceName is the service name to use in the health-checking request.
	ServiceName string
}

type jsonRetryPolicy struct {
	MaxAttempts          int
	InitialBackoff       string
	MaxBackoff           string
	BackoffMultiplier    float64
	RetryableStatusCodes []codes.Code
}

// retryThrottlingPolicy defines the go-native version of the retry throttling
// policy defined by the service config here:
// https://github.com/grpc/proposal/blob/master/A6-client-retries.md#integration-with-service-config
type retryThrottlingPolicy struct {
	// The number of tokens starts at maxTokens. The token_count will always be
	// between 0 and maxTokens.
	//
	// This field is required and must be greater than zero.
	MaxTokens float64
	// The amount of tokens to add on each successful RPC. Typically this will
	// be some number between 0 and 1, e.g., 0.1.
	//
	// This field is required and must be greater than zero. Up to 3 decimal
	// places are supported.
	TokenRatio float64
}

func parseDuration(s *string) (*time.Duration, error) {
	if s == nil {
		return nil, nil
	}
	if !strings.HasSuffix(*s, "s") {
		return nil, fmt.Errorf("malformed duration %q", *s)
	}
	ss := strings.SplitN((*s)[:len(*s)-1], ".", 3)
	if len(ss) > 2 {
		return nil, fmt.Errorf("malformed duration %q", *s)
	}
	// hasDigits is set if either the whole or fractional part of the number is
	// present, since both are optional but one is required.
	hasDigits := false
	var d time.Duration
	if len(ss[0]) > 0 {
		i, err := strconv.ParseInt(ss[0], 10, 32)
		if err != nil {
			return nil, fmt.Errorf("malformed duration %q: %v", *s, err)
		}
		d = time.Duration(i) * time.Second
		hasDigits = true
	}
	if len(ss) == 2 && len(ss[1]) > 0 {
		if len(ss[1]) > 9 {
			return nil, fmt.Errorf("malformed duration %q", *s)
		}
		f, err := strconv.ParseInt(ss[1], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("malformed duration %q: %v", *s, err)
		}
		for i := 9; i > len(ss[1]); i-- {
			f *= 10
		}
		d += time.Duration(f)
		hasDigits = true
	}
	if !hasDigits {
		return nil, fmt.Errorf("malformed duration %q", *s)
	}

	return &d, nil
}

type jsonName struct {
	Service string
	Method  string
}

var (
	errDuplicatedName             = errors.New("duplicated name")
	errEmptyServiceNonEmptyMethod = errors.New("cannot combine empty 'service' and non-empty 'method'")
)

func (j jsonName) generatePath() (string, error) {
	if j.Service == "" {
		if j.Method != "" {
			return "", errEmptyServiceNonEmptyMethod
		}
		return "", nil
	}
	res := "/" + j.Service + "/"
	if j.Method != "" {
		res += j.Method
	}
	return res, nil
}

// TODO(lyuxuan): delete this struct after cleaning up old service config implementation.
type jsonMC struct {
	Name                    *[]jsonName
	WaitForReady            *bool
	Timeout                 *string
	MaxRequestMessageBytes  *int64
	MaxResponseMessageBytes *int64
	RetryPolicy             *jsonRetryPolicy
}

// TODO(lyuxuan): delete this struct after cleaning up old service config implementation.
type jsonSC struct {
	LoadBalancingPolicy *string
	LoadBalancingConfig *internalserviceconfig.BalancerConfig
	MethodConfig        *[]jsonMC
	RetryThrottling     *retryThrottlingPolicy
	HealthCheckConfig   *healthCheckConfig
}

func init() {
	internal.ParseServiceConfig = parseServiceConfig
}
func parseServiceConfig(js string) *serviceconfig.ParseResult {
	if len(js) == 0 {
		return &serviceconfig.ParseResult{Err: fmt.Errorf("no JSON service config provided")}
	}
	var rsc jsonSC
	err := json.Unmarshal([]byte(js), &rsc)
	if err != nil {
		logger.Warningf("grpc: parseServiceConfig error unmarshaling %s due to %v", js, err)
		return &serviceconfig.ParseResult{Err: err}
	}
	sc := ServiceConfig{
		LB:                rsc.LoadBalancingPolicy,
		Methods:           make(map[string]MethodConfig),
		retryThrottling:   rsc.RetryThrottling,
		healthCheckConfig: rsc.HealthCheckConfig,
		rawJSONString:     js,
	}
	if c := rsc.LoadBalancingConfig; c != nil {
		sc.lbConfig = &lbConfig{
			name: c.Name,
			cfg:  c.Config,
		}
	}

	if rsc.MethodConfig == nil {
		return &serviceconfig.ParseResult{Config: &sc}
	}

	paths := map[string]struct{}{}
	for _, m := range *rsc.MethodConfig {
		if m.Name == nil {
			continue
		}
		d, err := parseDuration(m.Timeout)
		if err != nil {
			logger.Warningf("grpc: parseServiceConfig error unmarshaling %s due to %v", js, err)
			return &serviceconfig.ParseResult{Err: err}
		}

		mc := MethodConfig{
			WaitForReady: m.WaitForReady,
			Timeout:      d,
		}
		if mc.RetryPolicy, err = convertRetryPolicy(m.RetryPolicy); err != nil {
			logger.Warningf("grpc: parseServiceConfig error unmarshaling %s due to %v", js, err)
			return &serviceconfig.ParseResult{Err: err}
		}
		if m.MaxRequestMessageBytes != nil {
			if *m.MaxRequestMessageBytes > int64(maxInt) {
				mc.MaxReqSize = newInt(maxInt)
			} else {
				mc.MaxReqSize = newInt(int(*m.MaxRequestMessageBytes))
			}
		}
		if m.MaxResponseMessageBytes != nil {
			if *m.MaxResponseMessageBytes > int64(maxInt) {
				mc.MaxRespSize = newInt(maxInt)
			} else {
				mc.MaxRespSize = newInt(int(*m.MaxResponseMessageBytes))
			}
		}
		for i, n := range *m.Name {
			path, err := n.generatePath()
			if err != nil {
				logger.Warningf("grpc: parseServiceConfig error unmarshaling %s due to methodConfig[%d]: %v", js, i, err)
				return &serviceconfig.ParseResult{Err: err}
			}

			if _, ok := paths[path]; ok {
				err = errDuplicatedName
				logger.Warningf("grpc: parseServiceConfig error unmarshaling %s due to methodConfig[%d]: %v", js, i, err)
				return &serviceconfig.ParseResult{Err: err}
			}
			paths[path] = struct{}{}
			sc.Methods[path] = mc
		}
	}

	if sc.retryThrottling != nil {
		if mt := sc.retryThrottling.MaxTokens; mt <= 0 || mt > 1000 {
			return &serviceconfig.ParseResult{Err: fmt.Errorf("invalid retry throttling config: maxTokens (%v) out of range (0, 1000]", mt)}
		}
		if tr := sc.retryThrottling.TokenRatio; tr <= 0 {
			return &serviceconfig.ParseResult{Err: fmt.Errorf("invalid retry throttling config: tokenRatio (%v) may not be negative", tr)}
		}
	}
	return &serviceconfig.ParseResult{Config: &sc}
}

func convertRetryPolicy(jrp *jsonRetryPolicy) (p *internalserviceconfig.RetryPolicy, err error) {
	if jrp == nil {
		return nil, nil
	}
	ib, err := parseDuration(&jrp.InitialBackoff)
	if err != nil {
		return nil, err
	}
	mb, err := parseDuration(&jrp.MaxBackoff)
	if err != nil {
		return nil, err
	}

	if jrp.MaxAttempts <= 1 ||
		*ib <= 0 ||
		*mb <= 0 ||
		jrp.BackoffMultiplier <= 0 ||
		len(jrp.RetryableStatusCodes) == 0 {
		logger.Warningf("grpc: ignoring retry policy %v due to illegal configuration", jrp)
		return nil, nil
	}

	rp := &internalserviceconfig.RetryPolicy{
		MaxAttempts:          jrp.MaxAttempts,
		InitialBackoff:       *ib,
		MaxBackoff:           *mb,
		BackoffMultiplier:    jrp.BackoffMultiplier,
		RetryableStatusCodes: make(map[codes.Code]bool),
	}
	if rp.MaxAttempts > 5 {
		// TODO(retry): Make the max maxAttempts configurable.
		rp.MaxAttempts = 5
	}
	for _, code := range jrp.RetryableStatusCodes {
		rp.RetryableStatusCodes[code] = true
	}
	return rp, nil
}

func min(a, b *int) *int {
	if *a < *b {
		return a
	}
	return b
}

func getMaxSize(mcMax, doptMax *int, defaultVal int) *int {
	if mcMax == nil && doptMax == nil {
		return &defaultVal
	}
	if mcMax != nil && doptMax != nil {
		return min(mcMax, doptMax)
	}
	if mcMax != nil {
		return mcMax
	}
	return doptMax
}

func newInt(b int) *int {
	return &b
}

func init() {
	internal.EqualServiceConfigForTesting = equalServiceConfig
}

// equalServiceConfig compares two configs. The rawJSONString field is ignored,
// because they may diff in white spaces.
//
// If any of them is NOT *ServiceConfig, return false.
func equalServiceConfig(a, b serviceconfig.Config) bool {
	if a == nil && b == nil {
		return true
	}
	aa, ok := a.(*ServiceConfig)
	if !ok {
		return false
	}
	bb, ok := b.(*ServiceConfig)
	if !ok {
		return false
	}
	aaRaw := aa.rawJSONString
	aa.rawJSONString = ""
	bbRaw := bb.rawJSONString
	bb.rawJSONString = ""
	defer func() {
		aa.rawJSONString = aaRaw
		bb.rawJSONString = bbRaw
	}()
	// Using reflect.DeepEqual instead of cmp.Equal because many balancer
	// configs are unexported, and cmp.Equal cannot compare unexported fields
	// from unexported structs.
	return reflect.DeepEqual(aa, bb)
}
