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
	"fmt"
	"strconv"
	"strings"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/serviceconfig"
)

const maxInt = int(^uint(0) >> 1)

// MethodConfig defines the configuration recommended by the service providers for a
// particular method.
//
// Deprecated: Users should not use this struct. Service config should be received
// through name resolver, as specified here
// https://github.com/grpc/grpc/blob/master/doc/service_config.md
type MethodConfig struct {
	// WaitForReady indicates whether RPCs sent to this method should wait until
	// the connection is ready by default (!failfast). The value specified via the
	// gRPC client API will override the value set here.
	WaitForReady *bool
	// Timeout is the default timeout for RPCs sent to this method. The actual
	// deadline used will be the minimum of the value specified here and the value
	// set by the application via the gRPC client API.  If either one is not set,
	// then the other will be used.  If neither is set, then the RPC has no deadline.
	Timeout *time.Duration
	// MaxReqSize is the maximum allowed payload size for an individual request in a
	// stream (client->server) in bytes. The size which is measured is the serialized
	// payload after per-message compression (but before stream compression) in bytes.
	// The actual value used is the minimum of the value specified here and the value set
	// by the application via the gRPC client API. If either one is not set, then the other
	// will be used.  If neither is set, then the built-in default is used.
	MaxReqSize *int
	// MaxRespSize is the maximum allowed payload size for an individual response in a
	// stream (server->client) in bytes.
	MaxRespSize *int
	// RetryPolicy configures retry options for the method.
	retryPolicy *retryPolicy
}

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

	// LB is the load balancer the service providers recommends. The balancer
	// specified via grpc.WithBalancer will override this.  This is deprecated;
	// lbConfigs is preferred.  If lbConfig and LB are both present, lbConfig
	// will be used.
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

// retryPolicy defines the go-native version of the retry policy defined by the
// service config here:
// https://github.com/grpc/proposal/blob/master/A6-client-retries.md#integration-with-service-config
type retryPolicy struct {
	// MaxAttempts is the maximum number of attempts, including the original RPC.
	//
	// This field is required and must be two or greater.
	maxAttempts int

	// Exponential backoff parameters. The initial retry attempt will occur at
	// random(0, initialBackoff). In general, the nth attempt will occur at
	// random(0,
	//   min(initialBackoff*backoffMultiplier**(n-1), maxBackoff)).
	//
	// These fields are required and must be greater than zero.
	initialBackoff    time.Duration
	maxBackoff        time.Duration
	backoffMultiplier float64

	// The set of status codes which may be retried.
	//
	// Status codes are specified as strings, e.g., "UNAVAILABLE".
	//
	// This field is required and must be non-empty.
	// Note: a set is used to store this for easy lookup.
	retryableStatusCodes map[codes.Code]bool
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
	Service *string
	Method  *string
}

func (j jsonName) generatePath() (string, bool) {
	if j.Service == nil {
		return "", false
	}
	res := "/" + *j.Service + "/"
	if j.Method != nil {
		res += *j.Method
	}
	return res, true
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

type loadBalancingConfig map[string]json.RawMessage

// TODO(lyuxuan): delete this struct after cleaning up old service config implementation.
type jsonSC struct {
	LoadBalancingPolicy *string
	LoadBalancingConfig *[]loadBalancingConfig
	MethodConfig        *[]jsonMC
	RetryThrottling     *retryThrottlingPolicy
	HealthCheckConfig   *healthCheckConfig
}

func init() {
	internal.ParseServiceConfigForTesting = parseServiceConfig
}
func parseServiceConfig(js string) *serviceconfig.ParseResult {
	if len(js) == 0 {
		return &serviceconfig.ParseResult{Err: fmt.Errorf("no JSON service config provided")}
	}
	var rsc jsonSC
	err := json.Unmarshal([]byte(js), &rsc)
	if err != nil {
		grpclog.Warningf("grpc: parseServiceConfig error unmarshaling %s due to %v", js, err)
		return &serviceconfig.ParseResult{Err: err}
	}
	sc := ServiceConfig{
		LB:                rsc.LoadBalancingPolicy,
		Methods:           make(map[string]MethodConfig),
		retryThrottling:   rsc.RetryThrottling,
		healthCheckConfig: rsc.HealthCheckConfig,
		rawJSONString:     js,
	}
	if rsc.LoadBalancingConfig != nil {
		for i, lbcfg := range *rsc.LoadBalancingConfig {
			if len(lbcfg) != 1 {
				err := fmt.Errorf("invalid loadBalancingConfig: entry %v does not contain exactly 1 policy/config pair: %q", i, lbcfg)
				grpclog.Warningf(err.Error())
				return &serviceconfig.ParseResult{Err: err}
			}
			var name string
			var jsonCfg json.RawMessage
			for name, jsonCfg = range lbcfg {
			}
			builder := balancer.Get(name)
			if builder == nil {
				continue
			}
			sc.lbConfig = &lbConfig{name: name}
			if parser, ok := builder.(balancer.ConfigParser); ok {
				var err error
				sc.lbConfig.cfg, err = parser.ParseConfig(jsonCfg)
				if err != nil {
					return &serviceconfig.ParseResult{Err: fmt.Errorf("error parsing loadBalancingConfig for policy %q: %v", name, err)}
				}
			} else if string(jsonCfg) != "{}" {
				grpclog.Warningf("non-empty balancer configuration %q, but balancer does not implement ParseConfig", string(jsonCfg))
			}
			break
		}
		if sc.lbConfig == nil {
			// We had a loadBalancingConfig field but did not encounter a
			// supported policy.  The config is considered invalid in this
			// case.
			err := fmt.Errorf("invalid loadBalancingConfig: no supported policies found")
			grpclog.Warningf(err.Error())
			return &serviceconfig.ParseResult{Err: err}
		}
	}

	if rsc.MethodConfig == nil {
		return &serviceconfig.ParseResult{Config: &sc}
	}
	for _, m := range *rsc.MethodConfig {
		if m.Name == nil {
			continue
		}
		d, err := parseDuration(m.Timeout)
		if err != nil {
			grpclog.Warningf("grpc: parseServiceConfig error unmarshaling %s due to %v", js, err)
			return &serviceconfig.ParseResult{Err: err}
		}

		mc := MethodConfig{
			WaitForReady: m.WaitForReady,
			Timeout:      d,
		}
		if mc.retryPolicy, err = convertRetryPolicy(m.RetryPolicy); err != nil {
			grpclog.Warningf("grpc: parseServiceConfig error unmarshaling %s due to %v", js, err)
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
		for _, n := range *m.Name {
			if path, valid := n.generatePath(); valid {
				sc.Methods[path] = mc
			}
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

func convertRetryPolicy(jrp *jsonRetryPolicy) (p *retryPolicy, err error) {
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
		grpclog.Warningf("grpc: ignoring retry policy %v due to illegal configuration", jrp)
		return nil, nil
	}

	rp := &retryPolicy{
		maxAttempts:          jrp.MaxAttempts,
		initialBackoff:       *ib,
		maxBackoff:           *mb,
		backoffMultiplier:    jrp.BackoffMultiplier,
		retryableStatusCodes: make(map[codes.Code]bool),
	}
	if rp.maxAttempts > 5 {
		// TODO(retry): Make the max maxAttempts configurable.
		rp.maxAttempts = 5
	}
	for _, code := range jrp.RetryableStatusCodes {
		rp.retryableStatusCodes[code] = true
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
