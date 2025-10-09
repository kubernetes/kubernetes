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

// Package fault implements the Envoy Fault Injection HTTP filter.
package fault

import (
	"context"
	"errors"
	"fmt"
	"io"
	rand "math/rand/v2"
	"strconv"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/codes"
	iresolver "google.golang.org/grpc/internal/resolver"
	"google.golang.org/grpc/internal/xds/httpfilter"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"

	cpb "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/common/fault/v3"
	fpb "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/fault/v3"
	tpb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
)

const headerAbortHTTPStatus = "x-envoy-fault-abort-request"
const headerAbortGRPCStatus = "x-envoy-fault-abort-grpc-request"
const headerAbortPercentage = "x-envoy-fault-abort-request-percentage"

const headerDelayPercentage = "x-envoy-fault-delay-request-percentage"
const headerDelayDuration = "x-envoy-fault-delay-request"

var statusMap = map[int]codes.Code{
	400: codes.Internal,
	401: codes.Unauthenticated,
	403: codes.PermissionDenied,
	404: codes.Unimplemented,
	429: codes.Unavailable,
	502: codes.Unavailable,
	503: codes.Unavailable,
	504: codes.Unavailable,
}

func init() {
	httpfilter.Register(builder{})
}

type builder struct {
}

type config struct {
	httpfilter.FilterConfig
	config *fpb.HTTPFault
}

func (builder) TypeURLs() []string {
	return []string{"type.googleapis.com/envoy.extensions.filters.http.fault.v3.HTTPFault"}
}

// Parsing is the same for the base config and the override config.
func parseConfig(cfg proto.Message) (httpfilter.FilterConfig, error) {
	if cfg == nil {
		return nil, fmt.Errorf("fault: nil configuration message provided")
	}
	m, ok := cfg.(*anypb.Any)
	if !ok {
		return nil, fmt.Errorf("fault: error parsing config %v: unknown type %T", cfg, cfg)
	}
	msg := new(fpb.HTTPFault)
	if err := m.UnmarshalTo(msg); err != nil {
		return nil, fmt.Errorf("fault: error parsing config %v: %v", cfg, err)
	}
	return config{config: msg}, nil
}

func (builder) ParseFilterConfig(cfg proto.Message) (httpfilter.FilterConfig, error) {
	return parseConfig(cfg)
}

func (builder) ParseFilterConfigOverride(override proto.Message) (httpfilter.FilterConfig, error) {
	return parseConfig(override)
}

func (builder) IsTerminal() bool {
	return false
}

var _ httpfilter.ClientInterceptorBuilder = builder{}

func (builder) BuildClientInterceptor(cfg, override httpfilter.FilterConfig) (iresolver.ClientInterceptor, error) {
	if cfg == nil {
		return nil, fmt.Errorf("fault: nil config provided")
	}

	c, ok := cfg.(config)
	if !ok {
		return nil, fmt.Errorf("fault: incorrect config type provided (%T): %v", cfg, cfg)
	}

	if override != nil {
		// override completely replaces the listener configuration; but we
		// still validate the listener config type.
		c, ok = override.(config)
		if !ok {
			return nil, fmt.Errorf("fault: incorrect override config type provided (%T): %v", override, override)
		}
	}

	icfg := c.config
	if (icfg.GetMaxActiveFaults() != nil && icfg.GetMaxActiveFaults().GetValue() == 0) ||
		(icfg.GetDelay() == nil && icfg.GetAbort() == nil) {
		return nil, nil
	}
	return &interceptor{config: icfg}, nil
}

type interceptor struct {
	config *fpb.HTTPFault
}

var activeFaults uint32 // global active faults; accessed atomically

func (i *interceptor) NewStream(ctx context.Context, _ iresolver.RPCInfo, done func(), newStream func(ctx context.Context, done func()) (iresolver.ClientStream, error)) (iresolver.ClientStream, error) {
	if maxAF := i.config.GetMaxActiveFaults(); maxAF != nil {
		defer atomic.AddUint32(&activeFaults, ^uint32(0)) // decrement counter
		if af := atomic.AddUint32(&activeFaults, 1); af > maxAF.GetValue() {
			// Would exceed maximum active fault limit.
			return newStream(ctx, done)
		}
	}

	if err := injectDelay(ctx, i.config.GetDelay()); err != nil {
		return nil, err
	}

	if err := injectAbort(ctx, i.config.GetAbort()); err != nil {
		if err == errOKStream {
			return &okStream{ctx: ctx}, nil
		}
		return nil, err
	}
	return newStream(ctx, done)
}

// For overriding in tests
var randIntn = rand.IntN
var newTimer = time.NewTimer

func injectDelay(ctx context.Context, delayCfg *cpb.FaultDelay) error {
	numerator, denominator := splitPct(delayCfg.GetPercentage())
	var delay time.Duration
	switch delayType := delayCfg.GetFaultDelaySecifier().(type) {
	case *cpb.FaultDelay_FixedDelay:
		delay = delayType.FixedDelay.AsDuration()
	case *cpb.FaultDelay_HeaderDelay_:
		md, _ := metadata.FromOutgoingContext(ctx)
		v := md[headerDelayDuration]
		if v == nil {
			// No delay configured for this RPC.
			return nil
		}
		ms, ok := parseIntFromMD(v)
		if !ok {
			// Malformed header; no delay.
			return nil
		}
		delay = time.Duration(ms) * time.Millisecond
		if v := md[headerDelayPercentage]; v != nil {
			if num, ok := parseIntFromMD(v); ok && num < numerator {
				numerator = num
			}
		}
	}
	if delay == 0 || randIntn(denominator) >= numerator {
		return nil
	}
	t := newTimer(delay)
	select {
	case <-t.C:
	case <-ctx.Done():
		t.Stop()
		return ctx.Err()
	}
	return nil
}

func injectAbort(ctx context.Context, abortCfg *fpb.FaultAbort) error {
	numerator, denominator := splitPct(abortCfg.GetPercentage())
	code := codes.OK
	okCode := false
	switch errType := abortCfg.GetErrorType().(type) {
	case *fpb.FaultAbort_HttpStatus:
		code, okCode = grpcFromHTTP(int(errType.HttpStatus))
	case *fpb.FaultAbort_GrpcStatus:
		code, okCode = sanitizeGRPCCode(codes.Code(errType.GrpcStatus)), true
	case *fpb.FaultAbort_HeaderAbort_:
		md, _ := metadata.FromOutgoingContext(ctx)
		if v := md[headerAbortHTTPStatus]; v != nil {
			// HTTP status has priority over gRPC status.
			if httpStatus, ok := parseIntFromMD(v); ok {
				code, okCode = grpcFromHTTP(httpStatus)
			}
		} else if v := md[headerAbortGRPCStatus]; v != nil {
			if grpcStatus, ok := parseIntFromMD(v); ok {
				code, okCode = sanitizeGRPCCode(codes.Code(grpcStatus)), true
			}
		}
		if v := md[headerAbortPercentage]; v != nil {
			if num, ok := parseIntFromMD(v); ok && num < numerator {
				numerator = num
			}
		}
	}
	if !okCode || randIntn(denominator) >= numerator {
		return nil
	}
	if code == codes.OK {
		return errOKStream
	}
	return status.Errorf(code, "RPC terminated due to fault injection")
}

var errOKStream = errors.New("stream terminated early with OK status")

// parseIntFromMD returns the integer in the last header or nil if parsing
// failed.
func parseIntFromMD(header []string) (int, bool) {
	if len(header) == 0 {
		return 0, false
	}
	v, err := strconv.Atoi(header[len(header)-1])
	return v, err == nil
}

func splitPct(fp *tpb.FractionalPercent) (num int, den int) {
	if fp == nil {
		return 0, 100
	}
	num = int(fp.GetNumerator())
	switch fp.GetDenominator() {
	case tpb.FractionalPercent_HUNDRED:
		return num, 100
	case tpb.FractionalPercent_TEN_THOUSAND:
		return num, 10 * 1000
	case tpb.FractionalPercent_MILLION:
		return num, 1000 * 1000
	}
	return num, 100
}

func grpcFromHTTP(httpStatus int) (codes.Code, bool) {
	if httpStatus < 200 || httpStatus >= 600 {
		// Malformed; ignore this fault type.
		return codes.OK, false
	}
	if c := statusMap[httpStatus]; c != codes.OK {
		// OK = 0/the default for the map.
		return c, true
	}
	// All undefined HTTP status codes convert to Unknown. HTTP status of 200
	// is "success", but gRPC converts to Unknown due to missing grpc status.
	return codes.Unknown, true
}

func sanitizeGRPCCode(c codes.Code) codes.Code {
	if c > codes.Code(16) {
		return codes.Unknown
	}
	return c
}

type okStream struct {
	ctx context.Context
}

func (*okStream) Header() (metadata.MD, error) { return nil, nil }
func (*okStream) Trailer() metadata.MD         { return nil }
func (*okStream) CloseSend() error             { return nil }
func (o *okStream) Context() context.Context   { return o.ctx }
func (*okStream) SendMsg(any) error            { return io.EOF }
func (*okStream) RecvMsg(any) error            { return io.EOF }
