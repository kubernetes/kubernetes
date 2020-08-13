/*
 * Copyright 2016 gRPC authors.
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

// Package internal contains gRPC-internal code, to avoid polluting
// the godoc of the top-level grpc package.  It must not import any grpc
// symbols to avoid circular dependencies.
package internal

import (
	"context"
	"time"

	"google.golang.org/grpc/connectivity"
)

var (
	// WithHealthCheckFunc is set by dialoptions.go
	WithHealthCheckFunc interface{} // func (HealthChecker) DialOption
	// HealthCheckFunc is used to provide client-side LB channel health checking
	HealthCheckFunc HealthChecker
	// BalancerUnregister is exported by package balancer to unregister a balancer.
	BalancerUnregister func(name string)
	// KeepaliveMinPingTime is the minimum ping interval.  This must be 10s by
	// default, but tests may wish to set it lower for convenience.
	KeepaliveMinPingTime = 10 * time.Second
	// StatusRawProto is exported by status/status.go. This func returns a
	// pointer to the wrapped Status proto for a given status.Status without a
	// call to proto.Clone(). The returned Status proto should not be mutated by
	// the caller.
	StatusRawProto interface{} // func (*status.Status) *spb.Status
	// NewRequestInfoContext creates a new context based on the argument context attaching
	// the passed in RequestInfo to the new context.
	NewRequestInfoContext interface{} // func(context.Context, credentials.RequestInfo) context.Context
	// ParseServiceConfigForTesting is for creating a fake
	// ClientConn for resolver testing only
	ParseServiceConfigForTesting interface{} // func(string) *serviceconfig.ParseResult
)

// HealthChecker defines the signature of the client-side LB channel health checking function.
//
// The implementation is expected to create a health checking RPC stream by
// calling newStream(), watch for the health status of serviceName, and report
// it's health back by calling setConnectivityState().
//
// The health checking protocol is defined at:
// https://github.com/grpc/grpc/blob/master/doc/health-checking.md
type HealthChecker func(ctx context.Context, newStream func(string) (interface{}, error), setConnectivityState func(connectivity.State, error), serviceName string) error

const (
	// CredsBundleModeFallback switches GoogleDefaultCreds to fallback mode.
	CredsBundleModeFallback = "fallback"
	// CredsBundleModeBalancer switches GoogleDefaultCreds to grpclb balancer
	// mode.
	CredsBundleModeBalancer = "balancer"
	// CredsBundleModeBackendFromBalancer switches GoogleDefaultCreds to mode
	// that supports backend returned by grpclb balancer.
	CredsBundleModeBackendFromBalancer = "backend-from-balancer"
)
