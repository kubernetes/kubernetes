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

// Package ringhash (internal) contains functions and types that need to be
// shared by the ring hash balancer and other gRPC code (such as xDS)
// without being exported.
package ringhash

import (
	"context"

	"google.golang.org/grpc/serviceconfig"
)

// LBConfig is the balancer config for ring_hash balancer.
type LBConfig struct {
	serviceconfig.LoadBalancingConfig `json:"-"`

	MinRingSize       uint64 `json:"minRingSize,omitempty"`
	MaxRingSize       uint64 `json:"maxRingSize,omitempty"`
	RequestHashHeader string `json:"requestHashHeader,omitempty"`
}

// xdsHashKey is the type used as the key to store request hash in the context
// used when combining the Ring Hash load balancing policy with xDS.
type xdsHashKey struct{}

// XDSRequestHash returns the request hash in the context and true if it was set
// from the xDS config selector. If the xDS config selector has not set the hash,
// it returns 0 and false.
func XDSRequestHash(ctx context.Context) (uint64, bool) {
	requestHash := ctx.Value(xdsHashKey{})
	if requestHash == nil {
		return 0, false
	}
	return requestHash.(uint64), true
}

// SetXDSRequestHash adds the request hash to the context for use in Ring Hash
// Load Balancing using xDS route hash_policy.
func SetXDSRequestHash(ctx context.Context, requestHash uint64) context.Context {
	return context.WithValue(ctx, xdsHashKey{}, requestHash)
}
