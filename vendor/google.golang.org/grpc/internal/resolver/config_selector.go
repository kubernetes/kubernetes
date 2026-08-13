/*
 *
 * Copyright 2020 gRPC authors.
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

// Package resolver provides internal resolver-related functionality.
package resolver

import (
	"context"
	"sync"

	"google.golang.org/grpc/internal/serviceconfig"
	"google.golang.org/grpc/resolver"
)

// ConfigSelector controls what configuration to use for every RPC.
type ConfigSelector interface {
	// Selects the configuration for the RPC, or terminates it using the error.
	// This error will be converted by the gRPC library to a status error with
	// code UNKNOWN if it is not returned as a status error.
	SelectConfig(RPCInfo) (*RPCConfig, error)
}

// RPCInfo contains RPC information needed by a ConfigSelector.
type RPCInfo struct {
	// Context is the user's context for the RPC and contains headers and
	// application timeout.  It is passed for interception purposes and for
	// efficiency reasons.  SelectConfig should not be blocking.
	Context context.Context
	Method  string // i.e. "/Service/Method"
}

// RPCConfig describes the configuration to use for each RPC.
type RPCConfig struct {
	// The context to use for the remainder of the RPC; can pass info to LB
	// policy or affect timeout or metadata.
	Context      context.Context
	MethodConfig serviceconfig.MethodConfig // configuration to use for this RPC
	OnCommitted  func()                     // Called when the RPC has been committed (retries no longer possible)
	Interceptor  any
}

// ServerInterceptor is an interceptor for incoming RPC's on gRPC server side.
type ServerInterceptor interface {
	// AllowRPC checks if an incoming RPC is allowed to proceed based on
	// information about connection RPC was received on, and HTTP Headers. This
	// information will be piped into context.
	AllowRPC(ctx context.Context) error // TODO: Make this a real interceptor for filters such as rate limiting.
	// Close closes the interceptor. Once called, no new calls to NewStream are
	// accepted. Ongoing calls to NewStream are allowed to complete.
	Close()
}

type csKeyType string

const csKey = csKeyType("grpc.internal.resolver.configSelector")

// SetConfigSelector sets the config selector in state and returns the new
// state.
func SetConfigSelector(state resolver.State, cs ConfigSelector) resolver.State {
	state.Attributes = state.Attributes.WithValue(csKey, cs)
	return state
}

// GetConfigSelector retrieves the config selector from state, if present, and
// returns it or nil if absent.
func GetConfigSelector(state resolver.State) ConfigSelector {
	cs, _ := state.Attributes.Value(csKey).(ConfigSelector)
	return cs
}

// SafeConfigSelector allows for safe switching of ConfigSelector
// implementations such that previous values are guaranteed to not be in use
// when UpdateConfigSelector returns.
type SafeConfigSelector struct {
	mu sync.RWMutex
	cs ConfigSelector
}

// UpdateConfigSelector swaps to the provided ConfigSelector and blocks until
// all uses of the previous ConfigSelector have completed.
func (scs *SafeConfigSelector) UpdateConfigSelector(cs ConfigSelector) {
	scs.mu.Lock()
	defer scs.mu.Unlock()
	scs.cs = cs
}

// SelectConfig defers to the current ConfigSelector in scs.
func (scs *SafeConfigSelector) SelectConfig(r RPCInfo) (*RPCConfig, error) {
	scs.mu.RLock()
	defer scs.mu.RUnlock()
	return scs.cs.SelectConfig(r)
}
