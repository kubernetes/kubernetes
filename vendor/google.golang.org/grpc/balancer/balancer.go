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

// Package balancer defines APIs for load balancing in gRPC.
// All APIs in this package are experimental.
package balancer

import (
	"errors"
	"net"
	"strings"

	"golang.org/x/net/context"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/resolver"
)

var (
	// m is a map from name to balancer builder.
	m = make(map[string]Builder)
)

// Register registers the balancer builder to the balancer map.
// b.Name (lowercased) will be used as the name registered with
// this builder.
func Register(b Builder) {
	m[strings.ToLower(b.Name())] = b
}

// Get returns the resolver builder registered with the given name.
// Note that the compare is done in a case-insenstive fashion.
// If no builder is register with the name, nil will be returned.
func Get(name string) Builder {
	if b, ok := m[strings.ToLower(name)]; ok {
		return b
	}
	return nil
}

// SubConn represents a gRPC sub connection.
// Each sub connection contains a list of addresses. gRPC will
// try to connect to them (in sequence), and stop trying the
// remainder once one connection is successful.
//
// The reconnect backoff will be applied on the list, not a single address.
// For example, try_on_all_addresses -> backoff -> try_on_all_addresses.
//
// All SubConns start in IDLE, and will not try to connect. To trigger
// the connecting, Balancers must call Connect.
// When the connection encounters an error, it will reconnect immediately.
// When the connection becomes IDLE, it will not reconnect unless Connect is
// called.
//
// This interface is to be implemented by gRPC. Users should not need a
// brand new implementation of this interface. For the situations like
// testing, the new implementation should embed this interface. This allows
// gRPC to add new methods to this interface.
type SubConn interface {
	// UpdateAddresses updates the addresses used in this SubConn.
	// gRPC checks if currently-connected address is still in the new list.
	// If it's in the list, the connection will be kept.
	// If it's not in the list, the connection will gracefully closed, and
	// a new connection will be created.
	//
	// This will trigger a state transition for the SubConn.
	UpdateAddresses([]resolver.Address)
	// Connect starts the connecting for this SubConn.
	Connect()
}

// NewSubConnOptions contains options to create new SubConn.
type NewSubConnOptions struct{}

// ClientConn represents a gRPC ClientConn.
//
// This interface is to be implemented by gRPC. Users should not need a
// brand new implementation of this interface. For the situations like
// testing, the new implementation should embed this interface. This allows
// gRPC to add new methods to this interface.
type ClientConn interface {
	// NewSubConn is called by balancer to create a new SubConn.
	// It doesn't block and wait for the connections to be established.
	// Behaviors of the SubConn can be controlled by options.
	NewSubConn([]resolver.Address, NewSubConnOptions) (SubConn, error)
	// RemoveSubConn removes the SubConn from ClientConn.
	// The SubConn will be shutdown.
	RemoveSubConn(SubConn)

	// UpdateBalancerState is called by balancer to nofity gRPC that some internal
	// state in balancer has changed.
	//
	// gRPC will update the connectivity state of the ClientConn, and will call pick
	// on the new picker to pick new SubConn.
	UpdateBalancerState(s connectivity.State, p Picker)

	// ResolveNow is called by balancer to notify gRPC to do a name resolving.
	ResolveNow(resolver.ResolveNowOption)

	// Target returns the dial target for this ClientConn.
	Target() string
}

// BuildOptions contains additional information for Build.
type BuildOptions struct {
	// DialCreds is the transport credential the Balancer implementation can
	// use to dial to a remote load balancer server. The Balancer implementations
	// can ignore this if it does not need to talk to another party securely.
	DialCreds credentials.TransportCredentials
	// Dialer is the custom dialer the Balancer implementation can use to dial
	// to a remote load balancer server. The Balancer implementations
	// can ignore this if it doesn't need to talk to remote balancer.
	Dialer func(context.Context, string) (net.Conn, error)
}

// Builder creates a balancer.
type Builder interface {
	// Build creates a new balancer with the ClientConn.
	Build(cc ClientConn, opts BuildOptions) Balancer
	// Name returns the name of balancers built by this builder.
	// It will be used to pick balancers (for example in service config).
	Name() string
}

// PickOptions contains addition information for the Pick operation.
type PickOptions struct{}

// DoneInfo contains additional information for done.
type DoneInfo struct {
	// Err is the rpc error the RPC finished with. It could be nil.
	Err error
	// BytesSent indicates if any bytes have been sent to the server.
	BytesSent bool
	// BytesReceived indicates if any byte has been received from the server.
	BytesReceived bool
}

var (
	// ErrNoSubConnAvailable indicates no SubConn is available for pick().
	// gRPC will block the RPC until a new picker is available via UpdateBalancerState().
	ErrNoSubConnAvailable = errors.New("no SubConn is available")
	// ErrTransientFailure indicates all SubConns are in TransientFailure.
	// WaitForReady RPCs will block, non-WaitForReady RPCs will fail.
	ErrTransientFailure = errors.New("all SubConns are in TransientFailure")
)

// Picker is used by gRPC to pick a SubConn to send an RPC.
// Balancer is expected to generate a new picker from its snapshot everytime its
// internal state has changed.
//
// The pickers used by gRPC can be updated by ClientConn.UpdateBalancerState().
type Picker interface {
	// Pick returns the SubConn to be used to send the RPC.
	// The returned SubConn must be one returned by NewSubConn().
	//
	// This functions is expected to return:
	// - a SubConn that is known to be READY;
	// - ErrNoSubConnAvailable if no SubConn is available, but progress is being
	//   made (for example, some SubConn is in CONNECTING mode);
	// - other errors if no active connecting is happening (for example, all SubConn
	//   are in TRANSIENT_FAILURE mode).
	//
	// If a SubConn is returned:
	// - If it is READY, gRPC will send the RPC on it;
	// - If it is not ready, or becomes not ready after it's returned, gRPC will block
	//   until UpdateBalancerState() is called and will call pick on the new picker.
	//
	// If the returned error is not nil:
	// - If the error is ErrNoSubConnAvailable, gRPC will block until UpdateBalancerState()
	// - If the error is ErrTransientFailure:
	//   - If the RPC is wait-for-ready, gRPC will block until UpdateBalancerState()
	//     is called to pick again;
	//   - Otherwise, RPC will fail with unavailable error.
	// - Else (error is other non-nil error):
	//   - The RPC will fail with unavailable error.
	//
	// The returned done() function will be called once the rpc has finished, with the
	// final status of that RPC.
	// done may be nil if balancer doesn't care about the RPC status.
	Pick(ctx context.Context, opts PickOptions) (conn SubConn, done func(DoneInfo), err error)
}

// Balancer takes input from gRPC, manages SubConns, and collects and aggregates
// the connectivity states.
//
// It also generates and updates the Picker used by gRPC to pick SubConns for RPCs.
//
// HandleSubConnectionStateChange, HandleResolvedAddrs and Close are guaranteed
// to be called synchronously from the same goroutine.
// There's no guarantee on picker.Pick, it may be called anytime.
type Balancer interface {
	// HandleSubConnStateChange is called by gRPC when the connectivity state
	// of sc has changed.
	// Balancer is expected to aggregate all the state of SubConn and report
	// that back to gRPC.
	// Balancer should also generate and update Pickers when its internal state has
	// been changed by the new state.
	HandleSubConnStateChange(sc SubConn, state connectivity.State)
	// HandleResolvedAddrs is called by gRPC to send updated resolved addresses to
	// balancers.
	// Balancer can create new SubConn or remove SubConn with the addresses.
	// An empty address slice and a non-nil error will be passed if the resolver returns
	// non-nil error to gRPC.
	HandleResolvedAddrs([]resolver.Address, error)
	// Close closes the balancer. The balancer is not required to call
	// ClientConn.RemoveSubConn for its existing SubConns.
	Close()
}
