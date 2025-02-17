/*
 *
 * Copyright 2024 gRPC authors.
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

package balancer

import (
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/resolver"
)

// A SubConn represents a single connection to a gRPC backend service.
//
// All SubConns start in IDLE, and will not try to connect. To trigger a
// connection attempt, Balancers must call Connect.
//
// If the connection attempt fails, the SubConn will transition to
// TRANSIENT_FAILURE for a backoff period, and then return to IDLE.  If the
// connection attempt succeeds, it will transition to READY.
//
// If a READY SubConn becomes disconnected, the SubConn will transition to IDLE.
//
// If a connection re-enters IDLE, Balancers must call Connect again to trigger
// a new connection attempt.
//
// Each SubConn contains a list of addresses.  gRPC will try to connect to the
// addresses in sequence, and stop trying the remainder once the first
// connection is successful.  However, this behavior is deprecated.  SubConns
// should only use a single address.
//
// NOTICE: This interface is intended to be implemented by gRPC, or intercepted
// by custom load balancing poilices.  Users should not need their own complete
// implementation of this interface -- they should always delegate to a SubConn
// returned by ClientConn.NewSubConn() by embedding it in their implementations.
// An embedded SubConn must never be nil, or runtime panics will occur.
type SubConn interface {
	// UpdateAddresses updates the addresses used in this SubConn.
	// gRPC checks if currently-connected address is still in the new list.
	// If it's in the list, the connection will be kept.
	// If it's not in the list, the connection will gracefully close, and
	// a new connection will be created.
	//
	// This will trigger a state transition for the SubConn.
	//
	// Deprecated: this method will be removed.  Create new SubConns for new
	// addresses instead.
	UpdateAddresses([]resolver.Address)
	// Connect starts the connecting for this SubConn.
	Connect()
	// GetOrBuildProducer returns a reference to the existing Producer for this
	// ProducerBuilder in this SubConn, or, if one does not currently exist,
	// creates a new one and returns it.  Returns a close function which may be
	// called when the Producer is no longer needed.  Otherwise the producer
	// will automatically be closed upon connection loss or subchannel close.
	// Should only be called on a SubConn in state Ready.  Otherwise the
	// producer will be unable to create streams.
	GetOrBuildProducer(ProducerBuilder) (p Producer, close func())
	// Shutdown shuts down the SubConn gracefully.  Any started RPCs will be
	// allowed to complete.  No future calls should be made on the SubConn.
	// One final state update will be delivered to the StateListener (or
	// UpdateSubConnState; deprecated) with ConnectivityState of Shutdown to
	// indicate the shutdown operation.  This may be delivered before
	// in-progress RPCs are complete and the actual connection is closed.
	Shutdown()
	// RegisterHealthListener registers a health listener that receives health
	// updates for a Ready SubConn. Only one health listener can be registered
	// at a time. A health listener should be registered each time the SubConn's
	// connectivity state changes to READY. Registering a health listener when
	// the connectivity state is not READY may result in undefined behaviour.
	// This method must not be called synchronously while handling an update
	// from a previously registered health listener.
	RegisterHealthListener(func(SubConnState))
	// EnforceSubConnEmbedding is included to force implementers to embed
	// another implementation of this interface, allowing gRPC to add methods
	// without breaking users.
	internal.EnforceSubConnEmbedding
}

// A ProducerBuilder is a simple constructor for a Producer.  It is used by the
// SubConn to create producers when needed.
type ProducerBuilder interface {
	// Build creates a Producer.  The first parameter is always a
	// grpc.ClientConnInterface (a type to allow creating RPCs/streams on the
	// associated SubConn), but is declared as `any` to avoid a dependency
	// cycle.  Build also returns a close function that will be called when all
	// references to the Producer have been given up for a SubConn, or when a
	// connectivity state change occurs on the SubConn.  The close function
	// should always block until all asynchronous cleanup work is completed.
	Build(grpcClientConnInterface any) (p Producer, close func())
}

// SubConnState describes the state of a SubConn.
type SubConnState struct {
	// ConnectivityState is the connectivity state of the SubConn.
	ConnectivityState connectivity.State
	// ConnectionError is set if the ConnectivityState is TransientFailure,
	// describing the reason the SubConn failed.  Otherwise, it is nil.
	ConnectionError error
	// connectedAddr contains the connected address when ConnectivityState is
	// Ready. Otherwise, it is indeterminate.
	connectedAddress resolver.Address
}

// connectedAddress returns the connected address for a SubConnState. The
// address is only valid if the state is READY.
func connectedAddress(scs SubConnState) resolver.Address {
	return scs.connectedAddress
}

// setConnectedAddress sets the connected address for a SubConnState.
func setConnectedAddress(scs *SubConnState, addr resolver.Address) {
	scs.connectedAddress = addr
}

// A Producer is a type shared among potentially many consumers.  It is
// associated with a SubConn, and an implementation will typically contain
// other methods to provide additional functionality, e.g. configuration or
// subscription registration.
type Producer any
