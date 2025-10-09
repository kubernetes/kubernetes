/*
 *
 * Copyright 2023 gRPC authors.
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

// Package nop implements a balancer with all of its balancer operations as
// no-ops, other than returning a Transient Failure Picker on a Client Conn
// update.
package nop

import (
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/base"
	"google.golang.org/grpc/connectivity"
)

// bal is a balancer with all of its balancer operations as no-ops, other than
// returning a Transient Failure Picker on a Client Conn update.
type bal struct {
	cc  balancer.ClientConn
	err error
}

// NewBalancer returns a no-op balancer.
func NewBalancer(cc balancer.ClientConn, err error) balancer.Balancer {
	return &bal{
		cc:  cc,
		err: err,
	}
}

// UpdateClientConnState updates the bal's Client Conn with an Error Picker
// and a Connectivity State of TRANSIENT_FAILURE.
func (b *bal) UpdateClientConnState(_ balancer.ClientConnState) error {
	b.cc.UpdateState(balancer.State{
		Picker:            base.NewErrPicker(b.err),
		ConnectivityState: connectivity.TransientFailure,
	})
	return nil
}

// ResolverError is a no-op.
func (b *bal) ResolverError(_ error) {}

// UpdateSubConnState is a no-op.
func (b *bal) UpdateSubConnState(_ balancer.SubConn, _ balancer.SubConnState) {}

// Close is a no-op.
func (b *bal) Close() {}

// ExitIdle is a no-op.
func (b *bal) ExitIdle() {}
