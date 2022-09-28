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

// Package connectivity defines connectivity semantics.
// For details, see https://github.com/grpc/grpc/blob/master/doc/connectivity-semantics-and-api.md.
package connectivity

import (
	"google.golang.org/grpc/grpclog"
)

var logger = grpclog.Component("core")

// State indicates the state of connectivity.
// It can be the state of a ClientConn or SubConn.
type State int

func (s State) String() string {
	switch s {
	case Idle:
		return "IDLE"
	case Connecting:
		return "CONNECTING"
	case Ready:
		return "READY"
	case TransientFailure:
		return "TRANSIENT_FAILURE"
	case Shutdown:
		return "SHUTDOWN"
	default:
		logger.Errorf("unknown connectivity state: %d", s)
		return "INVALID_STATE"
	}
}

const (
	// Idle indicates the ClientConn is idle.
	Idle State = iota
	// Connecting indicates the ClientConn is connecting.
	Connecting
	// Ready indicates the ClientConn is ready for work.
	Ready
	// TransientFailure indicates the ClientConn has seen a failure but expects to recover.
	TransientFailure
	// Shutdown indicates the ClientConn has started shutting down.
	Shutdown
)

// ServingMode indicates the current mode of operation of the server.
//
// Only xDS enabled gRPC servers currently report their serving mode.
type ServingMode int

const (
	// ServingModeStarting indicates that the server is starting up.
	ServingModeStarting ServingMode = iota
	// ServingModeServing indicates that the server contains all required
	// configuration and is serving RPCs.
	ServingModeServing
	// ServingModeNotServing indicates that the server is not accepting new
	// connections. Existing connections will be closed gracefully, allowing
	// in-progress RPCs to complete. A server enters this mode when it does not
	// contain the required configuration to serve RPCs.
	ServingModeNotServing
)

func (s ServingMode) String() string {
	switch s {
	case ServingModeStarting:
		return "STARTING"
	case ServingModeServing:
		return "SERVING"
	case ServingModeNotServing:
		return "NOT_SERVING"
	default:
		logger.Errorf("unknown serving mode: %d", s)
		return "INVALID_MODE"
	}
}
