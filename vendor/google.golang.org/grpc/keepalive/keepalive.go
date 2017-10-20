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

// Package keepalive defines configurable parameters for point-to-point healthcheck.
package keepalive

import (
	"time"
)

// ClientParameters is used to set keepalive parameters on the client-side.
// These configure how the client will actively probe to notice when a connection is broken
// and send pings so intermediaries will be aware of the liveness of the connection.
// Make sure these parameters are set in coordination with the keepalive policy on the server,
// as incompatible settings can result in closing of connection.
type ClientParameters struct {
	// After a duration of this time if the client doesn't see any activity it pings the server to see if the transport is still alive.
	Time time.Duration // The current default value is infinity.
	// After having pinged for keepalive check, the client waits for a duration of Timeout and if no activity is seen even after that
	// the connection is closed.
	Timeout time.Duration // The current default value is 20 seconds.
	// If true, client runs keepalive checks even with no active RPCs.
	PermitWithoutStream bool // false by default.
}

// ServerParameters is used to set keepalive and max-age parameters on the server-side.
type ServerParameters struct {
	// MaxConnectionIdle is a duration for the amount of time after which an idle connection would be closed by sending a GoAway.
	// Idleness duration is defined since the most recent time the number of outstanding RPCs became zero or the connection establishment.
	MaxConnectionIdle time.Duration // The current default value is infinity.
	// MaxConnectionAge is a duration for the maximum amount of time a connection may exist before it will be closed by sending a GoAway.
	// A random jitter of +/-10% will be added to MaxConnectionAge to spread out connection storms.
	MaxConnectionAge time.Duration // The current default value is infinity.
	// MaxConnectinoAgeGrace is an additive period after MaxConnectionAge after which the connection will be forcibly closed.
	MaxConnectionAgeGrace time.Duration // The current default value is infinity.
	// After a duration of this time if the server doesn't see any activity it pings the client to see if the transport is still alive.
	Time time.Duration // The current default value is 2 hours.
	// After having pinged for keepalive check, the server waits for a duration of Timeout and if no activity is seen even after that
	// the connection is closed.
	Timeout time.Duration // The current default value is 20 seconds.
}

// EnforcementPolicy is used to set keepalive enforcement policy on the server-side.
// Server will close connection with a client that violates this policy.
type EnforcementPolicy struct {
	// MinTime is the minimum amount of time a client should wait before sending a keepalive ping.
	MinTime time.Duration // The current default value is 5 minutes.
	// If true, server expects keepalive pings even when there are no active streams(RPCs).
	PermitWithoutStream bool // false by default.
}
