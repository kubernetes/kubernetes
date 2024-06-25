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

// Package internal contains functionality internal to the dns resolver package.
package internal

import (
	"context"
	"errors"
	"net"
	"time"
)

// NetResolver groups the methods on net.Resolver that are used by the DNS
// resolver implementation. This allows the default net.Resolver instance to be
// overridden from tests.
type NetResolver interface {
	LookupHost(ctx context.Context, host string) (addrs []string, err error)
	LookupSRV(ctx context.Context, service, proto, name string) (cname string, addrs []*net.SRV, err error)
	LookupTXT(ctx context.Context, name string) (txts []string, err error)
}

var (
	// ErrMissingAddr is the error returned when building a DNS resolver when
	// the provided target name is empty.
	ErrMissingAddr = errors.New("dns resolver: missing address")

	// ErrEndsWithColon is the error returned when building a DNS resolver when
	// the provided target name ends with a colon that is supposed to be the
	// separator between host and port.  E.g. "::" is a valid address as it is
	// an IPv6 address (host only) and "[::]:" is invalid as it ends with a
	// colon as the host and port separator
	ErrEndsWithColon = errors.New("dns resolver: missing port after port-separator colon")
)

// The following vars are overridden from tests.
var (
	// TimeAfterFunc is used by the DNS resolver to wait for the given duration
	// to elapse. In non-test code, this is implemented by time.After.  In test
	// code, this can be used to control the amount of time the resolver is
	// blocked waiting for the duration to elapse.
	TimeAfterFunc func(time.Duration) <-chan time.Time

	// NewNetResolver returns the net.Resolver instance for the given target.
	NewNetResolver func(string) (NetResolver, error)

	// AddressDialer is the dialer used to dial the DNS server. It accepts the
	// Host portion of the URL corresponding to the user's dial target and
	// returns a dial function.
	AddressDialer func(address string) func(context.Context, string, string) (net.Conn, error)
)
