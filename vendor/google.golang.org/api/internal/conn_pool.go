// Copyright 2020 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"google.golang.org/grpc"
)

// ConnPool is a pool of grpc.ClientConns.
type ConnPool interface {
	// Conn returns a ClientConn from the pool.
	//
	// Conns aren't returned to the pool.
	Conn() *grpc.ClientConn

	// Num returns the number of connections in the pool.
	//
	// It will always return the same value.
	Num() int

	// Close closes every ClientConn in the pool.
	//
	// The error returned by Close may be a single error or multiple errors.
	Close() error

	// ConnPool implements grpc.ClientConnInterface to enable it to be used directly with generated proto stubs.
	grpc.ClientConnInterface
}
