/*
 *
 * Copyright 2018 gRPC authors.
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

// Package testutil include useful test utilities for the handshaker.
package testutil

import (
	"bytes"
	"encoding/binary"
	"io"
	"net"
	"sync"

	"google.golang.org/grpc/credentials/alts/internal/conn"
)

// Stats is used to collect statistics about concurrent handshake calls.
type Stats struct {
	mu                 sync.Mutex
	calls              int
	MaxConcurrentCalls int
}

// Update updates the statistics by adding one call.
func (s *Stats) Update() func() {
	s.mu.Lock()
	s.calls++
	if s.calls > s.MaxConcurrentCalls {
		s.MaxConcurrentCalls = s.calls
	}
	s.mu.Unlock()

	return func() {
		s.mu.Lock()
		s.calls--
		s.mu.Unlock()
	}
}

// Reset resets the statistics.
func (s *Stats) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = 0
	s.MaxConcurrentCalls = 0
}

// testConn mimics a net.Conn to the peer.
type testConn struct {
	net.Conn
	in  *bytes.Buffer
	out *bytes.Buffer
}

// NewTestConn creates a new instance of testConn object.
func NewTestConn(in *bytes.Buffer, out *bytes.Buffer) net.Conn {
	return &testConn{
		in:  in,
		out: out,
	}
}

// Read reads from the in buffer.
func (c *testConn) Read(b []byte) (n int, err error) {
	return c.in.Read(b)
}

// Write writes to the out buffer.
func (c *testConn) Write(b []byte) (n int, err error) {
	return c.out.Write(b)
}

// Close closes the testConn object.
func (c *testConn) Close() error {
	return nil
}

// unresponsiveTestConn mimics a net.Conn for an unresponsive peer. It is used
// for testing the PeerNotResponding case.
type unresponsiveTestConn struct {
	net.Conn
}

// NewUnresponsiveTestConn creates a new instance of unresponsiveTestConn object.
func NewUnresponsiveTestConn() net.Conn {
	return &unresponsiveTestConn{}
}

// Read reads from the in buffer.
func (c *unresponsiveTestConn) Read(b []byte) (n int, err error) {
	return 0, io.EOF
}

// Write writes to the out buffer.
func (c *unresponsiveTestConn) Write(b []byte) (n int, err error) {
	return 0, nil
}

// Close closes the TestConn object.
func (c *unresponsiveTestConn) Close() error {
	return nil
}

// MakeFrame creates a handshake frame.
func MakeFrame(pl string) []byte {
	f := make([]byte, len(pl)+conn.MsgLenFieldSize)
	binary.LittleEndian.PutUint32(f, uint32(len(pl)))
	copy(f[conn.MsgLenFieldSize:], []byte(pl))
	return f
}
