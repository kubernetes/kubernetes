/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package grpc

import (
	"errors"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/coreos/etcd/Godeps/_workspace/src/golang.org/x/net/context"
	"github.com/coreos/etcd/Godeps/_workspace/src/google.golang.org/grpc/credentials"
	"github.com/coreos/etcd/Godeps/_workspace/src/google.golang.org/grpc/grpclog"
	"github.com/coreos/etcd/Godeps/_workspace/src/google.golang.org/grpc/transport"
)

var (
	// ErrUnspecTarget indicates that the target address is unspecified.
	ErrUnspecTarget = errors.New("grpc: target is unspecified")
	// ErrClientConnClosing indicates that the operation is illegal because
	// the session is closing.
	ErrClientConnClosing = errors.New("grpc: the client connection is closing")
	// ErrClientConnTimeout indicates that the connection could not be
	// established or re-established within the specified timeout.
	ErrClientConnTimeout = errors.New("grpc: timed out trying to connect")
)

// dialOptions configure a Dial call. dialOptions are set by the DialOption
// values passed to Dial.
type dialOptions struct {
	codec Codec
	copts transport.ConnectOptions
}

// DialOption configures how we set up the connection.
type DialOption func(*dialOptions)

// WithCodec returns a DialOption which sets a codec for message marshaling and unmarshaling.
func WithCodec(c Codec) DialOption {
	return func(o *dialOptions) {
		o.codec = c
	}
}

// WithTransportCredentials returns a DialOption which configures a
// connection level security credentials (e.g., TLS/SSL).
func WithTransportCredentials(creds credentials.TransportAuthenticator) DialOption {
	return func(o *dialOptions) {
		o.copts.AuthOptions = append(o.copts.AuthOptions, creds)
	}
}

// WithPerRPCCredentials returns a DialOption which sets
// credentials which will place auth state on each outbound RPC.
func WithPerRPCCredentials(creds credentials.Credentials) DialOption {
	return func(o *dialOptions) {
		o.copts.AuthOptions = append(o.copts.AuthOptions, creds)
	}
}

// WithTimeout returns a DialOption that configures a timeout for dialing a client connection.
func WithTimeout(d time.Duration) DialOption {
	return func(o *dialOptions) {
		o.copts.Timeout = d
	}
}

// WithDialer returns a DialOption that specifies a function to use for dialing network addresses.
func WithDialer(f func(addr string, timeout time.Duration) (net.Conn, error)) DialOption {
	return func(o *dialOptions) {
		o.copts.Dialer = f
	}
}

// Dial creates a client connection the given target.
// TODO(zhaoq): Have an option to make Dial return immediately without waiting
// for connection to complete.
func Dial(target string, opts ...DialOption) (*ClientConn, error) {
	if target == "" {
		return nil, ErrUnspecTarget
	}
	cc := &ClientConn{
		target: target,
	}
	for _, opt := range opts {
		opt(&cc.dopts)
	}
	colonPos := strings.LastIndex(target, ":")
	if colonPos == -1 {
		colonPos = len(target)
	}
	cc.authority = target[:colonPos]
	if cc.dopts.codec == nil {
		// Set the default codec.
		cc.dopts.codec = protoCodec{}
	}
	if err := cc.resetTransport(false); err != nil {
		return nil, err
	}
	cc.shutdownChan = make(chan struct{})
	// Start to monitor the error status of transport.
	go cc.transportMonitor()
	return cc, nil
}

// ClientConn represents a client connection to an RPC service.
type ClientConn struct {
	target       string
	authority    string
	dopts        dialOptions
	shutdownChan chan struct{}

	mu sync.Mutex
	// ready is closed and becomes nil when a new transport is up or failed
	// due to timeout.
	ready chan struct{}
	// Indicates the ClientConn is under destruction.
	closing bool
	// Every time a new transport is created, this is incremented by 1. Used
	// to avoid trying to recreate a transport while the new one is already
	// under construction.
	transportSeq int
	transport    transport.ClientTransport
}

func (cc *ClientConn) resetTransport(closeTransport bool) error {
	var retries int
	start := time.Now()
	for {
		cc.mu.Lock()
		t := cc.transport
		ts := cc.transportSeq
		// Avoid wait() picking up a dying transport unnecessarily.
		cc.transportSeq = 0
		if cc.closing {
			cc.mu.Unlock()
			return ErrClientConnClosing
		}
		cc.mu.Unlock()
		if closeTransport {
			t.Close()
		}
		// Adjust timeout for the current try.
		copts := cc.dopts.copts
		if copts.Timeout < 0 {
			cc.Close()
			return ErrClientConnTimeout
		}
		if copts.Timeout > 0 {
			copts.Timeout -= time.Since(start)
			if copts.Timeout <= 0 {
				cc.Close()
				return ErrClientConnTimeout
			}
		}
		newTransport, err := transport.NewClientTransport(cc.target, &copts)
		if err != nil {
			sleepTime := backoff(retries)
			// Fail early before falling into sleep.
			if cc.dopts.copts.Timeout > 0 && cc.dopts.copts.Timeout < sleepTime+time.Since(start) {
				cc.Close()
				return ErrClientConnTimeout
			}
			closeTransport = false
			time.Sleep(sleepTime)
			retries++
			grpclog.Printf("grpc: ClientConn.resetTransport failed to create client transport: %v; Reconnecting to %q", err, cc.target)
			continue
		}
		cc.mu.Lock()
		if cc.closing {
			// cc.Close() has been invoked.
			cc.mu.Unlock()
			newTransport.Close()
			return ErrClientConnClosing
		}
		cc.transport = newTransport
		cc.transportSeq = ts + 1
		if cc.ready != nil {
			close(cc.ready)
			cc.ready = nil
		}
		cc.mu.Unlock()
		return nil
	}
}

// Run in a goroutine to track the error in transport and create the
// new transport if an error happens. It returns when the channel is closing.
func (cc *ClientConn) transportMonitor() {
	for {
		select {
		// shutdownChan is needed to detect the channel teardown when
		// the ClientConn is idle (i.e., no RPC in flight).
		case <-cc.shutdownChan:
			return
		case <-cc.transport.Error():
			if err := cc.resetTransport(true); err != nil {
				// The channel is closing.
				grpclog.Printf("grpc: ClientConn.transportMonitor exits due to: %v", err)
				return
			}
			continue
		}
	}
}

// When wait returns, either the new transport is up or ClientConn is
// closing. Used to avoid working on a dying transport. It updates and
// returns the transport and its version when there is no error.
func (cc *ClientConn) wait(ctx context.Context, ts int) (transport.ClientTransport, int, error) {
	for {
		cc.mu.Lock()
		switch {
		case cc.closing:
			cc.mu.Unlock()
			return nil, 0, ErrClientConnClosing
		case ts < cc.transportSeq:
			// Worked on a dying transport. Try the new one immediately.
			defer cc.mu.Unlock()
			return cc.transport, cc.transportSeq, nil
		default:
			ready := cc.ready
			if ready == nil {
				ready = make(chan struct{})
				cc.ready = ready
			}
			cc.mu.Unlock()
			select {
			case <-ctx.Done():
				return nil, 0, transport.ContextErr(ctx.Err())
			// Wait until the new transport is ready or failed.
			case <-ready:
			}
		}
	}
}

// Close starts to tear down the ClientConn. Returns ErrClientConnClosing if
// it has been closed (mostly due to dial time-out).
// TODO(zhaoq): Make this synchronous to avoid unbounded memory consumption in
// some edge cases (e.g., the caller opens and closes many ClientConn's in a
// tight loop.
func (cc *ClientConn) Close() error {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if cc.closing {
		return ErrClientConnClosing
	}
	cc.closing = true
	if cc.ready != nil {
		close(cc.ready)
		cc.ready = nil
	}
	if cc.transport != nil {
		cc.transport.Close()
	}
	if cc.shutdownChan != nil {
		close(cc.shutdownChan)
	}
	return nil
}
