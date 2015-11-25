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
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/transport"
)

var (
	// ErrUnspecTarget indicates that the target address is unspecified.
	ErrUnspecTarget = errors.New("grpc: target is unspecified")
	// ErrNoTransportSecurity indicates that there is no transport security
	// being set for ClientConn. Users should either set one or explicityly
	// call WithInsecure DialOption to disable security.
	ErrNoTransportSecurity = errors.New("grpc: no transport security set (use grpc.WithInsecure() explicitly or set credentials)")
	// ErrCredentialsMisuse indicates that users want to transmit security infomation
	// (e.g., oauth2 token) which requires secure connection on an insecure
	// connection.
	ErrCredentialsMisuse = errors.New("grpc: the credentials require transport level security (use grpc.WithTransportAuthenticator() to set)")
	// ErrClientConnClosing indicates that the operation is illegal because
	// the session is closing.
	ErrClientConnClosing = errors.New("grpc: the client connection is closing")
	// ErrClientConnTimeout indicates that the connection could not be
	// established or re-established within the specified timeout.
	ErrClientConnTimeout = errors.New("grpc: timed out trying to connect")
	// minimum time to give a connection to complete
	minConnectTimeout = 20 * time.Second
)

// dialOptions configure a Dial call. dialOptions are set by the DialOption
// values passed to Dial.
type dialOptions struct {
	codec    Codec
	block    bool
	insecure bool
	copts    transport.ConnectOptions
}

// DialOption configures how we set up the connection.
type DialOption func(*dialOptions)

// WithCodec returns a DialOption which sets a codec for message marshaling and unmarshaling.
func WithCodec(c Codec) DialOption {
	return func(o *dialOptions) {
		o.codec = c
	}
}

// WithBlock returns a DialOption which makes caller of Dial blocks until the underlying
// connection is up. Without this, Dial returns immediately and connecting the server
// happens in background.
func WithBlock() DialOption {
	return func(o *dialOptions) {
		o.block = true
	}
}

func WithInsecure() DialOption {
	return func(o *dialOptions) {
		o.insecure = true
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

// WithUserAgent returns a DialOption that specifies a user agent string for all the RPCs.
func WithUserAgent(s string) DialOption {
	return func(o *dialOptions) {
		o.copts.UserAgent = s
	}
}

// Dial creates a client connection the given target.
func Dial(target string, opts ...DialOption) (*ClientConn, error) {
	if target == "" {
		return nil, ErrUnspecTarget
	}
	cc := &ClientConn{
		target:       target,
		shutdownChan: make(chan struct{}),
	}
	for _, opt := range opts {
		opt(&cc.dopts)
	}
	if !cc.dopts.insecure {
		var ok bool
		for _, c := range cc.dopts.copts.AuthOptions {
			if _, ok := c.(credentials.TransportAuthenticator); !ok {
				continue
			}
			ok = true
		}
		if !ok {
			return nil, ErrNoTransportSecurity
		}
	} else {
		for _, c := range cc.dopts.copts.AuthOptions {
			if c.RequireTransportSecurity() {
				return nil, ErrCredentialsMisuse
			}
		}
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
	cc.stateCV = sync.NewCond(&cc.mu)
	if cc.dopts.block {
		if err := cc.resetTransport(false); err != nil {
			cc.Close()
			return nil, err
		}
		// Start to monitor the error status of transport.
		go cc.transportMonitor()
	} else {
		// Start a goroutine connecting to the server asynchronously.
		go func() {
			if err := cc.resetTransport(false); err != nil {
				grpclog.Printf("Failed to dial %s: %v; please retry.", target, err)
				cc.Close()
				return
			}
			go cc.transportMonitor()
		}()
	}
	return cc, nil
}

// ConnectivityState indicates the state of a client connection.
type ConnectivityState int

const (
	// Idle indicates the ClientConn is idle.
	Idle ConnectivityState = iota
	// Connecting indicates the ClienConn is connecting.
	Connecting
	// Ready indicates the ClientConn is ready for work.
	Ready
	// TransientFailure indicates the ClientConn has seen a failure but expects to recover.
	TransientFailure
	// Shutdown indicates the ClientConn has stated shutting down.
	Shutdown
)

func (s ConnectivityState) String() string {
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
		panic(fmt.Sprintf("unknown connectivity state: %d", s))
	}
}

// ClientConn represents a client connection to an RPC service.
type ClientConn struct {
	target       string
	authority    string
	dopts        dialOptions
	shutdownChan chan struct{}

	mu      sync.Mutex
	state   ConnectivityState
	stateCV *sync.Cond
	// ready is closed and becomes nil when a new transport is up or failed
	// due to timeout.
	ready chan struct{}
	// Every time a new transport is created, this is incremented by 1. Used
	// to avoid trying to recreate a transport while the new one is already
	// under construction.
	transportSeq int
	transport    transport.ClientTransport
}

// State returns the connectivity state of the ClientConn
func (cc *ClientConn) State() ConnectivityState {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.state
}

// WaitForStateChange blocks until the state changes to something other than the sourceState
// or timeout fires. It returns false if timeout fires and true otherwise.
func (cc *ClientConn) WaitForStateChange(timeout time.Duration, sourceState ConnectivityState) bool {
	start := time.Now()
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if sourceState != cc.state {
		return true
	}
	expired := timeout <= time.Since(start)
	if expired {
		return false
	}
	done := make(chan struct{})
	go func() {
		select {
		case <-time.After(timeout - time.Since(start)):
			cc.mu.Lock()
			expired = true
			cc.stateCV.Broadcast()
			cc.mu.Unlock()
		case <-done:
		}
	}()
	defer close(done)
	for sourceState == cc.state {
		cc.stateCV.Wait()
		if expired {
			return false
		}
	}
	return true
}

func (cc *ClientConn) resetTransport(closeTransport bool) error {
	var retries int
	start := time.Now()
	for {
		cc.mu.Lock()
		cc.state = Connecting
		cc.stateCV.Broadcast()
		t := cc.transport
		ts := cc.transportSeq
		// Avoid wait() picking up a dying transport unnecessarily.
		cc.transportSeq = 0
		if cc.state == Shutdown {
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
		sleepTime := backoff(retries)
		timeout := sleepTime
		if timeout < minConnectTimeout {
			timeout = minConnectTimeout
		}
		if copts.Timeout == 0 || copts.Timeout > timeout {
			copts.Timeout = timeout
		}
		connectTime := time.Now()
		newTransport, err := transport.NewClientTransport(cc.target, &copts)
		if err != nil {
			cc.mu.Lock()
			cc.state = TransientFailure
			cc.stateCV.Broadcast()
			cc.mu.Unlock()
			sleepTime -= time.Since(connectTime)
			if sleepTime < 0 {
				sleepTime = 0
			}
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
		if cc.state == Shutdown {
			// cc.Close() has been invoked.
			cc.mu.Unlock()
			newTransport.Close()
			return ErrClientConnClosing
		}
		cc.state = Ready
		cc.stateCV.Broadcast()
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
		// shutdownChan is needed to detect the teardown when
		// the ClientConn is idle (i.e., no RPC in flight).
		case <-cc.shutdownChan:
			return
		case <-cc.transport.Error():
			cc.mu.Lock()
			cc.state = TransientFailure
			cc.stateCV.Broadcast()
			cc.mu.Unlock()
			if err := cc.resetTransport(true); err != nil {
				// The ClientConn is closing.
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
		case cc.state == Shutdown:
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
	if cc.state == Shutdown {
		return ErrClientConnClosing
	}
	cc.state = Shutdown
	cc.stateCV.Broadcast()
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
