/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package net

import (
	"context"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
)

// connErrPair pairs conn and error which is returned by accept on sub-listeners.
type connErrPair struct {
	conn net.Conn
	err  error
}

// multiListener implements net.Listener
type multiListener struct {
	listeners []net.Listener
	wg        sync.WaitGroup

	// connCh passes accepted connections, from child listeners to parent.
	connCh chan connErrPair
	// stopCh communicates from parent to child listeners.
	stopCh chan struct{}
	closed atomic.Bool
}

// compile time check to ensure *multiListener implements net.Listener
var _ net.Listener = &multiListener{}

// MultiListen returns net.Listener which can listen on and accept connections for
// the given network on multiple addresses. Internally it uses stdlib to create
// sub-listener and multiplexes connection requests using go-routines.
// The network must be "tcp", "tcp4" or "tcp6".
// It follows the semantics of net.Listen that primarily means:
//  1. If the host is an unspecified/zero IP address with "tcp" network, MultiListen
//     listens on all available unicast and anycast IP addresses of the local system.
//  2. Use "tcp4" or "tcp6" to exclusively listen on IPv4 or IPv6 family, respectively.
//  3. The host can accept names (e.g, localhost) and it will create a listener for at
//     most one of the host's IP.
func MultiListen(ctx context.Context, network string, addrs ...string) (net.Listener, error) {
	var lc net.ListenConfig
	return multiListen(
		ctx,
		network,
		addrs,
		func(ctx context.Context, network, address string) (net.Listener, error) {
			return lc.Listen(ctx, network, address)
		})
}

// multiListen implements MultiListen by consuming stdlib functions as dependency allowing
// mocking for unit-testing.
func multiListen(
	ctx context.Context,
	network string,
	addrs []string,
	listenFunc func(ctx context.Context, network, address string) (net.Listener, error),
) (net.Listener, error) {
	if !(network == "tcp" || network == "tcp4" || network == "tcp6") {
		return nil, fmt.Errorf("network %q not supported", network)
	}
	if len(addrs) == 0 {
		return nil, fmt.Errorf("no address provided to listen on")
	}

	ml := &multiListener{
		connCh: make(chan connErrPair),
		stopCh: make(chan struct{}),
	}
	for _, addr := range addrs {
		l, err := listenFunc(ctx, network, addr)
		if err != nil {
			// close all the sub-listeners and exit
			_ = ml.Close()
			return nil, err
		}
		ml.listeners = append(ml.listeners, l)
	}

	for _, l := range ml.listeners {
		ml.wg.Add(1)
		go func(l net.Listener) {
			defer ml.wg.Done()
			for {
				// Accept() is blocking, unless ml.Close() is called, in which
				// case it will return immediately with an error.
				conn, err := l.Accept()
				// This assumes that ANY error from Accept() will terminate the
				// sub-listener. We could maybe be more precise, but it
				// doesn't seem necessary.
				terminate := err != nil

				select {
				case ml.connCh <- connErrPair{conn: conn, err: err}:
				case <-ml.stopCh:
					// In case we accepted a connection AND were stopped, and
					// this select-case was chosen, just throw away the
					// connection.  This avoids potentially blocking on connCh
					// or leaking a connection.
					if conn != nil {
						_ = conn.Close()
					}
					terminate = true
				}
				// Make sure we don't loop on Accept() returning an error and
				// the select choosing the channel case.
				if terminate {
					return
				}
			}
		}(l)
	}
	return ml, nil
}

// Accept implements net.Listener. It waits for and returns a connection from
// any of the sub-listener.
func (ml *multiListener) Accept() (net.Conn, error) {
	// wait for any sub-listener to enqueue an accepted connection
	connErr, ok := <-ml.connCh
	if !ok {
		// The channel will be closed only when Close() is called on the
		// multiListener. Closing of this channel implies that all
		// sub-listeners are also closed, which causes a "use of closed
		// network connection" error on their Accept() calls. We return the
		// same error for multiListener.Accept() if multiListener.Close()
		// has already been called.
		return nil, fmt.Errorf("use of closed network connection")
	}
	return connErr.conn, connErr.err
}

// Close implements net.Listener. It will close all sub-listeners and wait for
// the go-routines to exit.
func (ml *multiListener) Close() error {
	// Make sure this can be called repeatedly without explosions.
	if !ml.closed.CompareAndSwap(false, true) {
		return fmt.Errorf("use of closed network connection")
	}

	// Tell all sub-listeners to stop.
	close(ml.stopCh)

	// Closing the listeners causes Accept() to immediately return an error in
	// the sub-listener go-routines.
	for _, l := range ml.listeners {
		_ = l.Close()
	}

	// Wait for all the sub-listener go-routines to exit.
	ml.wg.Wait()
	close(ml.connCh)

	// Drain any already-queued connections.
	for connErr := range ml.connCh {
		if connErr.conn != nil {
			_ = connErr.conn.Close()
		}
	}
	return nil
}

// Addr is an implementation of the net.Listener interface.  It always returns
// the address of the first listener.  Callers should  use conn.LocalAddr() to
// obtain the actual local address of the sub-listener.
func (ml *multiListener) Addr() net.Addr {
	return ml.listeners[0].Addr()
}

// Addrs is like Addr, but returns the address for all registered listeners.
func (ml *multiListener) Addrs() []net.Addr {
	var ret []net.Addr
	for _, l := range ml.listeners {
		ret = append(ret, l.Addr())
	}
	return ret
}
