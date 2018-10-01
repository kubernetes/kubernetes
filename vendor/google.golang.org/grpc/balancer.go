/*
 *
 * Copyright 2016 gRPC authors.
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

package grpc

import (
	"fmt"
	"net"
	"sync"

	"golang.org/x/net/context"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/naming"
	"google.golang.org/grpc/status"
)

// Address represents a server the client connects to.
// This is the EXPERIMENTAL API and may be changed or extended in the future.
type Address struct {
	// Addr is the server address on which a connection will be established.
	Addr string
	// Metadata is the information associated with Addr, which may be used
	// to make load balancing decision.
	Metadata interface{}
}

// BalancerConfig specifies the configurations for Balancer.
type BalancerConfig struct {
	// DialCreds is the transport credential the Balancer implementation can
	// use to dial to a remote load balancer server. The Balancer implementations
	// can ignore this if it does not need to talk to another party securely.
	DialCreds credentials.TransportCredentials
	// Dialer is the custom dialer the Balancer implementation can use to dial
	// to a remote load balancer server. The Balancer implementations
	// can ignore this if it doesn't need to talk to remote balancer.
	Dialer func(context.Context, string) (net.Conn, error)
}

// BalancerGetOptions configures a Get call.
// This is the EXPERIMENTAL API and may be changed or extended in the future.
type BalancerGetOptions struct {
	// BlockingWait specifies whether Get should block when there is no
	// connected address.
	BlockingWait bool
}

// Balancer chooses network addresses for RPCs.
// This is the EXPERIMENTAL API and may be changed or extended in the future.
type Balancer interface {
	// Start does the initialization work to bootstrap a Balancer. For example,
	// this function may start the name resolution and watch the updates. It will
	// be called when dialing.
	Start(target string, config BalancerConfig) error
	// Up informs the Balancer that gRPC has a connection to the server at
	// addr. It returns down which is called once the connection to addr gets
	// lost or closed.
	// TODO: It is not clear how to construct and take advantage of the meaningful error
	// parameter for down. Need realistic demands to guide.
	Up(addr Address) (down func(error))
	// Get gets the address of a server for the RPC corresponding to ctx.
	// i) If it returns a connected address, gRPC internals issues the RPC on the
	// connection to this address;
	// ii) If it returns an address on which the connection is under construction
	// (initiated by Notify(...)) but not connected, gRPC internals
	//  * fails RPC if the RPC is fail-fast and connection is in the TransientFailure or
	//  Shutdown state;
	//  or
	//  * issues RPC on the connection otherwise.
	// iii) If it returns an address on which the connection does not exist, gRPC
	// internals treats it as an error and will fail the corresponding RPC.
	//
	// Therefore, the following is the recommended rule when writing a custom Balancer.
	// If opts.BlockingWait is true, it should return a connected address or
	// block if there is no connected address. It should respect the timeout or
	// cancellation of ctx when blocking. If opts.BlockingWait is false (for fail-fast
	// RPCs), it should return an address it has notified via Notify(...) immediately
	// instead of blocking.
	//
	// The function returns put which is called once the rpc has completed or failed.
	// put can collect and report RPC stats to a remote load balancer.
	//
	// This function should only return the errors Balancer cannot recover by itself.
	// gRPC internals will fail the RPC if an error is returned.
	Get(ctx context.Context, opts BalancerGetOptions) (addr Address, put func(), err error)
	// Notify returns a channel that is used by gRPC internals to watch the addresses
	// gRPC needs to connect. The addresses might be from a name resolver or remote
	// load balancer. gRPC internals will compare it with the existing connected
	// addresses. If the address Balancer notified is not in the existing connected
	// addresses, gRPC starts to connect the address. If an address in the existing
	// connected addresses is not in the notification list, the corresponding connection
	// is shutdown gracefully. Otherwise, there are no operations to take. Note that
	// the Address slice must be the full list of the Addresses which should be connected.
	// It is NOT delta.
	Notify() <-chan []Address
	// Close shuts down the balancer.
	Close() error
}

// downErr implements net.Error. It is constructed by gRPC internals and passed to the down
// call of Balancer.
type downErr struct {
	timeout   bool
	temporary bool
	desc      string
}

func (e downErr) Error() string   { return e.desc }
func (e downErr) Timeout() bool   { return e.timeout }
func (e downErr) Temporary() bool { return e.temporary }

func downErrorf(timeout, temporary bool, format string, a ...interface{}) downErr {
	return downErr{
		timeout:   timeout,
		temporary: temporary,
		desc:      fmt.Sprintf(format, a...),
	}
}

// RoundRobin returns a Balancer that selects addresses round-robin. It uses r to watch
// the name resolution updates and updates the addresses available correspondingly.
func RoundRobin(r naming.Resolver) Balancer {
	return &roundRobin{r: r}
}

type addrInfo struct {
	addr      Address
	connected bool
}

type roundRobin struct {
	r      naming.Resolver
	w      naming.Watcher
	addrs  []*addrInfo // all the addresses the client should potentially connect
	mu     sync.Mutex
	addrCh chan []Address // the channel to notify gRPC internals the list of addresses the client should connect to.
	next   int            // index of the next address to return for Get()
	waitCh chan struct{}  // the channel to block when there is no connected address available
	done   bool           // The Balancer is closed.
}

func (rr *roundRobin) watchAddrUpdates() error {
	updates, err := rr.w.Next()
	if err != nil {
		grpclog.Warningf("grpc: the naming watcher stops working due to %v.", err)
		return err
	}
	rr.mu.Lock()
	defer rr.mu.Unlock()
	for _, update := range updates {
		addr := Address{
			Addr:     update.Addr,
			Metadata: update.Metadata,
		}
		switch update.Op {
		case naming.Add:
			var exist bool
			for _, v := range rr.addrs {
				if addr == v.addr {
					exist = true
					grpclog.Infoln("grpc: The name resolver wanted to add an existing address: ", addr)
					break
				}
			}
			if exist {
				continue
			}
			rr.addrs = append(rr.addrs, &addrInfo{addr: addr})
		case naming.Delete:
			for i, v := range rr.addrs {
				if addr == v.addr {
					copy(rr.addrs[i:], rr.addrs[i+1:])
					rr.addrs = rr.addrs[:len(rr.addrs)-1]
					break
				}
			}
		default:
			grpclog.Errorln("Unknown update.Op ", update.Op)
		}
	}
	// Make a copy of rr.addrs and write it onto rr.addrCh so that gRPC internals gets notified.
	open := make([]Address, len(rr.addrs))
	for i, v := range rr.addrs {
		open[i] = v.addr
	}
	if rr.done {
		return ErrClientConnClosing
	}
	select {
	case <-rr.addrCh:
	default:
	}
	rr.addrCh <- open
	return nil
}

func (rr *roundRobin) Start(target string, config BalancerConfig) error {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	if rr.done {
		return ErrClientConnClosing
	}
	if rr.r == nil {
		// If there is no name resolver installed, it is not needed to
		// do name resolution. In this case, target is added into rr.addrs
		// as the only address available and rr.addrCh stays nil.
		rr.addrs = append(rr.addrs, &addrInfo{addr: Address{Addr: target}})
		return nil
	}
	w, err := rr.r.Resolve(target)
	if err != nil {
		return err
	}
	rr.w = w
	rr.addrCh = make(chan []Address, 1)
	go func() {
		for {
			if err := rr.watchAddrUpdates(); err != nil {
				return
			}
		}
	}()
	return nil
}

// Up sets the connected state of addr and sends notification if there are pending
// Get() calls.
func (rr *roundRobin) Up(addr Address) func(error) {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	var cnt int
	for _, a := range rr.addrs {
		if a.addr == addr {
			if a.connected {
				return nil
			}
			a.connected = true
		}
		if a.connected {
			cnt++
		}
	}
	// addr is only one which is connected. Notify the Get() callers who are blocking.
	if cnt == 1 && rr.waitCh != nil {
		close(rr.waitCh)
		rr.waitCh = nil
	}
	return func(err error) {
		rr.down(addr, err)
	}
}

// down unsets the connected state of addr.
func (rr *roundRobin) down(addr Address, err error) {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	for _, a := range rr.addrs {
		if addr == a.addr {
			a.connected = false
			break
		}
	}
}

// Get returns the next addr in the rotation.
func (rr *roundRobin) Get(ctx context.Context, opts BalancerGetOptions) (addr Address, put func(), err error) {
	var ch chan struct{}
	rr.mu.Lock()
	if rr.done {
		rr.mu.Unlock()
		err = ErrClientConnClosing
		return
	}

	if len(rr.addrs) > 0 {
		if rr.next >= len(rr.addrs) {
			rr.next = 0
		}
		next := rr.next
		for {
			a := rr.addrs[next]
			next = (next + 1) % len(rr.addrs)
			if a.connected {
				addr = a.addr
				rr.next = next
				rr.mu.Unlock()
				return
			}
			if next == rr.next {
				// Has iterated all the possible address but none is connected.
				break
			}
		}
	}
	if !opts.BlockingWait {
		if len(rr.addrs) == 0 {
			rr.mu.Unlock()
			err = status.Errorf(codes.Unavailable, "there is no address available")
			return
		}
		// Returns the next addr on rr.addrs for failfast RPCs.
		addr = rr.addrs[rr.next].addr
		rr.next++
		rr.mu.Unlock()
		return
	}
	// Wait on rr.waitCh for non-failfast RPCs.
	if rr.waitCh == nil {
		ch = make(chan struct{})
		rr.waitCh = ch
	} else {
		ch = rr.waitCh
	}
	rr.mu.Unlock()
	for {
		select {
		case <-ctx.Done():
			err = ctx.Err()
			return
		case <-ch:
			rr.mu.Lock()
			if rr.done {
				rr.mu.Unlock()
				err = ErrClientConnClosing
				return
			}

			if len(rr.addrs) > 0 {
				if rr.next >= len(rr.addrs) {
					rr.next = 0
				}
				next := rr.next
				for {
					a := rr.addrs[next]
					next = (next + 1) % len(rr.addrs)
					if a.connected {
						addr = a.addr
						rr.next = next
						rr.mu.Unlock()
						return
					}
					if next == rr.next {
						// Has iterated all the possible address but none is connected.
						break
					}
				}
			}
			// The newly added addr got removed by Down() again.
			if rr.waitCh == nil {
				ch = make(chan struct{})
				rr.waitCh = ch
			} else {
				ch = rr.waitCh
			}
			rr.mu.Unlock()
		}
	}
}

func (rr *roundRobin) Notify() <-chan []Address {
	return rr.addrCh
}

func (rr *roundRobin) Close() error {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	if rr.done {
		return errBalancerClosed
	}
	rr.done = true
	if rr.w != nil {
		rr.w.Close()
	}
	if rr.waitCh != nil {
		close(rr.waitCh)
		rr.waitCh = nil
	}
	if rr.addrCh != nil {
		close(rr.addrCh)
	}
	return nil
}

// pickFirst is used to test multi-addresses in one addrConn in which all addresses share the same addrConn.
// It is a wrapper around roundRobin balancer. The logic of all methods works fine because balancer.Get()
// returns the only address Up by resetTransport().
type pickFirst struct {
	*roundRobin
}

func pickFirstBalancerV1(r naming.Resolver) Balancer {
	return &pickFirst{&roundRobin{r: r}}
}
