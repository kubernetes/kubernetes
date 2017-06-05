// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package clientv3

import (
	"net/url"
	"strings"
	"sync"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

// simpleBalancer does the bare minimum to expose multiple eps
// to the grpc reconnection code path
type simpleBalancer struct {
	// addrs are the client's endpoints for grpc
	addrs []grpc.Address
	// notifyCh notifies grpc of the set of addresses for connecting
	notifyCh chan []grpc.Address

	// readyc closes once the first connection is up
	readyc    chan struct{}
	readyOnce sync.Once

	// mu protects upEps, pinAddr, and connectingAddr
	mu sync.RWMutex
	// upEps holds the current endpoints that have an active connection
	upEps map[string]struct{}
	// upc closes when upEps transitions from empty to non-zero or the balancer closes.
	upc chan struct{}

	// pinAddr is the currently pinned address; set to the empty string on
	// intialization and shutdown.
	pinAddr string

	closed bool
}

func newSimpleBalancer(eps []string) *simpleBalancer {
	notifyCh := make(chan []grpc.Address, 1)
	addrs := make([]grpc.Address, len(eps))
	for i := range eps {
		addrs[i].Addr = getHost(eps[i])
	}
	notifyCh <- addrs
	sb := &simpleBalancer{
		addrs:    addrs,
		notifyCh: notifyCh,
		readyc:   make(chan struct{}),
		upEps:    make(map[string]struct{}),
		upc:      make(chan struct{}),
	}
	return sb
}

func (b *simpleBalancer) Start(target string) error { return nil }

func (b *simpleBalancer) ConnectNotify() <-chan struct{} {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.upc
}

func (b *simpleBalancer) Up(addr grpc.Address) func(error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// gRPC might call Up after it called Close. We add this check
	// to "fix" it up at application layer. Or our simplerBalancer
	// might panic since b.upc is closed.
	if b.closed {
		return func(err error) {}
	}

	if len(b.upEps) == 0 {
		// notify waiting Get()s and pin first connected address
		close(b.upc)
		b.pinAddr = addr.Addr
	}
	b.upEps[addr.Addr] = struct{}{}

	// notify client that a connection is up
	b.readyOnce.Do(func() { close(b.readyc) })

	return func(err error) {
		b.mu.Lock()
		delete(b.upEps, addr.Addr)
		if len(b.upEps) == 0 && b.pinAddr != "" {
			b.upc = make(chan struct{})
		} else if b.pinAddr == addr.Addr {
			// choose new random up endpoint
			for k := range b.upEps {
				b.pinAddr = k
				break
			}
		}
		b.mu.Unlock()
	}
}

func (b *simpleBalancer) Get(ctx context.Context, opts grpc.BalancerGetOptions) (grpc.Address, func(), error) {
	var addr string
	for {
		b.mu.RLock()
		ch := b.upc
		b.mu.RUnlock()
		select {
		case <-ch:
		case <-ctx.Done():
			return grpc.Address{Addr: ""}, nil, ctx.Err()
		}
		b.mu.RLock()
		addr = b.pinAddr
		upEps := len(b.upEps)
		b.mu.RUnlock()
		if addr == "" {
			return grpc.Address{Addr: ""}, nil, grpc.ErrClientConnClosing
		}
		if upEps > 0 {
			break
		}
	}
	return grpc.Address{Addr: addr}, func() {}, nil
}

func (b *simpleBalancer) Notify() <-chan []grpc.Address { return b.notifyCh }

func (b *simpleBalancer) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	// In case gRPC calls close twice. TODO: remove the checking
	// when we are sure that gRPC wont call close twice.
	if b.closed {
		return nil
	}
	b.closed = true
	close(b.notifyCh)
	// terminate all waiting Get()s
	b.pinAddr = ""
	if len(b.upEps) == 0 {
		close(b.upc)
	}
	return nil
}

func getHost(ep string) string {
	url, uerr := url.Parse(ep)
	if uerr != nil || !strings.Contains(ep, "://") {
		return ep
	}
	return url.Host
}
