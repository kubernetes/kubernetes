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
	"google.golang.org/grpc/codes"
)

// ErrNoAddrAvilable is returned by Get() when the balancer does not have
// any active connection to endpoints at the time.
// This error is returned only when opts.BlockingWait is true.
var ErrNoAddrAvilable = grpc.Errorf(codes.Unavailable, "there is no address available")

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

	// grpc issues TLS cert checks using the string passed into dial so
	// that string must be the host. To recover the full scheme://host URL,
	// have a map from hosts to the original endpoint.
	host2ep map[string]string

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
		host2ep:  getHost2ep(eps),
	}
	return sb
}

func (b *simpleBalancer) Start(target string, config grpc.BalancerConfig) error { return nil }

func (b *simpleBalancer) ConnectNotify() <-chan struct{} {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.upc
}

func (b *simpleBalancer) getEndpoint(host string) string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.host2ep[host]
}

func getHost2ep(eps []string) map[string]string {
	hm := make(map[string]string, len(eps))
	for i := range eps {
		_, host, _ := parseEndpoint(eps[i])
		hm[host] = eps[i]
	}
	return hm
}

func (b *simpleBalancer) updateAddrs(eps []string) {
	np := getHost2ep(eps)

	b.mu.Lock()
	defer b.mu.Unlock()

	match := len(np) == len(b.host2ep)
	for k, v := range np {
		if b.host2ep[k] != v {
			match = false
			break
		}
	}
	if match {
		// same endpoints, so no need to update address
		return
	}

	b.host2ep = np

	addrs := make([]grpc.Address, 0, len(eps))
	for i := range eps {
		addrs = append(addrs, grpc.Address{Addr: getHost(eps[i])})
	}
	b.addrs = addrs
	b.notifyCh <- addrs
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

	// If opts.BlockingWait is false (for fail-fast RPCs), it should return
	// an address it has notified via Notify immediately instead of blocking.
	if !opts.BlockingWait {
		b.mu.RLock()
		closed := b.closed
		addr = b.pinAddr
		upEps := len(b.upEps)
		b.mu.RUnlock()
		if closed {
			return grpc.Address{Addr: ""}, nil, grpc.ErrClientConnClosing
		}

		if upEps == 0 {
			return grpc.Address{Addr: ""}, nil, ErrNoAddrAvilable
		}
		return grpc.Address{Addr: addr}, func() {}, nil
	}

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
