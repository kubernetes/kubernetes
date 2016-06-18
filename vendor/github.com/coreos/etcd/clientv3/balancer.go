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
	"sync/atomic"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

// simpleBalancer does the bare minimum to expose multiple eps
// to the grpc reconnection code path
type simpleBalancer struct {
	// eps are the client's endpoints stripped of any URL scheme
	eps     []string
	ch      chan []grpc.Address
	numGets uint32
}

func newSimpleBalancer(eps []string) grpc.Balancer {
	ch := make(chan []grpc.Address, 1)
	addrs := make([]grpc.Address, len(eps))
	for i := range eps {
		addrs[i].Addr = getHost(eps[i])
	}
	ch <- addrs
	return &simpleBalancer{eps: eps, ch: ch}
}

func (b *simpleBalancer) Start(target string) error        { return nil }
func (b *simpleBalancer) Up(addr grpc.Address) func(error) { return func(error) {} }
func (b *simpleBalancer) Get(ctx context.Context, opts grpc.BalancerGetOptions) (grpc.Address, func(), error) {
	v := atomic.AddUint32(&b.numGets, 1)
	ep := b.eps[v%uint32(len(b.eps))]
	return grpc.Address{Addr: getHost(ep)}, func() {}, nil
}
func (b *simpleBalancer) Notify() <-chan []grpc.Address { return b.ch }
func (b *simpleBalancer) Close() error {
	close(b.ch)
	return nil
}

func getHost(ep string) string {
	url, uerr := url.Parse(ep)
	if uerr != nil || !strings.Contains(ep, "://") {
		return ep
	}
	return url.Host
}
