/*
Copyright 2026 The Kubernetes Authors.

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

package transport

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"sync"
	"sync/atomic"

	utilnet "k8s.io/apimachinery/pkg/util/net"
)

// PoolRoundTripper load-balances HTTP requests across multiple underlying connections
// to the same endpoint, preventing single-replica pinning when using L4 load balancers.
type PoolRoundTripper interface {
	http.RoundTripper
	utilnet.RoundTripperWrapper

	// Size returns the number of connections in the pool.
	Size() int

	// Inflight returns the current in-flight request count for connection at index.
	Inflight(index int) int64

	// Strategy returns the configured balancing strategy.
	Strategy() BalancingStrategy

	// Transports returns the underlying round trippers in the pool.
	Transports() []http.RoundTripper
}

type pooledConnection struct {
	rt       http.RoundTripper
	inflight atomic.Int64
	index    int
}

type poolRoundTripper struct {
	conns    []*pooledConnection
	strategy BalancingStrategy
	rrIndex  atomic.Uint64
	randFn   func(n int) int
}

var _ PoolRoundTripper = &poolRoundTripper{}

// NewPoolRoundTripper creates a new PoolRoundTripper from a slice of underlying round trippers.
func NewPoolRoundTripper(transports []http.RoundTripper, strategy BalancingStrategy) (PoolRoundTripper, error) {
	if len(transports) == 0 {
		return nil, errors.New("cannot create connection pool with zero transports")
	}

	if strategy == "" {
		strategy = PowerOfTwoChoices
	}
	if strategy != PowerOfTwoChoices && strategy != RoundRobin {
		return nil, fmt.Errorf("unsupported connection pool strategy %q", strategy)
	}

	conns := make([]*pooledConnection, len(transports))
	for i, rt := range transports {
		if rt == nil {
			return nil, fmt.Errorf("transport at index %d is nil", i)
		}
		conns[i] = &pooledConnection{
			rt:    rt,
			index: i,
		}
	}

	return &poolRoundTripper{
		conns:    conns,
		strategy: strategy,
		randFn:   rand.Intn,
	}, nil
}

// Size returns the number of pooled connections.
func (p *poolRoundTripper) Size() int {
	return len(p.conns)
}

// Inflight returns the in-flight count for connection at index.
func (p *poolRoundTripper) Inflight(index int) int64 {
	if index < 0 || index >= len(p.conns) {
		return 0
	}
	return p.conns[index].inflight.Load()
}

// Strategy returns the configured balancing strategy.
func (p *poolRoundTripper) Strategy() BalancingStrategy {
	return p.strategy
}

// Transports returns the underlying round trippers.
func (p *poolRoundTripper) Transports() []http.RoundTripper {
	res := make([]http.RoundTripper, len(p.conns))
	for i, c := range p.conns {
		res[i] = c.rt
	}
	return res
}

// WrappedRoundTripper returns the primary underlying round tripper for wrapper introspection.
func (p *poolRoundTripper) WrappedRoundTripper() http.RoundTripper {
	if len(p.conns) > 0 {
		return p.conns[0].rt
	}
	return nil
}

func (p *poolRoundTripper) pickConnection() *pooledConnection {
	n := len(p.conns)
	if n == 1 {
		return p.conns[0]
	}

	switch p.strategy {
	case RoundRobin:
		idx := (p.rrIndex.Add(1) - 1) % uint64(n)
		return p.conns[idx]

	case PowerOfTwoChoices:
		fallthrough
	default:
		if n == 2 {
			load0 := p.conns[0].inflight.Load()
			load1 := p.conns[1].inflight.Load()
			if load0 <= load1 {
				return p.conns[0]
			}
			return p.conns[1]
		}

		// Choose two distinct random indices
		i := p.randFn(n)
		j := p.randFn(n - 1)
		if j >= i {
			j++
		}

		loadI := p.conns[i].inflight.Load()
		loadJ := p.conns[j].inflight.Load()
		if loadI <= loadJ {
			return p.conns[i]
		}
		return p.conns[j]
	}
}

// RoundTrip executes a single HTTP transaction on one connection selected from the pool.
func (p *poolRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	conn := p.pickConnection()
	conn.inflight.Add(1)

	resp, err := conn.rt.RoundTrip(req)
	if err != nil {
		conn.inflight.Add(-1)
		return nil, err
	}

	if resp == nil || resp.Body == nil {
		conn.inflight.Add(-1)
		return resp, nil
	}

	tracked := &poolTrackedBody{
		ReadCloser: resp.Body,
		onClose: func() {
			conn.inflight.Add(-1)
		},
	}

	ctx := req.Context()
	if ctx != nil && ctx.Done() != nil {
		tracked.stop = context.AfterFunc(ctx, func() {
			tracked.closeOnce()
		})
	}

	resp.Body = tracked
	return resp, nil
}

type poolTrackedBody struct {
	io.ReadCloser
	onClose func()
	once    sync.Once
	stop    func() bool
}

func (b *poolTrackedBody) closeOnce() {
	b.once.Do(func() {
		if b.onClose != nil {
			b.onClose()
		}
	})
}

func (b *poolTrackedBody) Close() error {
	if b.stop != nil {
		b.stop()
	}
	b.closeOnce()
	if b.ReadCloser != nil {
		return b.ReadCloser.Close()
	}
	return nil
}
