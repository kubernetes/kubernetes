/*
Copyright The Kubernetes Authors.

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
	"io"
	"math/rand/v2"
	"net/http"
	"sync"
	"sync/atomic"

	utilnet "k8s.io/apimachinery/pkg/util/net"
)

// connectionPool spreads requests across a fixed set of independent
// RoundTrippers ("members"), each of which keeps its own connections.
//
// For every request it picks two members at random and uses the one with
// fewer requests in flight ("power of two choices"). A request is in flight
// until its response body is closed or fully read, so long-running requests
// such as watches are taken into account.
type connectionPool struct {
	members []*poolMember
	// intn returns a random number in [0, n). It is replaced in tests.
	intn func(n int) int
}

type poolMember struct {
	rt       http.RoundTripper
	inFlight atomic.Int64
}

var _ http.RoundTripper = &connectionPool{}
var _ utilnet.RoundTripperWrapper = &connectionPool{}

// newConnectionPool returns a pool of the given RoundTrippers. rts must not be empty.
func newConnectionPool(rts []http.RoundTripper) *connectionPool {
	members := make([]*poolMember, len(rts))
	for i, rt := range rts {
		members[i] = &poolMember{rt: rt}
	}
	return &connectionPool{members: members, intn: rand.IntN}
}

// pick returns the less loaded of two distinct, randomly chosen members.
func (p *connectionPool) pick() *poolMember {
	n := len(p.members)
	if n == 1 {
		return p.members[0]
	}
	i := p.intn(n)
	j := p.intn(n - 1)
	if j >= i {
		j++
	}
	a, b := p.members[i], p.members[j]
	if b.inFlight.Load() < a.inFlight.Load() {
		return b
	}
	return a
}

func (p *connectionPool) RoundTrip(req *http.Request) (*http.Response, error) {
	m := p.pick()
	m.inFlight.Add(1)
	resp, err := m.rt.RoundTrip(req)
	if err != nil || resp == nil || resp.Body == nil {
		m.inFlight.Add(-1)
		return resp, err
	}
	resp.Body = newInFlightBody(resp.Body, func() { m.inFlight.Add(-1) })
	return resp, nil
}

func (p *connectionPool) CloseIdleConnections() {
	for _, m := range p.members {
		utilnet.CloseIdleConnectionsFor(m.rt)
	}
}

func (p *connectionPool) CancelRequest(req *http.Request) {
	for _, m := range p.members {
		tryCancelRequest(m.rt, req)
	}
}

// WrappedRoundTripper returns the first member. All members are configured
// the same way, so it can be used to inspect e.g. the TLS config.
func (p *connectionPool) WrappedRoundTripper() http.RoundTripper {
	return p.members[0].rt
}

// inFlightBody calls done once the body is closed or fully read.
type inFlightBody struct {
	io.ReadCloser
	once sync.Once
	done func()
}

func (b *inFlightBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err == io.EOF {
		b.once.Do(b.done)
	}
	return n, err
}

func (b *inFlightBody) Close() error {
	err := b.ReadCloser.Close()
	b.once.Do(b.done)
	return err
}

// inFlightReadWriteCloser keeps the body of "101 Switching Protocols"
// responses writable.
type inFlightReadWriteCloser struct {
	*inFlightBody
	io.Writer
}

func newInFlightBody(body io.ReadCloser, done func()) io.ReadCloser {
	b := &inFlightBody{ReadCloser: body, done: done}
	if w, ok := body.(io.Writer); ok {
		return &inFlightReadWriteCloser{inFlightBody: b, Writer: w}
	}
	return b
}
