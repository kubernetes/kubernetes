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
	"crypto/x509"
	"errors"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"sync"
	"sync/atomic"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	clientgofeaturegate "k8s.io/client-go/features"
)

// connectionPool spreads requests across a fixed set of independent
// RoundTrippers ("members"), each of which keeps its own connections.
//
// For every request it picks two members at random and uses the one with
// fewer requests in flight ("power of two choices"). A request is in flight
// until its response body is closed or fully read, so long-running requests
// such as watches are taken into account.
//
// If retryConnectionFailures is set and the picked member fails to connect to
// the server, the request is sent once more through the backup member. The
// request wasn't sent to the server, so this is safe for any request whose
// body can be read again.
type connectionPool struct {
	members                 []*poolMember
	retryConnectionFailures bool
	// intn returns a random number in [0, n).
	intn func(n int) int
}

type poolMember struct {
	rt       rotatableCARoundTripper
	inFlight atomic.Int64
}

var _ rotatableCARoundTripper = &connectionPool{}
var _ utilnet.RoundTripperWrapper = &connectionPool{}

// newConnectionPool returns a pool of the given RoundTrippers. rts must not be empty.
func newConnectionPool(rts []rotatableCARoundTripper) *connectionPool {
	members := make([]*poolMember, len(rts))
	for i, rt := range rts {
		members[i] = &poolMember{rt: rt}
	}
	return &connectionPool{
		members:                 members,
		retryConnectionFailures: clientgofeaturegate.FeatureGates().Enabled(clientgofeaturegate.ClientsAllowConnectionPoolFailureAvoidance),
		intn:                    rand.IntN,
	}
}

// pick chooses two distinct members at random and returns the less loaded
// one, and the other one as a backup. With a single member, backup is nil.
func (p *connectionPool) pick() (picked, backup *poolMember) {
	n := len(p.members)
	if n == 1 {
		return p.members[0], nil
	}
	i := p.intn(n)
	// Pick j uniformly from the other n-1 members: draw from [0, n-1) and
	// skip over i.
	j := p.intn(n - 1)
	if j >= i {
		j++
	}
	a, b := p.members[i], p.members[j]
	if b.inFlight.Load() < a.inFlight.Load() {
		return b, a
	}
	return a, b
}

func (p *connectionPool) RoundTrip(req *http.Request) (*http.Response, error) {
	m, backup := p.pick()
	resp, err := p.roundTrip(m, req)
	if err != nil && p.retryConnectionFailures && backup != nil && isConnectionFailure(err) && req.Context().Err() == nil {
		if retry, ok := rewindRequest(req); ok {
			resp, err = p.roundTrip(backup, retry)
		}
	}
	return resp, err
}

func (p *connectionPool) roundTrip(m *poolMember, req *http.Request) (*http.Response, error) {
	m.inFlight.Add(1)
	resp, err := m.rt.RoundTrip(req)
	if err != nil || resp == nil || resp.Body == nil {
		m.inFlight.Add(-1)
		return resp, err
	}
	resp.Body = newInFlightBody(resp.Body, func() { m.inFlight.Add(-1) })
	return resp, nil
}

// isConnectionFailure reports whether err is a failure to dial the server, such
// as a refused connection, a failed DNS lookup or a connect timeout.
func isConnectionFailure(err error) bool {
	var opErr *net.OpError
	return errors.As(err, &opErr) && opErr.Op == "dial"
}

// rewindRequest returns req with a fresh body, so that it can be sent again,
// or false if the body can't be read again.
func rewindRequest(req *http.Request) (*http.Request, bool) {
	if req.Body == nil || req.Body == http.NoBody {
		return req, true
	}
	if req.GetBody == nil {
		return nil, false
	}
	// The first attempt closed req.Body. GetBody returns a new copy of the
	// whole body, e.g. for the byte readers client-go uses.
	body, err := req.GetBody()
	if err != nil {
		return nil, false
	}
	// Clone shares Body with req, so replace it. Cloning keeps req unmodified.
	retry := req.Clone(req.Context())
	retry.Body = body
	return retry, true
}

func (p *connectionPool) CloseIdleConnections() {
	for _, m := range p.members {
		m.rt.CloseIdleConnections()
	}
}

// Clone returns a new pool with clones of all members. The new pool starts
// with no requests in flight.
func (p *connectionPool) Clone() rotatableCARoundTripper {
	rts := make([]rotatableCARoundTripper, len(p.members))
	for i, m := range p.members {
		rts[i] = m.rt.Clone()
	}
	clone := newConnectionPool(rts)
	clone.retryConnectionFailures = p.retryConnectionFailures
	clone.intn = p.intn
	return clone
}

// GetRootCAs returns the root CAs of the members, which all use the same ones.
func (p *connectionPool) GetRootCAs() *x509.CertPool {
	return p.members[0].rt.GetRootCAs()
}

// SetRootCAs sets the root CAs of all members.
func (p *connectionPool) SetRootCAs(pool *x509.CertPool) {
	for _, m := range p.members {
		m.rt.SetRootCAs(pool)
	}
}

func (p *connectionPool) CancelRequest(req *http.Request) {
	for _, m := range p.members {
		tryCancelRequest(m.rt, req)
	}
}

// WrappedRoundTripper returns the first member. All members are configured
// the same way, including their root CAs, which are only ever set for all
// members at once, so it can be used to inspect e.g. the TLS config.
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
