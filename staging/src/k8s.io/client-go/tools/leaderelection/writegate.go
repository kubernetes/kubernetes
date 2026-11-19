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

package leaderelection

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync"

	"k8s.io/client-go/transport"
	"k8s.io/client-go/util/workqueue"
)

// ErrWriteGateClosed is returned by a transport wrapped with
// LeaderElector.WriteGate for any request that is not a safe HTTP method
// while the elector is not leading, and is the cancellation cause of writes
// that were in flight when it stopped leading. Use errors.Is to tell such a
// rejection apart from an API server failure.
var ErrWriteGateClosed = errors.New("write gate is closed: not leading")

// Pausable is a component, such as a workqueue or a controller, that can pause
// and resume work. It is the workqueue package's Pausable; see RegisterPausable
// for what the elector expects of it.
type Pausable = workqueue.Pausable

// RegisterPausable registers p to be paused when the write gate closes and
// resumed when it opens. If the gate is closed when p is registered, p.Pause()
// is called before RegisterPausable returns.
//
// Pause and Resume are called synchronously from the election loop, so they
// must return promptly and must not call back into the LeaderElector. Pause
// must actually stop the component from acting. Mutual exclusion rests on the
// write gate, which rejects writes through wrapped clients whether or not a
// component pauses. Pausing spares the component the rejected requests and
// any work the gate does not cover.
//
// Registrations last for the lifetime of the LeaderElector.
func (le *LeaderElector) RegisterPausable(p Pausable) {
	if p == nil {
		return
	}
	le.writeGate.registerPausable(p)
}

// writeGate blocks write requests while it is closed.
//
// The gate is an allowlist of HTTP methods: only the safe methods of RFC 9110
// (GET, HEAD, OPTIONS, TRACE) pass a closed gate. Anything else, including
// methods the gate has never heard of, is treated as a write.
type writeGate struct {
	mu sync.Mutex
	// ctx is live while the gate is open and cancelled when it closes, so
	// that writes that are in flight at that point are cancelled.
	ctx    context.Context
	cancel context.CancelFunc

	// transitionMu serializes state transitions together with the Pause and
	// Resume callbacks they trigger, so that a registration racing with a
	// transition cannot leave a registrant in the wrong state. It is acquired
	// before mu and is never held by RoundTrip.
	transitionMu sync.Mutex
	// pausables is protected by transitionMu.
	pausables []Pausable
}

// open lets writes through and resumes registered pausables. Opening an open
// gate is a no-op.
func (g *writeGate) open() {
	g.transitionMu.Lock()
	defer g.transitionMu.Unlock()

	g.mu.Lock()
	if g.ctx != nil {
		g.mu.Unlock()
		return
	}
	g.ctx, g.cancel = context.WithCancel(context.Background())
	g.mu.Unlock()

	for _, p := range g.pausables {
		p.Resume()
	}
}

// close rejects new writes, cancels in-flight ones and pauses registered
// pausables. Closing a closed gate is a no-op.
func (g *writeGate) close() {
	g.transitionMu.Lock()
	defer g.transitionMu.Unlock()

	g.mu.Lock()
	if g.ctx == nil {
		g.mu.Unlock()
		return
	}
	// Cancel first so that in-flight writes are cut off before any registrant
	// gets a chance to slow this down.
	g.cancel()
	g.ctx, g.cancel = nil, nil
	g.mu.Unlock()

	for _, p := range g.pausables {
		p.Pause()
	}
}

// registerPausable adds p and brings it in line with the current gate state.
func (g *writeGate) registerPausable(p Pausable) {
	g.transitionMu.Lock()
	defer g.transitionMu.Unlock()

	g.pausables = append(g.pausables, p)
	if g.openContext() == nil {
		p.Pause()
	}
}

// openContext returns a context that is cancelled when the gate closes.
func (g *writeGate) openContext() context.Context {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.ctx
}

// wrapper returns a transport wrapper that applies the gate.
func (g *writeGate) wrapper() transport.WrapperFunc {
	return func(rt http.RoundTripper) http.RoundTripper {
		return &writeGateRoundTripper{gate: g, rt: rt}
	}
}

func isSafeMethod(method string) bool {
	switch method {
	case http.MethodGet, http.MethodHead, http.MethodOptions, http.MethodTrace:
		return true
	}
	return false
}

type writeGateRoundTripper struct {
	gate *writeGate
	rt   http.RoundTripper
}

var _ http.RoundTripper = &writeGateRoundTripper{}

func (t *writeGateRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if isSafeMethod(req.Method) {
		return t.rt.RoundTrip(req)
	}

	gateCtx := t.gate.openContext()
	if gateCtx == nil {
		// A RoundTripper must close the request body even when it fails.
		if req.Body != nil {
			_ = req.Body.Close()
		}
		return nil, ErrWriteGateClosed
	}

	// Cancel the request if the gate closes while it is in flight.
	ctx, cancel := context.WithCancelCause(req.Context())
	stop := context.AfterFunc(gateCtx, func() { cancel(ErrWriteGateClosed) })
	resp, err := t.rt.RoundTrip(req.WithContext(ctx))
	stop()
	if err != nil {
		cancel(nil)
		if errors.Is(context.Cause(ctx), ErrWriteGateClosed) {
			return nil, ErrWriteGateClosed
		}
		return nil, err
	}
	if resp.Body == nil {
		cancel(nil)
		return resp, nil
	}
	resp.Body = &cancelOnCloseBody{ReadCloser: resp.Body, cancel: func() { cancel(nil) }}
	return resp, nil
}

// WrappedRoundTripper implements k8s.io/apimachinery/pkg/util/net.RoundTripperWrapper.
func (t *writeGateRoundTripper) WrappedRoundTripper() http.RoundTripper {
	return t.rt
}

// cancelOnCloseBody releases the per-request context when the body is closed.
type cancelOnCloseBody struct {
	io.ReadCloser
	cancel func()
}

func (b *cancelOnCloseBody) Close() error {
	defer b.cancel()
	return b.ReadCloser.Close()
}
