/*
Copyright 2015 The Kubernetes Authors.

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

// Package chaosclient makes it easy to simulate network latency, misbehaving
// servers, and random errors from servers. It is intended to stress test components
// under failure conditions and expose weaknesses in the error handling logic
// of the codebase.
package chaosclient

import (
	"errors"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"reflect"
	"runtime"

	"k8s.io/apimachinery/pkg/util/net"
)

// chaosrt provides the ability to perform simulations of HTTP client failures
// under the Golang http.Transport interface.
type chaosrt struct {
	rt     http.RoundTripper
	notify ChaosNotifier
	c      []Chaos
}

// Chaos intercepts requests to a remote HTTP endpoint and can inject arbitrary
// failures.
type Chaos interface {
	// Intercept should return true if the normal flow should be skipped, and the
	// return response and error used instead. Modifications to the request will
	// be ignored, but may be used to make decisions about types of failures.
	Intercept(req *http.Request) (bool, *http.Response, error)
}

// ChaosNotifier notifies another component that the ChaosRoundTripper has simulated
// a failure.
type ChaosNotifier interface {
	// OnChaos is invoked when a chaotic outcome was triggered. fn is the
	// source of Chaos and req was the outgoing request
	OnChaos(req *http.Request, c Chaos)
}

// ChaosFunc takes an http.Request and decides whether to alter the response. It
// returns true if it wishes to mutate the response, with a http.Response or
// error.
type ChaosFunc func(req *http.Request) (bool, *http.Response, error)

// Intercept calls the nested method `Intercept`
func (fn ChaosFunc) Intercept(req *http.Request) (bool, *http.Response, error) {
	return fn.Intercept(req)
}

func (fn ChaosFunc) String() string {
	return runtime.FuncForPC(reflect.ValueOf(fn).Pointer()).Name()
}

// NewChaosRoundTripper creates an http.RoundTripper that will intercept requests
// based on the provided Chaos functions. The notifier is invoked when a Chaos
// Intercept is fired.
func NewChaosRoundTripper(rt http.RoundTripper, notify ChaosNotifier, c ...Chaos) http.RoundTripper {
	return &chaosrt{rt, notify, c}
}

// RoundTrip gives each ChaosFunc an opportunity to intercept the request. The first
// interceptor wins.
func (rt *chaosrt) RoundTrip(req *http.Request) (*http.Response, error) {
	for _, c := range rt.c {
		if intercept, resp, err := c.Intercept(req); intercept {
			rt.notify.OnChaos(req, c)
			return resp, err
		}
	}
	return rt.rt.RoundTrip(req)
}

var _ = net.RoundTripperWrapper(&chaosrt{})

func (rt *chaosrt) WrappedRoundTripper() http.RoundTripper {
	return rt.rt
}

// Seed represents a consistent stream of chaos.
type Seed struct {
	*rand.Rand
}

// NewSeed creates an object that assists in generating random chaotic events
// based on a deterministic seed.
func NewSeed(seed int64) Seed {
	return Seed{rand.New(rand.NewSource(seed))}
}

type pIntercept struct {
	Chaos
	s Seed
	p float64
}

// P returns a ChaosFunc that fires with a probability of p (p between 0.0
// and 1.0 with 0.0 meaning never and 1.0 meaning always).
func (s Seed) P(p float64, c Chaos) Chaos {
	return pIntercept{c, s, p}
}

// Intercept intercepts requests with the provided probability p.
func (c pIntercept) Intercept(req *http.Request) (bool, *http.Response, error) {
	if c.s.Float64() < c.p {
		return c.Chaos.Intercept(req)
	}
	return false, nil, nil
}

func (c pIntercept) String() string {
	return fmt.Sprintf("P{%f %s}", c.p, c.Chaos)
}

// ErrSimulatedConnectionResetByPeer emulates the golang net error when a connection
// is reset by a peer.
// TODO: make this more accurate
// TODO: add other error types
// TODO: add a helper for returning multiple errors randomly.
var ErrSimulatedConnectionResetByPeer = Error{errors.New("connection reset by peer")}

// Error returns the nested error when C() is invoked.
type Error struct {
	error
}

// Intercept returns the nested error
func (e Error) Intercept(_ *http.Request) (bool, *http.Response, error) {
	return true, nil, e.error
}

// LogChaos is the default ChaosNotifier and writes a message to the Golang log.
var LogChaos = ChaosNotifier(logChaos{})

type logChaos struct{}

func (logChaos) OnChaos(req *http.Request, c Chaos) {
	log.Printf("Triggered chaotic behavior for %s %s: %v", req.Method, req.URL.String(), c)
}
