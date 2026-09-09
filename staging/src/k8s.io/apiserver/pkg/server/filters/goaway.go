/*
Copyright 2020 The Kubernetes Authors.

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

package filters

import (
	"math/rand"
	"net/http"
	"sync"
)

// GoawayDecider decides if server should send a GOAWAY
type GoawayDecider interface {
	Goaway(r *http.Request) bool
}

var (
	// randPool used to get a rand.Rand and generate a random number thread-safely,
	// which improve the performance of using rand.Rand with a locker
	randPool = &sync.Pool{
		New: func() interface{} {
			return rand.New(rand.NewSource(rand.Int63()))
		},
	}
)

// WithProbabilisticGoaway returns an http.Handler that send GOAWAY probabilistically
// according to the given chance for HTTP2 requests. After client receive GOAWAY,
// the in-flight long-running requests will not be influenced, and the new requests
// will use a new TCP connection to re-balancing to another server behind the load balance.
func WithProbabilisticGoaway(inner http.Handler, chance float64) http.Handler {
	return &goaway{
		handler: inner,
		decider: &probabilisticGoawayDecider{
			chance: chance,
			next: func() float64 {
				rnd := randPool.Get().(*rand.Rand)
				ret := rnd.Float64()
				randPool.Put(rnd)
				return ret
			},
		},
	}
}

// goaway send a GOAWAY to client according to decider for HTTP2 requests
type goaway struct {
	handler http.Handler
	decider GoawayDecider
}

// ServeHTTP implement HTTP handler
func (h *goaway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Proto == "HTTP/2.0" && h.decider.Goaway(r) {
		// Send a GOAWAY and tear down the TCP connection when idle.
		w.Header().Set("Connection", "close")
	}

	h.handler.ServeHTTP(w, r)
}

// probabilisticGoawayDecider send GOAWAY probabilistically according to chance
type probabilisticGoawayDecider struct {
	chance float64
	next   func() float64
}

// Goaway implement GoawayDecider
func (p *probabilisticGoawayDecider) Goaway(r *http.Request) bool {
	return p.next() < p.chance
}
