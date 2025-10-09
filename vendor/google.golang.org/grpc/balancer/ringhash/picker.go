/*
 *
 * Copyright 2021 gRPC authors.
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

package ringhash

import (
	"fmt"
	"strings"

	xxhash "github.com/cespare/xxhash/v2"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	iringhash "google.golang.org/grpc/internal/ringhash"
	"google.golang.org/grpc/metadata"
)

type picker struct {
	ring *ring

	// endpointStates is a cache of endpoint states.
	// The ringhash balancer stores endpoint states in a `resolver.EndpointMap`,
	// with access guarded by `ringhashBalancer.mu`. The `endpointStates` cache
	// in the picker helps avoid locking the ringhash balancer's mutex when
	// reading the latest state at RPC time.
	endpointStates map[string]endpointState // endpointState.hashKey -> endpointState

	// requestHashHeader is the header key to look for the request hash. If it's
	// empty, the request hash is expected to be set in the context via xDS.
	// See gRFC A76.
	requestHashHeader string

	// hasEndpointInConnectingState is true if any of the endpoints is in
	// CONNECTING.
	hasEndpointInConnectingState bool

	randUint64 func() uint64
}

func (p *picker) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	usingRandomHash := false
	var requestHash uint64
	if p.requestHashHeader == "" {
		var ok bool
		if requestHash, ok = iringhash.XDSRequestHash(info.Ctx); !ok {
			return balancer.PickResult{}, fmt.Errorf("ringhash: expected xDS config selector to set the request hash")
		}
	} else {
		md, ok := metadata.FromOutgoingContext(info.Ctx)
		if !ok || len(md.Get(p.requestHashHeader)) == 0 {
			requestHash = p.randUint64()
			usingRandomHash = true
		} else {
			values := strings.Join(md.Get(p.requestHashHeader), ",")
			requestHash = xxhash.Sum64String(values)
		}
	}

	e := p.ring.pick(requestHash)
	ringSize := len(p.ring.items)
	if !usingRandomHash {
		// Per gRFC A61, because of sticky-TF with PickFirst's auto reconnect on TF,
		// we ignore all TF subchannels and find the first ring entry in READY,
		// CONNECTING or IDLE.  If that entry is in IDLE, we need to initiate a
		// connection. The idlePicker returned by the LazyLB or the new Pickfirst
		// should do this automatically.
		for i := 0; i < ringSize; i++ {
			index := (e.idx + i) % ringSize
			es := p.endpointState(p.ring.items[index])
			switch es.state.ConnectivityState {
			case connectivity.Ready, connectivity.Connecting, connectivity.Idle:
				return es.state.Picker.Pick(info)
			case connectivity.TransientFailure:
			default:
				panic(fmt.Sprintf("Found child balancer in unknown state: %v", es.state.ConnectivityState))
			}
		}
	} else {
		// If the picker has generated a random hash, it will walk the ring from
		// this hash, and pick the first READY endpoint. If no endpoint is
		// currently in CONNECTING state, it will trigger a connection attempt
		// on at most one endpoint that is in IDLE state along the way. - A76
		requestedConnection := p.hasEndpointInConnectingState
		for i := 0; i < ringSize; i++ {
			index := (e.idx + i) % ringSize
			es := p.endpointState(p.ring.items[index])
			if es.state.ConnectivityState == connectivity.Ready {
				return es.state.Picker.Pick(info)
			}
			if !requestedConnection && es.state.ConnectivityState == connectivity.Idle {
				requestedConnection = true
				// If the SubChannel is in idle state, initiate a connection but
				// continue to check other pickers to see if there is one in
				// ready state.
				es.balancer.ExitIdle()
			}
		}
		if requestedConnection {
			return balancer.PickResult{}, balancer.ErrNoSubConnAvailable
		}
	}

	// All children are in transient failure. Return the first failure.
	return p.endpointState(e).state.Picker.Pick(info)
}

func (p *picker) endpointState(e *ringEntry) endpointState {
	return p.endpointStates[e.hashKey]
}
