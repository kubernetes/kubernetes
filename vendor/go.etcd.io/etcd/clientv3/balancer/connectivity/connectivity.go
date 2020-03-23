// Copyright 2019 The etcd Authors
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

// Package connectivity implements client connectivity operations.
package connectivity

import (
	"sync"

	"go.uber.org/zap"
	"google.golang.org/grpc/connectivity"
)

// Recorder records gRPC connectivity.
type Recorder interface {
	GetCurrentState() connectivity.State
	RecordTransition(oldState, newState connectivity.State)
}

// New returns a new Recorder.
func New(lg *zap.Logger) Recorder {
	return &recorder{lg: lg}
}

// recorder takes the connectivity states of multiple SubConns
// and returns one aggregated connectivity state.
// ref. https://github.com/grpc/grpc-go/blob/master/balancer/balancer.go
type recorder struct {
	lg *zap.Logger

	mu sync.RWMutex

	cur connectivity.State

	numReady            uint64 // Number of addrConns in ready state.
	numConnecting       uint64 // Number of addrConns in connecting state.
	numTransientFailure uint64 // Number of addrConns in transientFailure.
}

func (rc *recorder) GetCurrentState() (state connectivity.State) {
	rc.mu.RLock()
	defer rc.mu.RUnlock()
	return rc.cur
}

// RecordTransition records state change happening in subConn and based on that
// it evaluates what aggregated state should be.
//
//  - If at least one SubConn in Ready, the aggregated state is Ready;
//  - Else if at least one SubConn in Connecting, the aggregated state is Connecting;
//  - Else the aggregated state is TransientFailure.
//
// Idle and Shutdown are not considered.
//
// ref. https://github.com/grpc/grpc-go/blob/master/balancer/balancer.go
func (rc *recorder) RecordTransition(oldState, newState connectivity.State) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	for idx, state := range []connectivity.State{oldState, newState} {
		updateVal := 2*uint64(idx) - 1 // -1 for oldState and +1 for new.
		switch state {
		case connectivity.Ready:
			rc.numReady += updateVal
		case connectivity.Connecting:
			rc.numConnecting += updateVal
		case connectivity.TransientFailure:
			rc.numTransientFailure += updateVal
		default:
			rc.lg.Warn("connectivity recorder received unknown state", zap.String("connectivity-state", state.String()))
		}
	}

	switch { // must be exclusive, no overlap
	case rc.numReady > 0:
		rc.cur = connectivity.Ready
	case rc.numConnecting > 0:
		rc.cur = connectivity.Connecting
	default:
		rc.cur = connectivity.TransientFailure
	}
}
