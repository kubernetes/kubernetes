/*
 *
 * Copyright 2017 gRPC authors.
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

package grpc

import (
	"sync"

	"golang.org/x/net/context"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/transport"
)

// pickerWrapper is a wrapper of balancer.Picker. It blocks on certain pick
// actions and unblock when there's a picker update.
type pickerWrapper struct {
	mu         sync.Mutex
	done       bool
	blockingCh chan struct{}
	picker     balancer.Picker

	// The latest connection happened.
	connErrMu sync.Mutex
	connErr   error
}

func newPickerWrapper() *pickerWrapper {
	bp := &pickerWrapper{blockingCh: make(chan struct{})}
	return bp
}

func (bp *pickerWrapper) updateConnectionError(err error) {
	bp.connErrMu.Lock()
	bp.connErr = err
	bp.connErrMu.Unlock()
}

func (bp *pickerWrapper) connectionError() error {
	bp.connErrMu.Lock()
	err := bp.connErr
	bp.connErrMu.Unlock()
	return err
}

// updatePicker is called by UpdateBalancerState. It unblocks all blocked pick.
func (bp *pickerWrapper) updatePicker(p balancer.Picker) {
	bp.mu.Lock()
	if bp.done {
		bp.mu.Unlock()
		return
	}
	bp.picker = p
	// bp.blockingCh should never be nil.
	close(bp.blockingCh)
	bp.blockingCh = make(chan struct{})
	bp.mu.Unlock()
}

// pick returns the transport that will be used for the RPC.
// It may block in the following cases:
// - there's no picker
// - the current picker returns ErrNoSubConnAvailable
// - the current picker returns other errors and failfast is false.
// - the subConn returned by the current picker is not READY
// When one of these situations happens, pick blocks until the picker gets updated.
func (bp *pickerWrapper) pick(ctx context.Context, failfast bool, opts balancer.PickOptions) (transport.ClientTransport, func(balancer.DoneInfo), error) {
	var (
		p  balancer.Picker
		ch chan struct{}
	)

	for {
		bp.mu.Lock()
		if bp.done {
			bp.mu.Unlock()
			return nil, nil, ErrClientConnClosing
		}

		if bp.picker == nil {
			ch = bp.blockingCh
		}
		if ch == bp.blockingCh {
			// This could happen when either:
			// - bp.picker is nil (the previous if condition), or
			// - has called pick on the current picker.
			bp.mu.Unlock()
			select {
			case <-ctx.Done():
				return nil, nil, ctx.Err()
			case <-ch:
			}
			continue
		}

		ch = bp.blockingCh
		p = bp.picker
		bp.mu.Unlock()

		subConn, done, err := p.Pick(ctx, opts)

		if err != nil {
			switch err {
			case balancer.ErrNoSubConnAvailable:
				continue
			case balancer.ErrTransientFailure:
				if !failfast {
					continue
				}
				return nil, nil, status.Errorf(codes.Unavailable, "%v, latest connection error: %v", err, bp.connectionError())
			default:
				// err is some other error.
				return nil, nil, toRPCErr(err)
			}
		}

		acw, ok := subConn.(*acBalancerWrapper)
		if !ok {
			grpclog.Infof("subconn returned from pick is not *acBalancerWrapper")
			continue
		}
		if t, ok := acw.getAddrConn().getReadyTransport(); ok {
			return t, done, nil
		}
		grpclog.Infof("blockingPicker: the picked transport is not ready, loop back to repick")
		// If ok == false, ac.state is not READY.
		// A valid picker always returns READY subConn. This means the state of ac
		// just changed, and picker will be updated shortly.
		// continue back to the beginning of the for loop to repick.
	}
}

func (bp *pickerWrapper) close() {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	if bp.done {
		return
	}
	bp.done = true
	close(bp.blockingCh)
}
