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
	"context"
	"io"
	"sync"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/internal/channelz"
	istatus "google.golang.org/grpc/internal/status"
	"google.golang.org/grpc/internal/transport"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
)

// pickerWrapper is a wrapper of balancer.Picker. It blocks on certain pick
// actions and unblock when there's a picker update.
type pickerWrapper struct {
	mu            sync.Mutex
	done          bool
	blockingCh    chan struct{}
	picker        balancer.Picker
	statsHandlers []stats.Handler // to record blocking picker calls
}

func newPickerWrapper(statsHandlers []stats.Handler) *pickerWrapper {
	return &pickerWrapper{
		blockingCh:    make(chan struct{}),
		statsHandlers: statsHandlers,
	}
}

// updatePicker is called by UpdateBalancerState. It unblocks all blocked pick.
func (pw *pickerWrapper) updatePicker(p balancer.Picker) {
	pw.mu.Lock()
	if pw.done {
		pw.mu.Unlock()
		return
	}
	pw.picker = p
	// pw.blockingCh should never be nil.
	close(pw.blockingCh)
	pw.blockingCh = make(chan struct{})
	pw.mu.Unlock()
}

// doneChannelzWrapper performs the following:
//   - increments the calls started channelz counter
//   - wraps the done function in the passed in result to increment the calls
//     failed or calls succeeded channelz counter before invoking the actual
//     done function.
func doneChannelzWrapper(acbw *acBalancerWrapper, result *balancer.PickResult) {
	ac := acbw.ac
	ac.incrCallsStarted()
	done := result.Done
	result.Done = func(b balancer.DoneInfo) {
		if b.Err != nil && b.Err != io.EOF {
			ac.incrCallsFailed()
		} else {
			ac.incrCallsSucceeded()
		}
		if done != nil {
			done(b)
		}
	}
}

// pick returns the transport that will be used for the RPC.
// It may block in the following cases:
// - there's no picker
// - the current picker returns ErrNoSubConnAvailable
// - the current picker returns other errors and failfast is false.
// - the subConn returned by the current picker is not READY
// When one of these situations happens, pick blocks until the picker gets updated.
func (pw *pickerWrapper) pick(ctx context.Context, failfast bool, info balancer.PickInfo) (transport.ClientTransport, balancer.PickResult, error) {
	var ch chan struct{}

	var lastPickErr error

	for {
		pw.mu.Lock()
		if pw.done {
			pw.mu.Unlock()
			return nil, balancer.PickResult{}, ErrClientConnClosing
		}

		if pw.picker == nil {
			ch = pw.blockingCh
		}
		if ch == pw.blockingCh {
			// This could happen when either:
			// - pw.picker is nil (the previous if condition), or
			// - has called pick on the current picker.
			pw.mu.Unlock()
			select {
			case <-ctx.Done():
				var errStr string
				if lastPickErr != nil {
					errStr = "latest balancer error: " + lastPickErr.Error()
				} else {
					errStr = ctx.Err().Error()
				}
				switch ctx.Err() {
				case context.DeadlineExceeded:
					return nil, balancer.PickResult{}, status.Error(codes.DeadlineExceeded, errStr)
				case context.Canceled:
					return nil, balancer.PickResult{}, status.Error(codes.Canceled, errStr)
				}
			case <-ch:
			}
			continue
		}

		// If the channel is set, it means that the pick call had to wait for a
		// new picker at some point. Either it's the first iteration and this
		// function received the first picker, or a picker errored with
		// ErrNoSubConnAvailable or errored with failfast set to false, which
		// will trigger a continue to the next iteration. In the first case this
		// conditional will hit if this call had to block (the channel is set).
		// In the second case, the only way it will get to this conditional is
		// if there is a new picker.
		if ch != nil {
			for _, sh := range pw.statsHandlers {
				sh.HandleRPC(ctx, &stats.PickerUpdated{})
			}
		}

		ch = pw.blockingCh
		p := pw.picker
		pw.mu.Unlock()

		pickResult, err := p.Pick(info)
		if err != nil {
			if err == balancer.ErrNoSubConnAvailable {
				continue
			}
			if st, ok := status.FromError(err); ok {
				// Status error: end the RPC unconditionally with this status.
				// First restrict the code to the list allowed by gRFC A54.
				if istatus.IsRestrictedControlPlaneCode(st) {
					err = status.Errorf(codes.Internal, "received picker error with illegal status: %v", err)
				}
				return nil, balancer.PickResult{}, dropError{error: err}
			}
			// For all other errors, wait for ready RPCs should block and other
			// RPCs should fail with unavailable.
			if !failfast {
				lastPickErr = err
				continue
			}
			return nil, balancer.PickResult{}, status.Error(codes.Unavailable, err.Error())
		}

		acbw, ok := pickResult.SubConn.(*acBalancerWrapper)
		if !ok {
			logger.Errorf("subconn returned from pick is type %T, not *acBalancerWrapper", pickResult.SubConn)
			continue
		}
		if t := acbw.ac.getReadyTransport(); t != nil {
			if channelz.IsOn() {
				doneChannelzWrapper(acbw, &pickResult)
				return t, pickResult, nil
			}
			return t, pickResult, nil
		}
		if pickResult.Done != nil {
			// Calling done with nil error, no bytes sent and no bytes received.
			// DoneInfo with default value works.
			pickResult.Done(balancer.DoneInfo{})
		}
		logger.Infof("blockingPicker: the picked transport is not ready, loop back to repick")
		// If ok == false, ac.state is not READY.
		// A valid picker always returns READY subConn. This means the state of ac
		// just changed, and picker will be updated shortly.
		// continue back to the beginning of the for loop to repick.
	}
}

func (pw *pickerWrapper) close() {
	pw.mu.Lock()
	defer pw.mu.Unlock()
	if pw.done {
		return
	}
	pw.done = true
	close(pw.blockingCh)
}

// reset clears the pickerWrapper and prepares it for being used again when idle
// mode is exited.
func (pw *pickerWrapper) reset() {
	pw.mu.Lock()
	defer pw.mu.Unlock()
	if pw.done {
		return
	}
	pw.blockingCh = make(chan struct{})
}

// dropError is a wrapper error that indicates the LB policy wishes to drop the
// RPC and not retry it.
type dropError struct {
	error
}
