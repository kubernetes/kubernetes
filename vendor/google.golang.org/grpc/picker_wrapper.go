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
	"fmt"
	"io"
	"sync"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/internal/transport"
	"google.golang.org/grpc/status"
)

// v2PickerWrapper wraps a balancer.Picker while providing the
// balancer.V2Picker API.  It requires a pickerWrapper to generate errors
// including the latest connectionError.  To be deleted when balancer.Picker is
// updated to the balancer.V2Picker API.
type v2PickerWrapper struct {
	picker  balancer.Picker
	connErr *connErr
}

func (v *v2PickerWrapper) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	sc, done, err := v.picker.Pick(info.Ctx, info)
	if err != nil {
		if err == balancer.ErrTransientFailure {
			return balancer.PickResult{}, balancer.TransientFailureError(fmt.Errorf("%v, latest connection error: %v", err, v.connErr.connectionError()))
		}
		return balancer.PickResult{}, err
	}
	return balancer.PickResult{SubConn: sc, Done: done}, nil
}

// pickerWrapper is a wrapper of balancer.Picker. It blocks on certain pick
// actions and unblock when there's a picker update.
type pickerWrapper struct {
	mu         sync.Mutex
	done       bool
	blockingCh chan struct{}
	picker     balancer.V2Picker

	// The latest connection error.  TODO: remove when V1 picker is deprecated;
	// balancer should be responsible for providing the error.
	*connErr
}

type connErr struct {
	mu  sync.Mutex
	err error
}

func (c *connErr) updateConnectionError(err error) {
	c.mu.Lock()
	c.err = err
	c.mu.Unlock()
}

func (c *connErr) connectionError() error {
	c.mu.Lock()
	err := c.err
	c.mu.Unlock()
	return err
}

func newPickerWrapper() *pickerWrapper {
	return &pickerWrapper{blockingCh: make(chan struct{}), connErr: &connErr{}}
}

// updatePicker is called by UpdateBalancerState. It unblocks all blocked pick.
func (pw *pickerWrapper) updatePicker(p balancer.Picker) {
	pw.updatePickerV2(&v2PickerWrapper{picker: p, connErr: pw.connErr})
}

// updatePicker is called by UpdateBalancerState. It unblocks all blocked pick.
func (pw *pickerWrapper) updatePickerV2(p balancer.V2Picker) {
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

func doneChannelzWrapper(acw *acBalancerWrapper, done func(balancer.DoneInfo)) func(balancer.DoneInfo) {
	acw.mu.Lock()
	ac := acw.ac
	acw.mu.Unlock()
	ac.incrCallsStarted()
	return func(b balancer.DoneInfo) {
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
func (pw *pickerWrapper) pick(ctx context.Context, failfast bool, info balancer.PickInfo) (transport.ClientTransport, func(balancer.DoneInfo), error) {
	var ch chan struct{}

	var lastPickErr error
	for {
		pw.mu.Lock()
		if pw.done {
			pw.mu.Unlock()
			return nil, nil, ErrClientConnClosing
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
				} else if connectionErr := pw.connectionError(); connectionErr != nil {
					errStr = "latest connection error: " + connectionErr.Error()
				} else {
					errStr = ctx.Err().Error()
				}
				switch ctx.Err() {
				case context.DeadlineExceeded:
					return nil, nil, status.Error(codes.DeadlineExceeded, errStr)
				case context.Canceled:
					return nil, nil, status.Error(codes.Canceled, errStr)
				}
			case <-ch:
			}
			continue
		}

		ch = pw.blockingCh
		p := pw.picker
		pw.mu.Unlock()

		pickResult, err := p.Pick(info)

		if err != nil {
			if err == balancer.ErrNoSubConnAvailable {
				continue
			}
			if tfe, ok := err.(interface{ IsTransientFailure() bool }); ok && tfe.IsTransientFailure() {
				if !failfast {
					lastPickErr = err
					continue
				}
				return nil, nil, status.Error(codes.Unavailable, err.Error())
			}
			if _, ok := status.FromError(err); ok {
				return nil, nil, err
			}
			// err is some other error.
			return nil, nil, status.Error(codes.Unknown, err.Error())
		}

		acw, ok := pickResult.SubConn.(*acBalancerWrapper)
		if !ok {
			grpclog.Error("subconn returned from pick is not *acBalancerWrapper")
			continue
		}
		if t, ok := acw.getAddrConn().getReadyTransport(); ok {
			if channelz.IsOn() {
				return t, doneChannelzWrapper(acw, pickResult.Done), nil
			}
			return t, pickResult.Done, nil
		}
		if pickResult.Done != nil {
			// Calling done with nil error, no bytes sent and no bytes received.
			// DoneInfo with default value works.
			pickResult.Done(balancer.DoneInfo{})
		}
		grpclog.Infof("blockingPicker: the picked transport is not ready, loop back to repick")
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
