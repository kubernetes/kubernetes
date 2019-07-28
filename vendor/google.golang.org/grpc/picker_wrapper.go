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
	"io"
	"sync"
	"sync/atomic"

	"golang.org/x/net/context"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/resolver"
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

	stickinessMDKey atomic.Value
	stickiness      *stickyStore
}

func newPickerWrapper() *pickerWrapper {
	bp := &pickerWrapper{
		blockingCh: make(chan struct{}),
		stickiness: newStickyStore(),
	}
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

func (bp *pickerWrapper) updateStickinessMDKey(newKey string) {
	// No need to check ok because mdKey == "" if ok == false.
	if oldKey, _ := bp.stickinessMDKey.Load().(string); oldKey != newKey {
		bp.stickinessMDKey.Store(newKey)
		bp.stickiness.reset(newKey)
	}
}

func (bp *pickerWrapper) getStickinessMDKey() string {
	// No need to check ok because mdKey == "" if ok == false.
	mdKey, _ := bp.stickinessMDKey.Load().(string)
	return mdKey
}

func (bp *pickerWrapper) clearStickinessState() {
	if oldKey := bp.getStickinessMDKey(); oldKey != "" {
		// There's no need to reset store if mdKey was "".
		bp.stickiness.reset(oldKey)
	}
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
func (bp *pickerWrapper) pick(ctx context.Context, failfast bool, opts balancer.PickOptions) (transport.ClientTransport, func(balancer.DoneInfo), error) {

	mdKey := bp.getStickinessMDKey()
	stickyKey, isSticky := stickyKeyFromContext(ctx, mdKey)

	// Potential race here: if stickinessMDKey is updated after the above two
	// lines, and this pick is a sticky pick, the following put could add an
	// entry to sticky store with an outdated sticky key.
	//
	// The solution: keep the current md key in sticky store, and at the
	// beginning of each get/put, check the mdkey against store.curMDKey.
	//  - Cons: one more string comparing for each get/put.
	//  - Pros: the string matching happens inside get/put, so the overhead for
	//  non-sticky RPCs will be minimal.

	if isSticky {
		if t, ok := bp.stickiness.get(mdKey, stickyKey); ok {
			// Done function returned is always nil.
			return t, nil, nil
		}
	}

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
			if isSticky {
				bp.stickiness.put(mdKey, stickyKey, acw)
			}
			if channelz.IsOn() {
				return t, doneChannelzWrapper(acw, done), nil
			}
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

const stickinessKeyCountLimit = 1000

type stickyStoreEntry struct {
	acw  *acBalancerWrapper
	addr resolver.Address
}

type stickyStore struct {
	mu sync.Mutex
	// curMDKey is check before every get/put to avoid races. The operation will
	// abort immediately when the given mdKey is different from the curMDKey.
	curMDKey string
	store    *linkedMap
}

func newStickyStore() *stickyStore {
	return &stickyStore{
		store: newLinkedMap(),
	}
}

// reset clears the map in stickyStore, and set the currentMDKey to newMDKey.
func (ss *stickyStore) reset(newMDKey string) {
	ss.mu.Lock()
	ss.curMDKey = newMDKey
	ss.store.clear()
	ss.mu.Unlock()
}

// stickyKey is the key to look up in store. mdKey will be checked against
// curMDKey to avoid races.
func (ss *stickyStore) put(mdKey, stickyKey string, acw *acBalancerWrapper) {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	if mdKey != ss.curMDKey {
		return
	}
	// TODO(stickiness): limit the total number of entries.
	ss.store.put(stickyKey, &stickyStoreEntry{
		acw:  acw,
		addr: acw.getAddrConn().getCurAddr(),
	})
	if ss.store.len() > stickinessKeyCountLimit {
		ss.store.removeOldest()
	}
}

// stickyKey is the key to look up in store. mdKey will be checked against
// curMDKey to avoid races.
func (ss *stickyStore) get(mdKey, stickyKey string) (transport.ClientTransport, bool) {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	if mdKey != ss.curMDKey {
		return nil, false
	}
	entry, ok := ss.store.get(stickyKey)
	if !ok {
		return nil, false
	}
	ac := entry.acw.getAddrConn()
	if ac.getCurAddr() != entry.addr {
		ss.store.remove(stickyKey)
		return nil, false
	}
	t, ok := ac.getReadyTransport()
	if !ok {
		ss.store.remove(stickyKey)
		return nil, false
	}
	return t, true
}

// Get one value from metadata in ctx with key stickinessMDKey.
//
// It returns "", false if stickinessMDKey is an empty string.
func stickyKeyFromContext(ctx context.Context, stickinessMDKey string) (string, bool) {
	if stickinessMDKey == "" {
		return "", false
	}

	md, added, ok := metadata.FromOutgoingContextRaw(ctx)
	if !ok {
		return "", false
	}

	if vv, ok := md[stickinessMDKey]; ok {
		if len(vv) > 0 {
			return vv[0], true
		}
	}

	for _, ss := range added {
		for i := 0; i < len(ss)-1; i += 2 {
			if ss[i] == stickinessMDKey {
				return ss[i+1], true
			}
		}
	}

	return "", false
}
