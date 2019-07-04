/*
 *
 * Copyright 2014 gRPC authors.
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

package transport

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// The default value of flow control window size in HTTP2 spec.
	defaultWindowSize = 65535
	// The initial window size for flow control.
	initialWindowSize             = defaultWindowSize // for an RPC
	infinity                      = time.Duration(math.MaxInt64)
	defaultClientKeepaliveTime    = infinity
	defaultClientKeepaliveTimeout = 20 * time.Second
	defaultMaxStreamsClient       = 100
	defaultMaxConnectionIdle      = infinity
	defaultMaxConnectionAge       = infinity
	defaultMaxConnectionAgeGrace  = infinity
	defaultServerKeepaliveTime    = 2 * time.Hour
	defaultServerKeepaliveTimeout = 20 * time.Second
	defaultKeepalivePolicyMinTime = 5 * time.Minute
	// max window limit set by HTTP2 Specs.
	maxWindowSize = math.MaxInt32
	// defaultWriteQuota is the default value for number of data
	// bytes that each stream can schedule before some of it being
	// flushed out.
	defaultWriteQuota = 64 * 1024
)

// writeQuota is a soft limit on the amount of data a stream can
// schedule before some of it is written out.
type writeQuota struct {
	quota int32
	// get waits on read from when quota goes less than or equal to zero.
	// replenish writes on it when quota goes positive again.
	ch chan struct{}
	// done is triggered in error case.
	done <-chan struct{}
	// replenish is called by loopyWriter to give quota back to.
	// It is implemented as a field so that it can be updated
	// by tests.
	replenish func(n int)
}

func newWriteQuota(sz int32, done <-chan struct{}) *writeQuota {
	w := &writeQuota{
		quota: sz,
		ch:    make(chan struct{}, 1),
		done:  done,
	}
	w.replenish = w.realReplenish
	return w
}

func (w *writeQuota) get(sz int32) error {
	for {
		if atomic.LoadInt32(&w.quota) > 0 {
			atomic.AddInt32(&w.quota, -sz)
			return nil
		}
		select {
		case <-w.ch:
			continue
		case <-w.done:
			return errStreamDone
		}
	}
}

func (w *writeQuota) realReplenish(n int) {
	sz := int32(n)
	a := atomic.AddInt32(&w.quota, sz)
	b := a - sz
	if b <= 0 && a > 0 {
		select {
		case w.ch <- struct{}{}:
		default:
		}
	}
}

type trInFlow struct {
	limit               uint32
	unacked             uint32
	effectiveWindowSize uint32
}

func (f *trInFlow) newLimit(n uint32) uint32 {
	d := n - f.limit
	f.limit = n
	f.updateEffectiveWindowSize()
	return d
}

func (f *trInFlow) onData(n uint32) uint32 {
	f.unacked += n
	if f.unacked >= f.limit/4 {
		w := f.unacked
		f.unacked = 0
		f.updateEffectiveWindowSize()
		return w
	}
	f.updateEffectiveWindowSize()
	return 0
}

func (f *trInFlow) reset() uint32 {
	w := f.unacked
	f.unacked = 0
	f.updateEffectiveWindowSize()
	return w
}

func (f *trInFlow) updateEffectiveWindowSize() {
	atomic.StoreUint32(&f.effectiveWindowSize, f.limit-f.unacked)
}

func (f *trInFlow) getSize() uint32 {
	return atomic.LoadUint32(&f.effectiveWindowSize)
}

// TODO(mmukhi): Simplify this code.
// inFlow deals with inbound flow control
type inFlow struct {
	mu sync.Mutex
	// The inbound flow control limit for pending data.
	limit uint32
	// pendingData is the overall data which have been received but not been
	// consumed by applications.
	pendingData uint32
	// The amount of data the application has consumed but grpc has not sent
	// window update for them. Used to reduce window update frequency.
	pendingUpdate uint32
	// delta is the extra window update given by receiver when an application
	// is reading data bigger in size than the inFlow limit.
	delta uint32
}

// newLimit updates the inflow window to a new value n.
// It assumes that n is always greater than the old limit.
func (f *inFlow) newLimit(n uint32) uint32 {
	f.mu.Lock()
	d := n - f.limit
	f.limit = n
	f.mu.Unlock()
	return d
}

func (f *inFlow) maybeAdjust(n uint32) uint32 {
	if n > uint32(math.MaxInt32) {
		n = uint32(math.MaxInt32)
	}
	f.mu.Lock()
	// estSenderQuota is the receiver's view of the maximum number of bytes the sender
	// can send without a window update.
	estSenderQuota := int32(f.limit - (f.pendingData + f.pendingUpdate))
	// estUntransmittedData is the maximum number of bytes the sends might not have put
	// on the wire yet. A value of 0 or less means that we have already received all or
	// more bytes than the application is requesting to read.
	estUntransmittedData := int32(n - f.pendingData) // Casting into int32 since it could be negative.
	// This implies that unless we send a window update, the sender won't be able to send all the bytes
	// for this message. Therefore we must send an update over the limit since there's an active read
	// request from the application.
	if estUntransmittedData > estSenderQuota {
		// Sender's window shouldn't go more than 2^31 - 1 as specified in the HTTP spec.
		if f.limit+n > maxWindowSize {
			f.delta = maxWindowSize - f.limit
		} else {
			// Send a window update for the whole message and not just the difference between
			// estUntransmittedData and estSenderQuota. This will be helpful in case the message
			// is padded; We will fallback on the current available window(at least a 1/4th of the limit).
			f.delta = n
		}
		f.mu.Unlock()
		return f.delta
	}
	f.mu.Unlock()
	return 0
}

// onData is invoked when some data frame is received. It updates pendingData.
func (f *inFlow) onData(n uint32) error {
	f.mu.Lock()
	f.pendingData += n
	if f.pendingData+f.pendingUpdate > f.limit+f.delta {
		limit := f.limit
		rcvd := f.pendingData + f.pendingUpdate
		f.mu.Unlock()
		return fmt.Errorf("received %d-bytes data exceeding the limit %d bytes", rcvd, limit)
	}
	f.mu.Unlock()
	return nil
}

// onRead is invoked when the application reads the data. It returns the window size
// to be sent to the peer.
func (f *inFlow) onRead(n uint32) uint32 {
	f.mu.Lock()
	if f.pendingData == 0 {
		f.mu.Unlock()
		return 0
	}
	f.pendingData -= n
	if n > f.delta {
		n -= f.delta
		f.delta = 0
	} else {
		f.delta -= n
		n = 0
	}
	f.pendingUpdate += n
	if f.pendingUpdate >= f.limit/4 {
		wu := f.pendingUpdate
		f.pendingUpdate = 0
		f.mu.Unlock()
		return wu
	}
	f.mu.Unlock()
	return 0
}
