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

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
)

const (
	// The default value of flow control window size in HTTP2 spec.
	defaultWindowSize = 65535
	// The initial window size for flow control.
	initialWindowSize             = defaultWindowSize // for an RPC
	infinity                      = time.Duration(math.MaxInt64)
	defaultClientKeepaliveTime    = infinity
	defaultClientKeepaliveTimeout = time.Duration(20 * time.Second)
	defaultMaxStreamsClient       = 100
	defaultMaxConnectionIdle      = infinity
	defaultMaxConnectionAge       = infinity
	defaultMaxConnectionAgeGrace  = infinity
	defaultServerKeepaliveTime    = time.Duration(2 * time.Hour)
	defaultServerKeepaliveTimeout = time.Duration(20 * time.Second)
	defaultKeepalivePolicyMinTime = time.Duration(5 * time.Minute)
	// max window limit set by HTTP2 Specs.
	maxWindowSize = math.MaxInt32
	// defaultLocalSendQuota sets is default value for number of data
	// bytes that each stream can schedule before some of it being
	// flushed out.
	defaultLocalSendQuota = 64 * 1024
)

// The following defines various control items which could flow through
// the control buffer of transport. They represent different aspects of
// control tasks, e.g., flow control, settings, streaming resetting, etc.

type headerFrame struct {
	streamID  uint32
	hf        []hpack.HeaderField
	endStream bool
}

func (*headerFrame) item() {}

type continuationFrame struct {
	streamID            uint32
	endHeaders          bool
	headerBlockFragment []byte
}

type dataFrame struct {
	streamID  uint32
	endStream bool
	d         []byte
	f         func()
}

func (*dataFrame) item() {}

func (*continuationFrame) item() {}

type windowUpdate struct {
	streamID  uint32
	increment uint32
}

func (*windowUpdate) item() {}

type settings struct {
	ack bool
	ss  []http2.Setting
}

func (*settings) item() {}

type resetStream struct {
	streamID uint32
	code     http2.ErrCode
}

func (*resetStream) item() {}

type goAway struct {
	code      http2.ErrCode
	debugData []byte
	headsUp   bool
	closeConn bool
}

func (*goAway) item() {}

type flushIO struct {
}

func (*flushIO) item() {}

type ping struct {
	ack  bool
	data [8]byte
}

func (*ping) item() {}

// quotaPool is a pool which accumulates the quota and sends it to acquire()
// when it is available.
type quotaPool struct {
	c chan int

	mu      sync.Mutex
	version uint32
	quota   int
}

// newQuotaPool creates a quotaPool which has quota q available to consume.
func newQuotaPool(q int) *quotaPool {
	qb := &quotaPool{
		c: make(chan int, 1),
	}
	if q > 0 {
		qb.c <- q
	} else {
		qb.quota = q
	}
	return qb
}

// add cancels the pending quota sent on acquired, incremented by v and sends
// it back on acquire.
func (qb *quotaPool) add(v int) {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	qb.lockedAdd(v)
}

func (qb *quotaPool) lockedAdd(v int) {
	select {
	case n := <-qb.c:
		qb.quota += n
	default:
	}
	qb.quota += v
	if qb.quota <= 0 {
		return
	}
	// After the pool has been created, this is the only place that sends on
	// the channel. Since mu is held at this point and any quota that was sent
	// on the channel has been retrieved, we know that this code will always
	// place any positive quota value on the channel.
	select {
	case qb.c <- qb.quota:
		qb.quota = 0
	default:
	}
}

func (qb *quotaPool) addAndUpdate(v int) {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	qb.lockedAdd(v)
	// Update the version only after having added to the quota
	// so that if acquireWithVesrion sees the new vesrion it is
	// guaranteed to have seen the updated quota.
	// Also, still keep this inside of the lock, so that when
	// compareAndExecute is processing, this function doesn't
	// get executed partially (quota gets updated but the version
	// doesn't).
	atomic.AddUint32(&(qb.version), 1)
}

func (qb *quotaPool) acquireWithVersion() (<-chan int, uint32) {
	return qb.c, atomic.LoadUint32(&(qb.version))
}

func (qb *quotaPool) compareAndExecute(version uint32, success, failure func()) bool {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	if version == atomic.LoadUint32(&(qb.version)) {
		success()
		return true
	}
	failure()
	return false
}

// acquire returns the channel on which available quota amounts are sent.
func (qb *quotaPool) acquire() <-chan int {
	return qb.c
}

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
	defer f.mu.Unlock()
	d := n - f.limit
	f.limit = n
	return d
}

func (f *inFlow) maybeAdjust(n uint32) uint32 {
	if n > uint32(math.MaxInt32) {
		n = uint32(math.MaxInt32)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
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
		// Sender's window shouldn't go more than 2^31 - 1 as speecified in the HTTP spec.
		if f.limit+n > maxWindowSize {
			f.delta = maxWindowSize - f.limit
		} else {
			// Send a window update for the whole message and not just the difference between
			// estUntransmittedData and estSenderQuota. This will be helpful in case the message
			// is padded; We will fallback on the current available window(at least a 1/4th of the limit).
			f.delta = n
		}
		return f.delta
	}
	return 0
}

// onData is invoked when some data frame is received. It updates pendingData.
func (f *inFlow) onData(n uint32) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.pendingData += n
	if f.pendingData+f.pendingUpdate > f.limit+f.delta {
		return fmt.Errorf("received %d-bytes data exceeding the limit %d bytes", f.pendingData+f.pendingUpdate, f.limit)
	}
	return nil
}

// onRead is invoked when the application reads the data. It returns the window size
// to be sent to the peer.
func (f *inFlow) onRead(n uint32) uint32 {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.pendingData == 0 {
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
		return wu
	}
	return 0
}

func (f *inFlow) resetPendingUpdate() uint32 {
	f.mu.Lock()
	defer f.mu.Unlock()
	n := f.pendingUpdate
	f.pendingUpdate = 0
	return n
}
