/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package transport

import (
	"fmt"
	"sync"

	"golang.org/x/net/http2"
)

const (
	// The default value of flow control window size in HTTP2 spec.
	defaultWindowSize = 65535
	// The initial window size for flow control.
	initialWindowSize     = defaultWindowSize      // for an RPC
	initialConnWindowSize = defaultWindowSize * 16 // for a connection
)

// The following defines various control items which could flow through
// the control buffer of transport. They represent different aspects of
// control tasks, e.g., flow control, settings, streaming resetting, etc.
type windowUpdate struct {
	streamID  uint32
	increment uint32
}

func (windowUpdate) isItem() bool {
	return true
}

type settings struct {
	ack bool
	ss  []http2.Setting
}

func (settings) isItem() bool {
	return true
}

type resetStream struct {
	streamID uint32
	code     http2.ErrCode
}

func (resetStream) isItem() bool {
	return true
}

type flushIO struct {
}

func (flushIO) isItem() bool {
	return true
}

type ping struct {
	ack  bool
	data [8]byte
}

func (ping) isItem() bool {
	return true
}

// quotaPool is a pool which accumulates the quota and sends it to acquire()
// when it is available.
type quotaPool struct {
	c chan int

	mu    sync.Mutex
	quota int
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

// add adds n to the available quota and tries to send it on acquire.
func (qb *quotaPool) add(n int) {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	qb.quota += n
	if qb.quota <= 0 {
		return
	}
	select {
	case qb.c <- qb.quota:
		qb.quota = 0
	default:
	}
}

// cancel cancels the pending quota sent on acquire, if any.
func (qb *quotaPool) cancel() {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	select {
	case n := <-qb.c:
		qb.quota += n
	default:
	}
}

// reset cancels the pending quota sent on acquired, incremented by v and sends
// it back on acquire.
func (qb *quotaPool) reset(v int) {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	select {
	case n := <-qb.c:
		qb.quota += n
	default:
	}
	qb.quota += v
	if qb.quota <= 0 {
		return
	}
	select {
	case qb.c <- qb.quota:
		qb.quota = 0
	default:
	}
}

// acquire returns the channel on which available quota amounts are sent.
func (qb *quotaPool) acquire() <-chan int {
	return qb.c
}

// inFlow deals with inbound flow control
type inFlow struct {
	// The inbound flow control limit for pending data.
	limit uint32
	// conn points to the shared connection-level inFlow that is shared
	// by all streams on that conn. It is nil for the inFlow on the conn
	// directly.
	conn *inFlow

	mu sync.Mutex
	// pendingData is the overall data which have been received but not been
	// consumed by applications.
	pendingData uint32
	// The amount of data the application has consumed but grpc has not sent
	// window update for them. Used to reduce window update frequency.
	pendingUpdate uint32
}

// onData is invoked when some data frame is received. It increments not only its
// own pendingData but also that of the associated connection-level flow.
func (f *inFlow) onData(n uint32) error {
	if n == 0 {
		return nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.pendingData+f.pendingUpdate+n > f.limit {
		return fmt.Errorf("recieved %d-bytes data exceeding the limit %d bytes", f.pendingData+f.pendingUpdate+n, f.limit)
	}
	if f.conn != nil {
		if err := f.conn.onData(n); err != nil {
			return ConnectionErrorf("%v", err)
		}
	}
	f.pendingData += n
	return nil
}

// connOnRead updates the connection level states when the application consumes data.
func (f *inFlow) connOnRead(n uint32) uint32 {
	if n == 0 || f.conn != nil {
		return 0
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.pendingData -= n
	f.pendingUpdate += n
	if f.pendingUpdate >= f.limit/4 {
		ret := f.pendingUpdate
		f.pendingUpdate = 0
		return ret
	}
	return 0
}

// onRead is invoked when the application reads the data. It returns the window updates
// for both stream and connection level.
func (f *inFlow) onRead(n uint32) (swu, cwu uint32) {
	if n == 0 {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.pendingData == 0 {
		// pendingData has been adjusted by restoreConn.
		return
	}
	f.pendingData -= n
	f.pendingUpdate += n
	if f.pendingUpdate >= f.limit/4 {
		swu = f.pendingUpdate
		f.pendingUpdate = 0
	}
	cwu = f.conn.connOnRead(n)
	return
}

// restoreConn is invoked when a stream is terminated. It removes its stake in
// the connection-level flow and resets its own state.
func (f *inFlow) restoreConn() uint32 {
	if f.conn == nil {
		return 0
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	n := f.pendingData
	f.pendingData = 0
	f.pendingUpdate = 0
	return f.conn.connOnRead(n)
}
