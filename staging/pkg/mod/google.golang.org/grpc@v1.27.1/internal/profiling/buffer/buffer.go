// +build !appengine

/*
 *
 * Copyright 2019 gRPC authors.
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

// Package buffer provides a high-performant lock free implementation of a
// circular buffer used by the profiling code.
package buffer

import (
	"errors"
	"math/bits"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

type queue struct {
	// An array of pointers as references to the items stored in this queue.
	arr []unsafe.Pointer
	// The maximum number of elements this queue may store before it wraps around
	// and overwrites older values. Must be an exponent of 2.
	size uint32
	// Always size - 1. A bitwise AND is performed with this mask in place of a
	// modulo operation by the Push operation.
	mask uint32
	// Each Push operation into this queue increments the acquired counter before
	// proceeding forwarding with the actual write to arr. This counter is also
	// used by the Drain operation's drainWait subroutine to wait for all pushes
	// to complete.
	acquired uint32 // Accessed atomically.
	// After the completion of a Push operation, the written counter is
	// incremented. Also used by drainWait to wait for all pushes to complete.
	written uint32
}

// Allocates and returns a new *queue. size needs to be a exponent of two.
func newQueue(size uint32) *queue {
	return &queue{
		arr:  make([]unsafe.Pointer, size),
		size: size,
		mask: size - 1,
	}
}

// drainWait blocks the caller until all Pushes on this queue are complete.
func (q *queue) drainWait() {
	for atomic.LoadUint32(&q.acquired) != atomic.LoadUint32(&q.written) {
		runtime.Gosched()
	}
}

// A queuePair has two queues. At any given time, Pushes go into the queue
// referenced by queuePair.q. The active queue gets switched when there's a
// drain operation on the circular buffer.
type queuePair struct {
	q0 unsafe.Pointer
	q1 unsafe.Pointer
	q  unsafe.Pointer
}

// Allocates and returns a new *queuePair with its internal queues allocated.
func newQueuePair(size uint32) *queuePair {
	qp := &queuePair{}
	qp.q0 = unsafe.Pointer(newQueue(size))
	qp.q1 = unsafe.Pointer(newQueue(size))
	qp.q = qp.q0
	return qp
}

// Switches the current queue for future Pushes to proceed to the other queue
// so that there's no blocking in Push. Returns a pointer to the old queue that
// was in place before the switch.
func (qp *queuePair) switchQueues() *queue {
	// Even though we have mutual exclusion across drainers (thanks to mu.Lock in
	// drain), Push operations may access qp.q whilst we're writing to it.
	if atomic.CompareAndSwapPointer(&qp.q, qp.q0, qp.q1) {
		return (*queue)(qp.q0)
	}

	atomic.CompareAndSwapPointer(&qp.q, qp.q1, qp.q0)
	return (*queue)(qp.q1)
}

// In order to not have expensive modulo operations, we require the maximum
// number of elements in the circular buffer (N) to be an exponent of two to
// use a bitwise AND mask. Since a CircularBuffer is a collection of queuePairs
// (see below), we need to divide N; since exponents of two are only divisible
// by other exponents of two, we use floorCPUCount number of queuePairs within
// each CircularBuffer.
//
// Floor of the number of CPUs (and not the ceiling) was found to the be the
// optimal number through experiments.
func floorCPUCount() uint32 {
	floorExponent := bits.Len32(uint32(runtime.NumCPU())) - 1
	if floorExponent < 0 {
		floorExponent = 0
	}
	return 1 << uint32(floorExponent)
}

var numCircularBufferPairs = floorCPUCount()

// CircularBuffer is a lock-free data structure that supports Push and Drain
// operations.
//
// Note that CircularBuffer is built for performance more than reliability.
// That is, some Push operations may fail without retries in some situations
// (such as during a Drain operation). Order of pushes is not maintained
// either; that is, if A was pushed before B, the Drain operation may return an
// array with B before A. These restrictions are acceptable within gRPC's
// profiling, but if your use-case does not permit these relaxed constraints
// or if performance is not a primary concern, you should probably use a
// lock-based data structure such as internal/buffer.UnboundedBuffer.
type CircularBuffer struct {
	drainMutex sync.Mutex
	qp         []*queuePair
	// qpn is an monotonically incrementing counter that's used to determine
	// which queuePair a Push operation should write to. This approach's
	// performance was found to be better than writing to a random queue.
	qpn    uint32
	qpMask uint32
}

var errInvalidCircularBufferSize = errors.New("buffer size is not an exponent of two")

// NewCircularBuffer allocates a circular buffer of size size and returns a
// reference to the struct. Only circular buffers of size 2^k are allowed
// (saves us from having to do expensive modulo operations).
func NewCircularBuffer(size uint32) (*CircularBuffer, error) {
	if size&(size-1) != 0 {
		return nil, errInvalidCircularBufferSize
	}

	n := numCircularBufferPairs
	if size/numCircularBufferPairs < 8 {
		// If each circular buffer is going to hold less than a very small number
		// of items (let's say 8), using multiple circular buffers is very likely
		// wasteful. Instead, fallback to one circular buffer holding everything.
		n = 1
	}

	cb := &CircularBuffer{
		qp:     make([]*queuePair, n),
		qpMask: n - 1,
	}

	for i := uint32(0); i < n; i++ {
		cb.qp[i] = newQueuePair(size / n)
	}

	return cb, nil
}

// Push pushes an element in to the circular buffer. Guaranteed to complete in
// a finite number of steps (also lock-free). Does not guarantee that push
// order will be retained. Does not guarantee that the operation will succeed
// if a Drain operation concurrently begins execution.
func (cb *CircularBuffer) Push(x interface{}) {
	n := atomic.AddUint32(&cb.qpn, 1) & cb.qpMask
	qptr := atomic.LoadPointer(&cb.qp[n].q)
	q := (*queue)(qptr)

	acquired := atomic.AddUint32(&q.acquired, 1) - 1

	// If true, it means that we have incremented acquired before any queuePair
	// was switched, and therefore before any drainWait completion. Therefore, it
	// is safe to proceed with the Push operation on this queue. Otherwise, it
	// means that a Drain operation has begun execution, but we don't know how
	// far along the process it is. If it is past the drainWait check, it is not
	// safe to proceed with the Push operation. We choose to drop this sample
	// entirely instead of retrying, as retrying may potentially send the Push
	// operation into a spin loop (we want to guarantee completion of the Push
	// operation within a finite time). Before exiting, we increment written so
	// that any existing drainWaits can proceed.
	if atomic.LoadPointer(&cb.qp[n].q) != qptr {
		atomic.AddUint32(&q.written, 1)
		return
	}

	// At this point, we're definitely writing to the right queue. That is, one
	// of the following is true:
	//   1. No drainer is in execution on this queue.
	//   2. A drainer is in execution on this queue and it is waiting at the
	//      acquired == written barrier.
	//
	// Let's say two Pushes A and B happen on the same queue. Say A and B are
	// q.size apart; i.e. they get the same index. That is,
	//
	//   index_A = index_B
	//   acquired_A + q.size = acquired_B
	//
	// We say "B has wrapped around A" when this happens. In this case, since A
	// occurred before B, B's Push should be the final value. However, we
	// accommodate A being the final value because wrap-arounds are extremely
	// rare and accounting for them requires an additional counter and a
	// significant performance penalty. Note that the below approach never leads
	// to any data corruption.
	index := acquired & q.mask
	atomic.StorePointer(&q.arr[index], unsafe.Pointer(&x))

	// Allows any drainWait checks to proceed.
	atomic.AddUint32(&q.written, 1)
}

// Dereferences non-nil pointers from arr into result. Range of elements from
// arr that are copied is [from, to). Assumes that the result slice is already
// allocated and is large enough to hold all the elements that might be copied.
// Also assumes mutual exclusion on the array of pointers.
func dereferenceAppend(result []interface{}, arr []unsafe.Pointer, from, to uint32) []interface{} {
	for i := from; i < to; i++ {
		// We have mutual exclusion on arr, there's no need for atomics.
		x := (*interface{})(arr[i])
		if x != nil {
			result = append(result, *x)
		}
	}
	return result
}

// Drain allocates and returns an array of things Pushed in to the circular
// buffer. Push order is not maintained; that is, if B was Pushed after A,
// drain may return B at a lower index than A in the returned array.
func (cb *CircularBuffer) Drain() []interface{} {
	cb.drainMutex.Lock()

	qs := make([]*queue, len(cb.qp))
	for i := 0; i < len(cb.qp); i++ {
		qs[i] = cb.qp[i].switchQueues()
	}

	var wg sync.WaitGroup
	wg.Add(int(len(qs)))
	for i := 0; i < len(qs); i++ {
		go func(qi int) {
			qs[qi].drainWait()
			wg.Done()
		}(i)
	}
	wg.Wait()

	result := make([]interface{}, 0)
	for i := 0; i < len(qs); i++ {
		if acquired := atomic.LoadUint32(&qs[i].acquired); acquired < qs[i].size {
			result = dereferenceAppend(result, qs[i].arr, 0, acquired)
		} else {
			result = dereferenceAppend(result, qs[i].arr, 0, qs[i].size)
		}
	}

	for i := 0; i < len(qs); i++ {
		atomic.StoreUint32(&qs[i].acquired, 0)
		atomic.StoreUint32(&qs[i].written, 0)
	}

	cb.drainMutex.Unlock()
	return result
}
