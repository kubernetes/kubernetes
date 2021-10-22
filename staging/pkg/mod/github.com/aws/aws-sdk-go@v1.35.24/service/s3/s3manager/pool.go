package s3manager

import (
	"fmt"
	"sync"

	"github.com/aws/aws-sdk-go/aws"
)

type byteSlicePool interface {
	Get(aws.Context) (*[]byte, error)
	Put(*[]byte)
	ModifyCapacity(int)
	SliceSize() int64
	Close()
}

type maxSlicePool struct {
	// allocator is defined as a function pointer to allow
	// for test cases to instrument custom tracers when allocations
	// occur.
	allocator sliceAllocator

	slices         chan *[]byte
	allocations    chan struct{}
	capacityChange chan struct{}

	max       int
	sliceSize int64

	mtx sync.RWMutex
}

func newMaxSlicePool(sliceSize int64) *maxSlicePool {
	p := &maxSlicePool{sliceSize: sliceSize}
	p.allocator = p.newSlice

	return p
}

var errZeroCapacity = fmt.Errorf("get called on zero capacity pool")

func (p *maxSlicePool) Get(ctx aws.Context) (*[]byte, error) {
	// check if context is canceled before attempting to get a slice
	// this ensures priority is given to the cancel case first
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	p.mtx.RLock()

	for {
		select {
		case bs, ok := <-p.slices:
			p.mtx.RUnlock()
			if !ok {
				// attempt to get on a zero capacity pool
				return nil, errZeroCapacity
			}
			return bs, nil
		case <-ctx.Done():
			p.mtx.RUnlock()
			return nil, ctx.Err()
		default:
			// pass
		}

		select {
		case _, ok := <-p.allocations:
			p.mtx.RUnlock()
			if !ok {
				// attempt to get on a zero capacity pool
				return nil, errZeroCapacity
			}
			return p.allocator(), nil
		case <-ctx.Done():
			p.mtx.RUnlock()
			return nil, ctx.Err()
		default:
			// In the event that there are no slices or allocations available
			// This prevents some deadlock situations that can occur around sync.RWMutex
			// When a lock request occurs on ModifyCapacity, no new readers are allowed to acquire a read lock.
			// By releasing the read lock here and waiting for a notification, we prevent a deadlock situation where
			// Get could hold the read lock indefinitely waiting for capacity, ModifyCapacity is waiting for a write lock,
			// and a Put is blocked trying to get a read-lock which is blocked by ModifyCapacity.

			// Short-circuit if the pool capacity is zero.
			if p.max == 0 {
				p.mtx.RUnlock()
				return nil, errZeroCapacity
			}

			// Since we will be releasing the read-lock we need to take the reference to the channel.
			// Since channels are references we will still get notified if slices are added, or if
			// the channel is closed due to a capacity modification. This specifically avoids a data race condition
			// where ModifyCapacity both closes a channel and initializes a new one while we don't have a read-lock.
			c := p.capacityChange

			p.mtx.RUnlock()

			select {
			case _ = <-c:
				p.mtx.RLock()
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}
}

func (p *maxSlicePool) Put(bs *[]byte) {
	p.mtx.RLock()
	defer p.mtx.RUnlock()

	if p.max == 0 {
		return
	}

	select {
	case p.slices <- bs:
		p.notifyCapacity()
	default:
		// If the new channel when attempting to add the slice then we drop the slice.
		// The logic here is to prevent a deadlock situation if channel is already at max capacity.
		// Allows us to reap allocations that are returned and are no longer needed.
	}
}

func (p *maxSlicePool) ModifyCapacity(delta int) {
	if delta == 0 {
		return
	}

	p.mtx.Lock()
	defer p.mtx.Unlock()

	p.max += delta

	if p.max == 0 {
		p.empty()
		return
	}

	if p.capacityChange != nil {
		close(p.capacityChange)
	}
	p.capacityChange = make(chan struct{}, p.max)

	origAllocations := p.allocations
	p.allocations = make(chan struct{}, p.max)

	newAllocs := len(origAllocations) + delta
	for i := 0; i < newAllocs; i++ {
		p.allocations <- struct{}{}
	}

	if origAllocations != nil {
		close(origAllocations)
	}

	origSlices := p.slices
	p.slices = make(chan *[]byte, p.max)
	if origSlices == nil {
		return
	}

	close(origSlices)
	for bs := range origSlices {
		select {
		case p.slices <- bs:
		default:
			// If the new channel blocks while adding slices from the old channel
			// then we drop the slice. The logic here is to prevent a deadlock situation
			// if the new channel has a smaller capacity then the old.
		}
	}
}

func (p *maxSlicePool) notifyCapacity() {
	select {
	case p.capacityChange <- struct{}{}:
	default:
		// This *shouldn't* happen as the channel is both buffered to the max pool capacity size and is resized
		// on capacity modifications. This is just a safety to ensure that a blocking situation can't occur.
	}
}

func (p *maxSlicePool) SliceSize() int64 {
	return p.sliceSize
}

func (p *maxSlicePool) Close() {
	p.mtx.Lock()
	defer p.mtx.Unlock()
	p.empty()
}

func (p *maxSlicePool) empty() {
	p.max = 0

	if p.capacityChange != nil {
		close(p.capacityChange)
		p.capacityChange = nil
	}

	if p.allocations != nil {
		close(p.allocations)
		for range p.allocations {
			// drain channel
		}
		p.allocations = nil
	}

	if p.slices != nil {
		close(p.slices)
		for range p.slices {
			// drain channel
		}
		p.slices = nil
	}
}

func (p *maxSlicePool) newSlice() *[]byte {
	bs := make([]byte, p.sliceSize)
	return &bs
}

type returnCapacityPoolCloser struct {
	byteSlicePool
	returnCapacity int
}

func (n *returnCapacityPoolCloser) ModifyCapacity(delta int) {
	if delta > 0 {
		n.returnCapacity = -1 * delta
	}
	n.byteSlicePool.ModifyCapacity(delta)
}

func (n *returnCapacityPoolCloser) Close() {
	if n.returnCapacity < 0 {
		n.byteSlicePool.ModifyCapacity(n.returnCapacity)
	}
}

type sliceAllocator func() *[]byte

var newByteSlicePool = func(sliceSize int64) byteSlicePool {
	return newMaxSlicePool(sliceSize)
}
