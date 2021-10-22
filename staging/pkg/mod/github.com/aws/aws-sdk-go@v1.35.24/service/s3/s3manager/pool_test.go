// +build go1.7

package s3manager

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
)

func TestMaxSlicePool(t *testing.T) {
	pool := newMaxSlicePool(0)

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// increase pool capacity by 2
			pool.ModifyCapacity(2)

			// remove 2 items
			bsOne, err := pool.Get(context.Background())
			if err != nil {
				t.Errorf("failed to get slice from pool: %v", err)
			}
			bsTwo, err := pool.Get(context.Background())
			if err != nil {
				t.Errorf("failed to get slice from pool: %v", err)
			}

			done := make(chan struct{})
			go func() {
				defer close(done)

				// attempt to remove a 3rd in parallel
				bs, err := pool.Get(context.Background())
				if err != nil {
					t.Errorf("failed to get slice from pool: %v", err)
				}
				pool.Put(bs)

				// attempt to remove a 4th that has been canceled
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				bs, err = pool.Get(ctx)
				if err == nil {
					pool.Put(bs)
					t.Errorf("expected no slice to be returned")
					return
				}
			}()

			pool.Put(bsOne)

			<-done

			pool.ModifyCapacity(-1)

			pool.Put(bsTwo)

			pool.ModifyCapacity(-1)

			// any excess returns should drop
			rando := make([]byte, 0)
			pool.Put(&rando)
		}()
	}
	wg.Wait()

	if e, a := 0, len(pool.slices); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 0, len(pool.allocations); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 0, pool.max; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	_, err := pool.Get(context.Background())
	if err == nil {
		t.Errorf("expected error on zero capacity pool")
	}

	pool.Close()
}

func TestPoolShouldPreferAllocatedSlicesOverNewAllocations(t *testing.T) {
	pool := newMaxSlicePool(0)
	defer pool.Close()

	// Prepare pool: make it so that pool contains 1 allocated slice and 1 allocation permit
	pool.ModifyCapacity(2)
	initialSlice, err := pool.Get(context.Background())
	if err != nil {
		t.Errorf("failed to get slice from pool: %v", err)
	}
	pool.Put(initialSlice)

	for i := 0; i < 100; i++ {
		newSlice, err := pool.Get(context.Background())
		if err != nil {
			t.Errorf("failed to get slice from pool: %v", err)
			return
		}

		if newSlice != initialSlice {
			t.Errorf("pool allocated a new slice despite it having pre-allocated one")
			return
		}
		pool.Put(newSlice)
	}
}

type recordedPartPool struct {
	recordedAllocs      uint64
	recordedGets        uint64
	recordedOutstanding int64
	*maxSlicePool
}

func newRecordedPartPool(sliceSize int64) *recordedPartPool {
	sp := newMaxSlicePool(sliceSize)

	rp := &recordedPartPool{}

	allocator := sp.allocator
	sp.allocator = func() *[]byte {
		atomic.AddUint64(&rp.recordedAllocs, 1)
		return allocator()
	}

	rp.maxSlicePool = sp

	return rp
}

func (r *recordedPartPool) Get(ctx aws.Context) (*[]byte, error) {
	atomic.AddUint64(&r.recordedGets, 1)
	atomic.AddInt64(&r.recordedOutstanding, 1)
	return r.maxSlicePool.Get(ctx)
}

func (r *recordedPartPool) Put(b *[]byte) {
	atomic.AddInt64(&r.recordedOutstanding, -1)
	r.maxSlicePool.Put(b)
}

func swapByteSlicePool(f func(sliceSize int64) byteSlicePool) func() {
	orig := newByteSlicePool

	newByteSlicePool = f

	return func() {
		newByteSlicePool = orig
	}
}

type syncSlicePool struct {
	sync.Pool
	sliceSize int64
}

func newSyncSlicePool(sliceSize int64) *syncSlicePool {
	p := &syncSlicePool{sliceSize: sliceSize}
	p.New = func() interface{} {
		bs := make([]byte, p.sliceSize)
		return &bs
	}
	return p
}

func (s *syncSlicePool) Get(ctx aws.Context) (*[]byte, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		return s.Pool.Get().(*[]byte), nil
	}
}

func (s *syncSlicePool) Put(bs *[]byte) {
	s.Pool.Put(bs)
}

func (s *syncSlicePool) ModifyCapacity(_ int) {
	return
}

func (s *syncSlicePool) SliceSize() int64 {
	return s.sliceSize
}

func (s *syncSlicePool) Close() {
	return
}
