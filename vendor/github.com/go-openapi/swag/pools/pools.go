// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package pools

import (
	"iter"
	"slices"
	"sync"
	"sync/atomic"
)

// Resettable is an interface for types that want to recycle a clean instance
// from a [Pool].
//
// When T (or rather *T) implements [Resettable], the pool calls Reset on an
// instance both when it is redeemed and when it is borrowed:
//
//   - on redeem, so that no references held by the instance are retained while it sits idle in the
//     pool (which would pin a reference graph alive across a GC cycle);
//   - on borrow, so that the next borrower receives a clean object regardless of how the instance
//     reached the pool.
//
// Reset must be safe to call more than once on the same instance (it runs at
// least twice per cycle).
type Resettable interface {
	Reset()
}

// resetIfResettable calls Reset on v when *T implements [Resettable].
func resetIfResettable[T any](v *T) {
	if r, ok := any(v).(Resettable); ok {
		r.Reset()
	}
}

// borrow state of a [redeemable] wrapper, used to detect double-redeem.
const (
	stateIdle     uint32 = iota // sitting in the pool (or freshly created), not checked out
	stateBorrowed               // checked out by a borrower
)

type redeemable[T any] struct {
	inner    *T
	redeemer func()
	// state guards against a double-redeem (the same wrapper Put into the pool
	// twice, which would let one object be handed to two borrowers).
	//
	// It is set to stateBorrowed on borrow and atomically flipped back to
	// stateIdle on redeem; a redeem that finds it already idle panics.
	state atomic.Uint32
}

// redeemPanic is the message raised when a slot is redeemed while already idle.
const redeemPanic = "pools: " +
	"double redeem detected (object already returned to the pool); " +
	"a borrowed object must be redeemed exactly once"

// Pool wraps a [sync.Pool] to make it available for any type.
//
// T must be the value type of the pooled object (e.g. Pool[bytes.Buffer]):
// [Pool.Borrow] returns a *T. Using a pointer type as T (e.g.
// Pool[*bytes.Buffer]) would yield a **T and is almost certainly a mistake.
type Pool[T any] struct {
	pool    sync.Pool
	tracker tracker[T] // empty (zero-cost) unless built with the poolsdebug tag
}

// PoolRedeemable wraps a [sync.Pool] to make it available for any type.
//
// It differs from [Pool] in the way objects are redeemed to the pool: borrowing
// also yields a cached redeem closure, so no closure is allocated at redeem
// time.
type PoolRedeemable[T any] struct {
	pool    sync.Pool
	tracker tracker[redeemable[T]] // empty (zero-cost) unless built with the poolsdebug tag
}

// New builds a new [Pool] to recycle allocations of type T explicitly using
// [Pool.Redeem] and the allocated pointer.
//
// Freshly allocated instances of type T are set to their zero value; like
// recycled instances they are reset (if [Resettable]) when borrowed, so
// [Pool.Borrow] always yields a clean object.
func New[T any]() *Pool[T] {
	p := &Pool[T]{}
	p.pool = sync.Pool{
		New: func() any {
			return new(T)
		},
	}
	p.tracker.register()

	return p
}

// NewRedeemable builds a new redeemable [Pool] to recycle allocations of type
// T, and use the inner redeemer to relinquish objects to the pool.
func NewRedeemable[T any]() *PoolRedeemable[T] {
	p := &PoolRedeemable[T]{}
	p.pool = sync.Pool{
		New: func() any {
			r := &redeemable[T]{inner: new(T)}
			r.redeemer = func() {
				if !r.state.CompareAndSwap(stateBorrowed, stateIdle) {
					panic(redeemPanic)
				}
				resetIfResettable(r.inner)
				p.pool.Put(r)
			}

			return r
		},
	}
	p.tracker.register()

	return p
}

// Borrow an instance from the pool.
//
// If the type implements [Resettable], the returned instance is reset before
// being handed out, so it is always clean.
func (p *Pool[T]) Borrow() *T {
	target := p.pool.Get().(*T)
	resetIfResettable(target)
	p.tracker.onBorrow(target)

	return target
}

// Redeem a borrowed instance to the pool.
//
// A nil pointer is ignored (it would otherwise corrupt the pool: a typed-nil
// boxed into an interface is not the nil interface that [sync.Pool.Put] skips).
//
// The instance is reset (if it implements [Resettable]) before being returned
// to the pool.
// After calling Redeem, the caller must drop its reference to ptr: continuing
// to use it is a use-after-redeem bug.
//
// Unlike [PoolRedeemable], this plain pool holds no per-object state, so it
// cannot detect a double-redeem of the same pointer (which corrupts the pool).
//
// Prefer [PoolRedeemable] when you want that guard, or the debug build for full
// tracking.
func (p *Pool[T]) Redeem(ptr *T) {
	if ptr == nil {
		return
	}
	p.tracker.onRedeem(ptr)
	resetIfResettable(ptr)
	p.pool.Put(ptr)
}

// BorrowWithRedeem borrows an instance from the pool and provides the
// corresponding redeem function.
//
// This is useful for instance to use with defer.
//
// The instance is reset (if it implements [Resettable]) both when borrowed and
// when the returned redeem closure is called.
// After calling the redeem closure, the caller must drop its reference to the
// returned instance.
//
// Calling the redeem closure more than once panics (see [redeemable.state]): a
// borrowed instance must be redeemed exactly once.
func (p *PoolRedeemable[T]) BorrowWithRedeem() (*T, func()) {
	container := p.pool.Get().(*redeemable[T])
	container.state.Store(stateBorrowed)
	resetIfResettable(container.inner)

	// In release builds borrowRedeemer returns container.redeemer unchanged (zero
	// cost).
	// Under the poolsdebug tag it returns a generation-stamped wrapper that tracks
	// the borrow and detects double-redeem (incl.
	//
	// ABA), foreign-redeem and leaks.
	return container.inner, p.tracker.borrowRedeemer(container, container.redeemer)
}

// Slice is a struct that wraps a slice []T.
//
// This is useful to borrow and redeem slices from a pool, without having to
// constantly manipulate pointers to the slice.
//
// The wrapper holds the authoritative slice header.
//
// Its mutating methods ([Slice.Append], [Slice.Concat], [Slice.Grow]) return
// the current backing slice for convenience, so it reads as an idiomatic []T.
//
// But the returned slice is only a snapshot of the wrapper's state at that
// moment: if you keep it and grow it yourself with the builtin append and it
// reallocates, the new backing array lives only in your local copy and is NOT
// tracked by the wrapper — it will not be recycled when the wrapper is
// redeemed (and a later borrower would get the old, smaller array).
//
// Rule of thumb: it is fine to read or pass the returned []T to a consumer; but
// if you plan to grow the slice, keep calling the wrapper's methods so the
// growth is tracked and recycled.
type Slice[T any] struct {
	length int
	inner  []T
}

// Slice returns the inner slice.
//
// Treat the result as a read-only view (for ranging or passing to a consumer),
// valid until the next mutation or redeem.
// To grow or append, use the wrapper methods so the new backing array is
// tracked and recycled (see [Slice]).
func (s *Slice[T]) Slice() []T {
	return s.inner
}

// Grow the inner slice so it can accommodate at least size more elements
// without reallocating, and return the current backing slice.
//
// Growth is tracked by the wrapper, so the enlarged backing array is recycled
// on redeem.
// See [Slice] for the caveat about growing the returned slice yourself.
func (s *Slice[T]) Grow(size int) []T {
	s.inner = slices.Grow(s.inner, size)

	return s.inner
}

func (s *Slice[T]) Len() int {
	return len(s.inner)
}

func (s *Slice[T]) Cap() int {
	return cap(s.inner)
}

// Append elements to the inner slice and return the current backing slice.
//
// This should be preferred to the append builtin if you plan that the slice will
// grow and you want the newly allocated space to be tracked and recycled.
// See [Slice] for the caveat about growing the returned slice yourself.
func (s *Slice[T]) Append(elems ...T) []T {
	s.inner = append(s.inner, elems...)

	return s.inner
}

// Concat another slice to the inner slice and return the current backing slice.
//
// Unlike [slices.Concat], this reuses the inner slice's capacity instead of
// always allocating a fresh backing array.
// See [Slice] for the caveat about growing the returned slice yourself.
func (s *Slice[T]) Concat(slice []T) []T {
	s.inner = append(s.inner, slice...)

	return s.inner
}

// IndexedElems iterates over the inner slice.
func (s *Slice[T]) IndexedElems() iter.Seq2[int, T] {
	return func(yield func(int, T) bool) {
		for i, elem := range s.inner {
			if !yield(i, elem) {
				return
			}
		}
	}
}

// Reset the inner slice to its configured initial length, keeping allocated
// capacity.
//
// All elements are zeroed, so the pool never retains stale element references
// (which would keep a referenced graph alive for slices of pointers) and so a
// [WithLength] slice is handed out clean rather than carrying data from a
// previous borrower.
func (s *Slice[T]) Reset() {
	clear(s.inner)
	if s.length > cap(s.inner) {
		s.inner = slices.Grow(s.inner[:0], s.length)
	}
	s.inner = s.inner[:s.length]
}

// Clip removes unused capacity from the inner slice.
func (s *Slice[T]) Clip() {
	s.inner = slices.Clip(s.inner)
}

// resetWithCapacity discards the current backing array and replaces it with a
// fresh one of the configured length and the given capacity.
//
// It is used by a capacity-capped pool to stop recycling an oversized backing
// array (the old array is left for the GC).
func (s *Slice[T]) resetWithCapacity(capacity int) {
	s.inner = make([]T, s.length, max(s.length, capacity))
}

// PoolSlice is a pool of [Slice[T]].
//
// [PoolSlice.BorrowWithRedeem] will return an empty inner slice by default.
// This default may be altered using [WithMinimumCapacity].
//
// Use [PoolSlice.BorrowWithSizeAndRedeem] or [Slice.Grow] to grow the capacity
// of the inner slice.
type PoolSlice[T any] struct {
	// redeemable is held as an unexported field rather than embedded, so the
	// underlying [PoolRedeemable] and its [sync.Pool] are not part of PoolSlice's
	// public surface.
	redeemable *PoolRedeemable[Slice[T]]
}

// PoolSliceOption alters the default settings to allocate new pooled slices
type PoolSliceOption func(*poolSliceOptions)

type poolSliceOptions struct {
	minCapacity int
	length      int
	maxCapacity int
}

func WithMinimumCapacity(size int) PoolSliceOption {
	return func(o *poolSliceOptions) {
		o.minCapacity = size
	}
}

// WithMaxCapacity bounds the capacity of recycled slices.
//
// When a borrowed slice has grown past size at redeem time, its (oversized)
// backing array is discarded and replaced with a fresh one sized to the minimum
// capacity, instead of being recycled.
//
// This stops the pool from accumulating large backing arrays after an
// occasional large request, keeping the steady-state memory bounded.
//
// The trade-off: a workload that genuinely needs slices larger than size will
// reallocate on every cycle.
// Set size from the high-water mark you actually expect, not below it.
// A size of 0 (the default) means no cap: grown slices are recycled as-is.
func WithMaxCapacity(size int) PoolSliceOption {
	return func(o *poolSliceOptions) {
		o.maxCapacity = size
	}
}

// WithLength ensures that the borrowed slices have a fixed given initial
// length.
//
// By default, the borrowed slices are reset to length 0.
func WithLength(size int) PoolSliceOption {
	return func(o *poolSliceOptions) {
		o.length = size
	}
}

// NewPoolSlice builds a pool to recycle slices of type []T.
func NewPoolSlice[T any](opts ...PoolSliceOption) *PoolSlice[T] {
	var o poolSliceOptions
	for _, apply := range opts {
		apply(&o)
	}

	rp := &PoolRedeemable[Slice[T]]{}
	rp.pool = sync.Pool{
		New: func() any {
			s := &redeemable[Slice[T]]{
				inner: &Slice[T]{
					length: o.length,
					inner:  make([]T, o.length, max(o.length, o.minCapacity)),
				},
			}

			s.redeemer = func() {
				if !s.state.CompareAndSwap(stateBorrowed, stateIdle) {
					panic(redeemPanic)
				}
				if o.maxCapacity > 0 && s.inner.Cap() > o.maxCapacity {
					s.inner.resetWithCapacity(o.minCapacity)
				} else {
					s.inner.Reset()
				}
				rp.pool.Put(s)
			}

			return s
		},
	}
	rp.tracker.register()

	return &PoolSlice[T]{redeemable: rp}
}

// BorrowWithRedeem returns the slice wrapper and the redeem closure to
// relinquish the allocated wrapper.
//
// The wrapper is reset (elements zeroed, length restored) both on borrow and
// when the redeem closure is called.
// Calling the redeem closure more than once panics.
func (p *PoolSlice[T]) BorrowWithRedeem() (*Slice[T], func()) {
	return p.redeemable.BorrowWithRedeem()
}

// BorrowWithSizeAndRedeem borrows a slice []T from the pool and ensures that
// its capacity is at least the provided size.
func (p *PoolSlice[T]) BorrowWithSizeAndRedeem(size int) (*Slice[T], func()) {
	s, redeem := p.BorrowWithRedeem()
	s.Grow(size)

	return s, redeem
}
