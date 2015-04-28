// Copyright 2014 The bufs Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bufs implements a simple buffer cache.
//
// The intended use scheme is like:
//
//	type Foo struct {
//		buffers bufs.Buffers
//		...
//	}
//
//	// Bar can call Qux, but not the other way around (in this example).
//	const maxFooDepth = 2
//
//	func NewFoo() *Foo {
//		return &Foo{buffers: bufs.New(maxFooDepth), ...}
//	}
//
//	func (f *Foo) Bar(n int) {
//		buf := f.buffers.Alloc(n) // needed locally for computation and/or I/O
//		defer f.buffers.Free()
//		...
//		f.Qux(whatever)
//	}
//
//	func (f *Foo) Qux(n int) {
//		buf := f.buffers.Alloc(n) // needed locally for computation and/or I/O
//		defer f.buffers.Free()
//		...
//	}
//
// The whole idea behind 'bufs' is that when calling e.g. Foo.Bar N times, then
// normally, without using 'bufs', there will be 2*N (in this example) []byte
// buffers allocated.  While using 'bufs', only 2 buffers (in this example)
// will ever be created. For large N it can be a substantial difference.
//
// It's not a good idea to use Buffers to cache too big buffers. The cost of
// having a cached buffer is that the buffer is naturally not eligible for
// garbage collection.  Of course, that holds only while the Foo instance is
// reachable, in the above example.
//
// The buffer count limit is intentionally "hard" (read panicking), although
// configurable in New().  The rationale is to prevent recursive calls, using
// Alloc, to cause excessive, "static" memory consumption. Tune the limit
// carefully or do not use Buffers from within [mutually] recursive functions
// where the nesting depth is not realistically bounded to some rather small
// number.
//
// Buffers cannot guarantee improvements to you program performance. There may
// be a gain in case where they fit well. Firm grasp on what your code is
// actually doing, when and in what order is essential to proper use of
// Buffers. It's _highly_ recommended to first do profiling and memory
// profiling before even thinking about using 'bufs'. The real world example,
// and cause for this package, was a first correct, yet no optimizations done
// version of a program; producing few MB of useful data while allocating 20+GB
// of memory.  Of course the garbage collector properly kicked in, yet the
// memory abuse caused ~80+% of run time to be spent memory management.  The
// program _was_ expected to be slow in its still development phase, but the
// bottleneck was guessed to be in I/O.  Actually the hard disk was waiting for
// the billions bytes being allocated and zeroed. Garbage collect on low
// memory, rinse and repeat.
//
// In the provided tests, TestFoo and TestFooBufs do the same simulated work,
// except the later uses Buffers while the former does not. Suggested test runs
// which show the differences:
//
//	$ go test -bench . -benchmem
//
//	or
//
//	$ go test -c
//	$ ./bufs.test -test.v -test.run Foo -test.memprofile mem.out -test.memprofilerate 1
//	$ go tool pprof bufs.test mem.out --alloc_space --nodefraction 0.0001 --edgefraction 0 -web
//	$ # Note: Foo vs FooBufs allocated memory is in hundreds of MBs vs 8 kB.
//
//	or
//
//	$ make demo # same as all of the above
//
//
// NOTE: Alloc/Free calls must be properly nested in the same way as in for
// example BeginTransaction/EndTransaction pairs. If your code can panic then
// the pairing should be enforced by deferred calls.
//
// NOTE: Buffers objects do not allocate any space until requested by Alloc,
// the mechanism works on demand only.
//
// FAQ: Why the 'bufs' package name?
//
// Package name 'bufs' was intentionally chosen instead of the perhaps more
// conventional 'buf'. There are already too many 'buf' named things in the
// code out there and that'll be a source of a lot of trouble. It's a bit
// similar situation as in the case of package "strings" (not "string").
package bufs

import (
	"errors"
	"sort"
	"sync"
)

// Buffers type represents a buffer ([]byte) cache.
//
// NOTE: Do not modify Buffers directly, use only its methods. Do not create
// additional values (copies) of Buffers, that'll break its functionality. Use
// a pointer instead to refer to a single instance from different
// places/scopes.
type Buffers [][]byte

// New returns a newly created instance of Buffers with a maximum capacity of n
// buffers.
//
// NOTE: 'bufs.New(n)' is the same as 'make(bufs.Buffers, n)'.
func New(n int) Buffers {
	return make(Buffers, n)
}

// Alloc will return a buffer such that len(r) == n. It will firstly try to
// find an existing and unused buffer of big enough size. Only when there is no
// such, then one of the buffer slots is reallocated to a bigger size.
//
// It's okay to use append with buffers returned by Alloc. But it can cause
// allocation in that case and will again be producing load for the garbage
// collector. The best use of Alloc is for I/O buffers where the needed size of
// the buffer is figured out at some point of the code path in a 'final size'
// sense. Another real world example are compression/decompression buffers.
//
// NOTE: The buffer returned by Alloc _is not_ zeroed. That's okay for e.g.
// passing a buffer to io.Reader. If you need a zeroed buffer use Calloc.
//
// NOTE: Buffers returned from Alloc _must not_ be exposed/returned to your
// clients.  Those buffers are intended to be used strictly internally, within
// the methods of some "object".
//
// NOTE: Alloc will panic if there are no buffers (buffer slots) left.
func (p *Buffers) Alloc(n int) (r []byte) {
	b := *p
	if len(b) == 0 {
		panic(errors.New("Buffers.Alloc: out of buffers"))
	}

	biggest, best, biggestI, bestI := -1, -1, -1, -1
	for i, v := range b {
		//ln := len(v)
		// The above was correct, buts it's just confusing. It worked
		// because not the buffers, but slices of them are returned in
		// the 'if best >= n' code path.
		ln := cap(v)

		if ln >= biggest {
			biggest, biggestI = ln, i
		}

		if ln >= n && (bestI < 0 || best > ln) {
			best, bestI = ln, i
			if ln == n {
				break
			}
		}
	}

	last := len(b) - 1
	if best >= n {
		r = b[bestI]
		b[last], b[bestI] = b[bestI], b[last]
		*p = b[:last]
		return r[:n]
	}

	r = make([]byte, n, overCommit(n))
	b[biggestI] = r
	b[last], b[biggestI] = b[biggestI], b[last]
	*p = b[:last]
	return
}

// Calloc will acquire a buffer using Alloc and then clears it to zeros. The
// zeroing goes up to n, not cap(r).
func (p *Buffers) Calloc(n int) (r []byte) {
	r = p.Alloc(n)
	for i := range r {
		r[i] = 0
	}
	return
}

// Free makes the lastly allocated by Alloc buffer free (available) again for
// Alloc.
//
// NOTE: Improper Free invocations, like in the sequence {New, Alloc, Free,
// Free}, will panic.
func (p *Buffers) Free() {
	b := *p
	b = b[:len(b)+1]
	*p = b
}

// Stats reports memory consumed by Buffers, without accounting for some
// (smallish) additional overhead.
func (p *Buffers) Stats() (bytes int) {
	b := *p
	b = b[:cap(b)]
	for _, v := range b {
		bytes += cap(v)
	}
	return
}

// Cache caches buffers ([]byte). A zero value of Cache is ready for use.
//
// NOTE: Do not modify a Cache directly, use only its methods. Do not create
// additional values (copies) of a Cache, that'll break its functionality. Use
// a pointer instead to refer to a single instance from different
// places/scopes.
type Cache [][]byte

// Get returns a buffer ([]byte) of length n. If no such buffer is cached then
// a biggest cached buffer is resized to have length n and returned. If there
// are no cached items at all, Get returns a newly allocated buffer.
//
// In other words the cache policy is:
//
// - If the cache is empty, the buffer must be newly created and returned.
// Cache remains empty.
//
// - If a buffer of sufficient size is found in the cache, remove it from the
// cache and return it.
//
// - Otherwise the cache is non empty, but no cached buffer is big enough.
// Enlarge the biggest cached buffer, remove it from the cache and return it.
// This provide cached buffers size adjustment based on demand.
//
// In short, if the cache is not empty, Get guarantees to make it always one
// item less.  This rules prevent uncontrolled cache grow in some scenarios.
// The older policy was not preventing that. Another advantage is better cached
// buffers sizes "auto tuning", although not in every possible use case.
//
// NOTE: The buffer returned by Get _is not guaranteed_ to be zeroed. That's
// okay for e.g.  passing a buffer to io.Reader. If you need a zeroed buffer
// use Cget.
func (c *Cache) Get(n int) []byte {
	r, _ := c.get(n)
	return r
}

func (c *Cache) get(n int) (r []byte, isZeroed bool) {
	s := *c
	lens := len(s)
	if lens == 0 {
		r, isZeroed = make([]byte, n, overCommit(n)), true
		return
	}

	i := sort.Search(lens, func(x int) bool { return len(s[x]) >= n })
	if i == lens {
		i--
		s[i] = make([]byte, n, overCommit(n))
	}
	r = s[i][:n]
	copy(s[i:], s[i+1:])
	s[lens-1] = nil
	s = s[:lens-1]
	*c = s
	return r, false
}

// Cget will acquire a buffer using Get and then clears it to zeros. The
// zeroing goes up to n, not cap(r).
func (c *Cache) Cget(n int) (r []byte) {
	r, ok := c.get(n)
	if ok {
		return
	}

	for i := range r {
		r[i] = 0
	}
	return
}

// Put caches b for possible later reuse (via Get). No other references to b's
// backing array may exist. Otherwise a big mess is sooner or later inevitable.
func (c *Cache) Put(b []byte) {
	b = b[:cap(b)]
	lenb := len(b)
	if lenb == 0 {
		return
	}

	s := *c
	lens := len(s)
	i := sort.Search(lens, func(x int) bool { return len(s[x]) >= lenb })
	s = append(s, nil)
	copy(s[i+1:], s[i:])
	s[i] = b
	*c = s
	return
}

// Stats reports memory consumed by a Cache, without accounting for some
// (smallish) additional overhead. 'n' is the number of cached buffers, bytes
// is their combined capacity.
func (c Cache) Stats() (n, bytes int) {
	n = len(c)
	for _, v := range c {
		bytes += cap(v)
	}
	return
}

// CCache is a Cache which is safe for concurrent use by multiple goroutines.
type CCache struct {
	c  Cache
	mu sync.Mutex
}

// Get returns a buffer ([]byte) of length n. If no such buffer is cached then
// a biggest cached buffer is resized to have length n and returned. If there
// are no cached items at all, Get returns a newly allocated buffer.
//
// In other words the cache policy is:
//
// - If the cache is empty, the buffer must be newly created and returned.
// Cache remains empty.
//
// - If a buffer of sufficient size is found in the cache, remove it from the
// cache and return it.
//
// - Otherwise the cache is non empty, but no cached buffer is big enough.
// Enlarge the biggest cached buffer, remove it from the cache and return it.
// This provide cached buffers size adjustment based on demand.
//
// In short, if the cache is not empty, Get guarantees to make it always one
// item less.  This rules prevent uncontrolled cache grow in some scenarios.
// The older policy was not preventing that. Another advantage is better cached
// buffers sizes "auto tuning", although not in every possible use case.
//
// NOTE: The buffer returned by Get _is not guaranteed_ to be zeroed. That's
// okay for e.g.  passing a buffer to io.Reader. If you need a zeroed buffer
// use Cget.
func (c *CCache) Get(n int) []byte {
	c.mu.Lock()
	r, _ := c.c.get(n)
	c.mu.Unlock()
	return r
}

// Cget will acquire a buffer using Get and then clears it to zeros. The
// zeroing goes up to n, not cap(r).
func (c *CCache) Cget(n int) (r []byte) {
	c.mu.Lock()
	r = c.c.Cget(n)
	c.mu.Unlock()
	return
}

// Put caches b for possible later reuse (via Get). No other references to b's
// backing array may exist. Otherwise a big mess is sooner or later inevitable.
func (c *CCache) Put(b []byte) {
	c.mu.Lock()
	c.c.Put(b)
	c.mu.Unlock()
}

// Stats reports memory consumed by a Cache, without accounting for some
// (smallish) additional overhead. 'n' is the number of cached buffers, bytes
// is their combined capacity.
func (c *CCache) Stats() (n, bytes int) {
	c.mu.Lock()
	n, bytes = c.c.Stats()
	c.mu.Unlock()
	return
}

// GCache is a ready to use global instance of a CCache.
var GCache CCache

func overCommit(n int) int {
	switch {
	case n < 8:
		return 8
	case n < 1e5:
		return 2 * n
	case n < 1e6:
		return 3 * n / 2
	default:
		return n
	}
}
