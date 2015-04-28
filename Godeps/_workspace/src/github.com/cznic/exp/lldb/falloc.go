// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The storage space management.

package lldb

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/cznic/bufs"
	"github.com/cznic/mathutil"
	"github.com/cznic/zappy"
)

const (
	maxBuf = maxRq + 20 // bufs,Buffers.Alloc
)

// Options are passed to the NewAllocator to amend some configuration.  The
// compatibility promise is the same as of struct types in the Go standard
// library - introducing changes can be made only by adding new exported
// fields, which is backward compatible as long as client code uses field names
// to assign values of imported struct types literals.
//
// NOTE: No options are currently defined.
type Options struct{}

// AllocStats record statistics about a Filer. It can be optionally filled by
// Allocator.Verify, if successful.
type AllocStats struct {
	Handles     int64           // total valid handles in use
	Compression int64           // number of compressed blocks
	TotalAtoms  int64           // total number of atoms == AllocAtoms + FreeAtoms
	AllocBytes  int64           // bytes allocated (after decompression, if/where used)
	AllocAtoms  int64           // atoms allocated/used, including relocation atoms
	Relocations int64           // number of relocated used blocks
	FreeAtoms   int64           // atoms unused
	AllocMap    map[int64]int64 // allocated block size in atoms -> count of such blocks
	FreeMap     map[int64]int64 // free block size in atoms -> count of such blocks
}

/*

Allocator implements "raw" storage space management (allocation and
deallocation) for a low level of a DB engine.  The storage is an abstraction
provided by a Filer.

The terms MUST or MUST NOT, if/where used in the documentation of Allocator,
written in all caps as seen here, are a requirement for any possible
alternative implementations aiming for compatibility with this one.

Filer file

A Filer file, or simply 'file', is a linear, contiguous sequence of blocks.
Blocks may be either free (currently unused) or allocated (currently used).
Some blocks may eventually become virtual in a sense as they may not be
realized in the storage (sparse files).

Free Lists Table

File starts with a FLT. This table records heads of 14 doubly linked free
lists. The zero based index (I) vs minimal size of free blocks in that list,
except the last one which registers free blocks of size 4112+ atoms:

	MinSize == 2^I

	For example 0 -> 1, 1 -> 2, ... 12 -> 4096.

Each entry in the FLT is 8 bytes in netwtork order, MSB MUST be zero, ie. the
slot value is effectively only 7 bytes. The value is the handle of the head of
the respective doubly linked free list. The FLT size is 14*8 == 112(0x70)
bytes. If the free blocks list for any particular size is empty, the respective
FLT slot is zero. Sizes of free blocks in one list MUST NOT overlap with sizes
of free lists in other list. For example, even though a free block of size 2
technically is of minimal size >= 1, it MUST NOT be put to the list for slot 0
(minimal size 1), but in slot 1( minimal size 2).

	slot 0:		sizes [1, 2)
	slot 1:		sizes [2, 4)
	slot 2:		sizes [4, 8)
	...
	slot 11:	sizes [2048, 4096)
	slot 12:	sizes [4096, 4112)
	slot 13:	sizes [4112, inf)

The last FLT slot collects all free blocks bigger than its minimal size. That
still respects the 'no overlap' invariant.

File blocks

A block is a linear, contiguous sequence of atoms. The first and last atoms of
a block provide information about, for example, whether the block is free or
used, what is the size of the block, etc.  Details are discussed elsewhere. The
first block of a file starts immediately after FLT, ie. at file offset
112(0x70).

Block atoms

An atom is a fixed size piece of a block (and thus of a file too); it is 16
bytes long. A consequence is that for a valid file:

 filesize == 0 (mod 16)

The first atom of the first block is considered to be atom #1.

Block handles

A handle is an integer referring to a block. The reference is the number of the
atom the block starts with. Put in other way:

 handle == offset/16 - 6
 offset == 16 * (handle + 6)

`offset` is the offset of the first byte of the block, measured in bytes
- as in fseek(3). Handle has type `int64`, but only the lower 7 bytes may be
nonzero while referring to a block, both in code as well as when persisted in
the the file's internal bookkeeping structures - see 'Block types' bellow. So a
handle is effectively only `uint56`.  This also means that the maximum usable
size of a file is 2^56 atoms.  That is 2^60 bytes == 1 exabyte (10^18 bytes).

Nil handles

A handle with numeric value of '0' refers to no block.

Zero padding

A padding is used to round-up a block size to be a whole number of atoms. Any
padding, if present, MUST be all zero bytes. Note that the size of padding is
in [0, 15].

Content wiping

When a block is deallocated, its data content is not wiped as the added
overhead may be substantial while not necessarily needed. Client code should
however overwrite the content of any block having sensitive data with eg. zeros
(good compression) - before deallocating the block.

Block tags

Every block is tagged in its first byte (a head tag) and last byte (tail tag).
Block types are:

 1. Short content used block (head tags 0x00-0xFB)
 2. Long content used block (head tag 0xFC)
 3. Relocated used block (head tag 0xFD)
 4. Short, single atom, free block (head tag 0xFE)
 5. Long free block (head tag 0xFF)

Note: Relocated used block, 3. above (head tag 0xFD) MUST NOT refer to blocks
other then 1. or 2. above (head tags 0x00-0xFC).

Content blocks

Used blocks (head tags 0x00-0xFC) tail tag distinguish used/unused block and if
content is compressed or not.

Content compression

The tail flag of an used block is one of

	CC == 0 // Content is not compressed.
	CC == 1 // Content is in zappy compression format.

If compression of written content is enabled, there are two cases: If
compressed size < original size then the compressed content should be written
if it will save at least one atom of the block. If compressed size >= original
size then the compressed content should not be used.

It's recommended to use compression. For example the BTrees implementation
assumes compression is used. Using compression may cause a slowdown in some
cases while it may as well cause a speedup.

Short content block

Short content block carries content of length between N == 0(0x00) and N ==
251(0xFB) bytes.

	|<-first atom start  ...  last atom end->|
	+---++--   ...   --+--   ...   --++------+
	| 0 ||    1...     |  0x*...0x*E || 0x*F |
	+---++--   ...   --+--   ...   --++------+
	| N ||   content   |   padding   ||  CC  |
	+---++--   ...   --+--   ...   --++------+

	A == (N+1)/16 + 1        // The number of atoms in the block [1, 16]
	padding == 15 - (N+1)%16 // Length of the zero padding

Long content block

Long content block carries content of length between N == 252(0xFC) and N ==
65787(0x100FB) bytes.

	|<-first atom start    ...     last atom end->|
	+------++------+-- ... --+--  ...   --++------+
	|  0   || 1..2 |   3...  | 0x*...0x*E || 0x*F |
	+------++------+-- ... --+--  ...   --++------+
	| 0xFC ||  M   | content |  padding   ||  CC  |
	+------++------+-- ... --+--  ...   --++------+

	A == (N+3)/16 + 1        // The number of atoms in the block [16, 4112]
	M == N % 0x10000         // Stored as 2 bytes in network byte order
	padding == 15 - (N+3)%16 // Length of the zero padding

Relocated used block

Relocated block allows to permanently assign a handle to some content and
resize the content anytime afterwards without having to update all the possible
existing references; the handle can be constant while the content size may be
dynamic. When relocating a block, any space left by the original block content,
above this single atom block, MUST be reclaimed.

Relocations MUST point only to a used short or long block == blocks with tags
0x00...0xFC.

	+------++------+---------++----+
	|  0   || 1..7 | 8...14  || 15 |
	+------++------+---------++----+
	| 0xFD ||  H   | padding || 0  |
	+------++------+---------++----+

H is the handle of the relocated block in network byte order.

Free blocks

Free blocks are the result of space deallocation. Free blocks are organized in
one or more doubly linked lists, abstracted by the FLT interface. Free blocks
MUST be "registered" by putting them in such list. Allocator MUST reuse a big
enough free block, if such exists, before growing the file size. When a free
block is created by deallocation or reallocation it MUST be joined with any
adjacently existing free blocks before "registering". If the resulting free
block is now a last block of a file, the free block MUST be discarded and the
file size MUST be truncated accordingly instead. Put differently, there MUST
NOT ever be a free block at the file end.

A single free atom

Is an unused block of size 1 atom.

	+------++------+--------++------+
	|  0   || 1..7 | 8...14 ||  15  |
	+------++------+--------++------+
	| 0xFE ||  P   |   N    || 0xFE |
	+------++------+--------++------+

P and N, stored in network byte order, are the previous and next free block
handles in the doubly linked list to which this free block belongs.

A long unused block

Is an unused block of size > 1 atom.

	+------++------+-------+---------+- ... -+----------++------+
	|  0   || 1..7 | 8..14 | 15...21 |       | Z-7..Z-1 ||  Z   |
	+------++------+-------+---------+- ... -+----------++------+
	| 0xFF ||  S   |   P   |    N    | Leak  |    S     || 0xFF |
	+------++------+-------+---------+- ... -+----------++------+

	Z == 16 * S - 1

S is the size of this unused block in atoms. P and N are the previous and next
free block handles in the doubly linked list to which this free block belongs.
Leak contains any data the block had before deallocating this block.  See also
the subtitle 'Content wiping' above. S, P and N are stored in network byte
order. Large free blocks may trigger a consideration of file hole punching of
the Leak field - for some value of 'large'.

Note: Allocator methods vs CRUD[1]:

	Alloc	[C]reate
	Get	[R]ead
	Realloc	[U]pdate
	Free	[D]elete

Note: No Allocator method returns io.EOF.

  [1]: http://en.wikipedia.org/wiki/Create,_read,_update_and_delete

*/
type Allocator struct {
	f        Filer
	flt      flt
	Compress bool // enables content compression
	cache    cache
	m        map[int64]*node
	lru      lst
	expHit   int64
	expMiss  int64
	cacheSz  int
	hit      uint16
	miss     uint16
}

// NewAllocator returns a new Allocator. To open an existing file, pass its
// Filer. To create a "new" file, pass a Filer which file is of zero size.
func NewAllocator(f Filer, opts *Options) (a *Allocator, err error) {
	if opts == nil { // Enforce *Options is always passed
		return nil, errors.New("NewAllocator: nil opts passed")
	}

	a = &Allocator{
		f:       f,
		cacheSz: 10,
	}

	a.cinit()
	switch x := f.(type) {
	case *RollbackFiler:
		x.afterRollback = func() error {
			a.cinit()
			return a.flt.load(a.f, 0)
		}
	case *ACIDFiler0:
		x.RollbackFiler.afterRollback = func() error {
			a.cinit()
			return a.flt.load(a.f, 0)
		}
	}

	sz, err := f.Size()
	if err != nil {
		return
	}

	a.flt.init()
	if sz == 0 {
		var b [fltSz]byte
		if err = a.f.BeginUpdate(); err != nil {
			return
		}

		if _, err = f.WriteAt(b[:], 0); err != nil {
			a.f.Rollback()
			return
		}

		return a, a.f.EndUpdate()
	}

	return a, a.flt.load(f, 0)
}

// CacheStats reports cache statistics.
//
//TODO return a struct perhaps.
func (a *Allocator) CacheStats() (buffersUsed, buffersTotal int, bytesUsed, bytesTotal, hits, misses int64) {
	buffersUsed = len(a.m)
	buffersTotal = buffersUsed + len(a.cache)
	bytesUsed = a.lru.size()
	bytesTotal = bytesUsed + a.cache.size()
	hits = a.expHit
	misses = a.expMiss
	return
}

func (a *Allocator) cinit() {
	for h, n := range a.m {
		a.cache.put(a.lru.remove(n))
		delete(a.m, h)
	}
	if a.m == nil {
		a.m = map[int64]*node{}
	}
}

func (a *Allocator) cadd(b []byte, h int64) {
	if len(a.m) < a.cacheSz {
		n := a.cache.get(len(b))
		n.h = h
		copy(n.b, b)
		a.m[h] = a.lru.pushFront(n)
		return
	}

	// cache full
	delete(a.m, a.cache.put(a.lru.removeBack()).h)
	n := a.cache.get(len(b))
	n.h = h
	copy(n.b, b)
	a.m[h] = a.lru.pushFront(n)
	return
}

func (a *Allocator) cfree(h int64) {
	n, ok := a.m[h]
	if !ok { // must have been evicted
		return
	}

	a.cache.put(a.lru.remove(n))
	delete(a.m, h)
}

// Alloc allocates storage space for b and returns the handle of the new block
// with content set to b or an error, if any. The returned handle is valid only
// while the block is used - until the block is deallocated. No two valid
// handles share the same value within the same Filer, but any value of a
// handle not referring to any used block may become valid any time as a result
// of Alloc.
//
// Invoking Alloc on an empty Allocator is guaranteed to return handle with
// value 1. The intended use of content of handle 1 is a root "directory" of
// other data held by an Allocator.
//
// Passing handles not obtained initially from Alloc or not anymore valid to
// any other Allocator methods can result in an irreparably corrupted database.
func (a *Allocator) Alloc(b []byte) (handle int64, err error) {
	buf := bufs.GCache.Get(zappy.MaxEncodedLen(len(b)))
	defer bufs.GCache.Put(buf)
	buf, _, cc, err := a.makeUsedBlock(buf, b)
	if err != nil {
		return
	}

	if handle, err = a.alloc(buf, cc); err == nil {
		a.cadd(b, handle)
	}
	return
}

func (a *Allocator) alloc(b []byte, cc byte) (h int64, err error) {
	rqAtoms := n2atoms(len(b))
	if h = a.flt.find(rqAtoms); h == 0 { // must grow
		var sz int64
		if sz, err = a.f.Size(); err != nil {
			return
		}

		h = off2h(sz)
		err = a.writeUsedBlock(h, cc, b)
		return
	}

	// Handle is the first item of a free blocks list.
	tag, s, prev, next, err := a.nfo(h)
	if err != nil {
		return
	}

	if tag != tagFreeShort && tag != tagFreeLong {
		err = &ErrILSEQ{Type: ErrExpFreeTag, Off: h2off(h), Arg: int64(tag)}
		return
	}

	if prev != 0 {
		err = &ErrILSEQ{Type: ErrHead, Off: h2off(h), Arg: prev}
		return
	}

	if s < int64(rqAtoms) {
		err = &ErrILSEQ{Type: ErrSmall, Arg: int64(rqAtoms), Arg2: s, Off: h2off(h)}
		return
	}

	if err = a.unlink(h, s, prev, next); err != nil {
		return
	}

	if s > int64(rqAtoms) {
		freeH := h + int64(rqAtoms)
		freeAtoms := s - int64(rqAtoms)
		if err = a.link(freeH, freeAtoms); err != nil {
			return
		}
	}
	return h, a.writeUsedBlock(h, cc, b)
}

// Free deallocates the block referred to by handle or returns an error, if
// any.
//
// After Free succeeds, handle is invalid and must not be used.
//
// Handle must have been obtained initially from Alloc and must be still valid,
// otherwise a database may get irreparably corrupted.
func (a *Allocator) Free(handle int64) (err error) {
	if handle <= 0 || handle > maxHandle {
		return &ErrINVAL{"Allocator.Free: handle out of limits", handle}
	}

	a.cfree(handle)
	return a.free(handle, 0, true)
}

func (a *Allocator) free(h, from int64, acceptRelocs bool) (err error) {
	tag, atoms, _, n, err := a.nfo(h)
	if err != nil {
		return
	}

	switch tag {
	default:
		// nop
	case tagUsedLong:
		// nop
	case tagUsedRelocated:
		if !acceptRelocs {
			return &ErrILSEQ{Type: ErrUnexpReloc, Off: h2off(h), Arg: h2off(from)}
		}

		if err = a.free(n, h, false); err != nil {
			return
		}
	case tagFreeShort, tagFreeLong:
		return &ErrINVAL{"Allocator.Free: attempt to free a free block at off", h2off(h)}
	}

	return a.free2(h, atoms)
}

func (a *Allocator) free2(h, atoms int64) (err error) {
	sz, err := a.f.Size()
	if err != nil {
		return
	}

	ltag, latoms, lp, ln, err := a.leftNfo(h)
	if err != nil {
		return
	}

	if ltag != tagFreeShort && ltag != tagFreeLong {
		latoms = 0
	}

	var rtag byte
	var ratoms, rp, rn int64

	isTail := h2off(h)+atoms*16 == sz
	if !isTail {
		if rtag, ratoms, rp, rn, err = a.nfo(h + atoms); err != nil {
			return
		}
	}

	if rtag != tagFreeShort && rtag != tagFreeLong {
		ratoms = 0
	}

	switch {
	case latoms == 0 && ratoms == 0:
		// -> isolated <-
		if isTail { // cut tail
			return a.f.Truncate(h2off(h))
		}

		return a.link(h, atoms)
	case latoms == 0 && ratoms != 0:
		// right join ->
		if err = a.unlink(h+atoms, ratoms, rp, rn); err != nil {
			return
		}

		return a.link(h, atoms+ratoms)
	case latoms != 0 && ratoms == 0:
		// <- left join
		if err = a.unlink(h-latoms, latoms, lp, ln); err != nil {
			return
		}

		if isTail {
			return a.f.Truncate(h2off(h - latoms))
		}

		return a.link(h-latoms, latoms+atoms)
	}

	// case latoms != 0 && ratoms != 0:
	// <- middle join ->
	lh, rh := h-latoms, h+atoms
	if err = a.unlink(lh, latoms, lp, ln); err != nil {
		return
	}

	// Prev unlink may have invalidated rp or rn
	if _, _, rp, rn, err = a.nfo(rh); err != nil {
		return
	}

	if err = a.unlink(rh, ratoms, rp, rn); err != nil {
		return
	}

	return a.link(h-latoms, latoms+atoms+ratoms)
}

// Add a free block h to the appropriate free list
func (a *Allocator) link(h, atoms int64) (err error) {
	if err = a.makeFree(h, atoms, 0, a.flt.head(atoms)); err != nil {
		return
	}

	return a.flt.setHead(h, atoms, a.f)
}

// Remove free block h from the free list
func (a *Allocator) unlink(h, atoms, p, n int64) (err error) {
	switch {
	case p == 0 && n == 0:
		// single item list, must be head
		return a.flt.setHead(0, atoms, a.f)
	case p == 0 && n != 0:
		// head of list (has next item[s])
		if err = a.prev(n, 0); err != nil {
			return
		}

		// new head
		return a.flt.setHead(n, atoms, a.f)
	case p != 0 && n == 0:
		// last item in list
		return a.next(p, 0)
	}
	// case p != 0 && n != 0:
	// intermediate item in a list
	if err = a.next(p, n); err != nil {
		return
	}

	return a.prev(n, p)
}

//TODO remove ?
// Return len(slice) == n, reuse src if possible.
func need(n int, src []byte) []byte {
	if cap(src) < n {
		bufs.GCache.Put(src)
		return bufs.GCache.Get(n)
	}

	return src[:n]
}

// Get returns the data content of a block referred to by handle or an error if
// any.  The returned slice may be a sub-slice of buf if buf was large enough
// to hold the entire content.  Otherwise, a newly allocated slice will be
// returned.  It is valid to pass a nil buf.
//
// If the content was stored using compression then it is transparently
// returned decompressed.
//
// Handle must have been obtained initially from Alloc and must be still valid,
// otherwise invalid data may be returned without detecting the error.
func (a *Allocator) Get(buf []byte, handle int64) (b []byte, err error) {
	buf = buf[:cap(buf)]
	if n, ok := a.m[handle]; ok {
		a.lru.moveToFront(n)
		b = need(len(n.b), buf)
		copy(b, n.b)
		a.expHit++
		a.hit++
		return
	}

	a.expMiss++
	a.miss++
	if a.miss > 10 && len(a.m) < 500 {
		if 100*a.hit/a.miss < 95 {
			a.cacheSz++
		}
		a.hit, a.miss = 0, 0
	}
	defer func(h int64) {
		if err == nil {
			a.cadd(b, h)
		}
	}(handle)

	first := bufs.GCache.Get(16)
	defer bufs.GCache.Put(first)
	relocated := false
	relocSrc := handle
reloc:
	if handle <= 0 || handle > maxHandle {
		return nil, &ErrINVAL{"Allocator.Get: handle out of limits", handle}
	}

	off := h2off(handle)
	if err = a.read(first, off); err != nil {
		return
	}

	switch tag := first[0]; tag {
	default:
		dlen := int(tag)
		atoms := n2atoms(dlen)
		switch atoms {
		case 1:
			switch tag := first[15]; tag {
			default:
				return nil, &ErrILSEQ{Type: ErrTailTag, Off: off, Arg: int64(tag)}
			case tagNotCompressed:
				b = need(dlen, buf)
				copy(b, first[1:])
				return
			case tagCompressed:
				return zappy.Decode(buf, first[1:dlen+1])
			}
		default:
			cc := bufs.GCache.Get(1)
			defer bufs.GCache.Put(cc)
			dlen := int(tag)
			atoms := n2atoms(dlen)
			tailOff := off + 16*int64(atoms) - 1
			if err = a.read(cc, tailOff); err != nil {
				return
			}

			switch tag := cc[0]; tag {
			default:
				return nil, &ErrILSEQ{Type: ErrTailTag, Off: off, Arg: int64(tag)}
			case tagNotCompressed:
				b = need(dlen, buf)
				off += 1
				if err = a.read(b, off); err != nil {
					b = buf[:0]
				}
				return
			case tagCompressed:
				zbuf := bufs.GCache.Get(dlen)
				defer bufs.GCache.Put(zbuf)
				off += 1
				if err = a.read(zbuf, off); err != nil {
					return buf[:0], err
				}

				return zappy.Decode(buf, zbuf)
			}
		}
	case 0:
		return buf[:0], nil
	case tagUsedLong:
		cc := bufs.GCache.Get(1)
		defer bufs.GCache.Put(cc)
		dlen := m2n(int(first[1])<<8 | int(first[2]))
		atoms := n2atoms(dlen)
		tailOff := off + 16*int64(atoms) - 1
		if err = a.read(cc, tailOff); err != nil {
			return
		}

		switch tag := cc[0]; tag {
		default:
			return nil, &ErrILSEQ{Type: ErrTailTag, Off: off, Arg: int64(tag)}
		case tagNotCompressed:
			b = need(dlen, buf)
			off += 3
			if err = a.read(b, off); err != nil {
				b = buf[:0]
			}
			return
		case tagCompressed:
			zbuf := bufs.GCache.Get(dlen)
			defer bufs.GCache.Put(zbuf)
			off += 3
			if err = a.read(zbuf, off); err != nil {
				return buf[:0], err
			}

			return zappy.Decode(buf, zbuf)
		}
	case tagFreeShort, tagFreeLong:
		return nil, &ErrILSEQ{Type: ErrExpUsedTag, Off: off, Arg: int64(tag)}
	case tagUsedRelocated:
		if relocated {
			return nil, &ErrILSEQ{Type: ErrUnexpReloc, Off: off, Arg: relocSrc}
		}

		handle = b2h(first[1:])
		relocated = true
		goto reloc
	}
}

var reallocTestHook bool

// Realloc sets the content of a block referred to by handle or returns an
// error, if any.
//
// Handle must have been obtained initially from Alloc and must be still valid,
// otherwise a database may get irreparably corrupted.
func (a *Allocator) Realloc(handle int64, b []byte) (err error) {
	if handle <= 0 || handle > maxHandle {
		return &ErrINVAL{"Realloc: handle out of limits", handle}
	}

	a.cfree(handle)
	if err = a.realloc(handle, b); err != nil {
		return
	}

	if reallocTestHook {
		if err = cacheAudit(a.m, &a.lru); err != nil {
			return
		}
	}

	a.cadd(b, handle)
	return
}

func (a *Allocator) realloc(handle int64, b []byte) (err error) {
	var dlen, needAtoms0 int

	b8 := bufs.GCache.Get(8)
	defer bufs.GCache.Put(b8)
	dst := bufs.GCache.Get(zappy.MaxEncodedLen(len(b)))
	defer bufs.GCache.Put(dst)
	b, needAtoms0, cc, err := a.makeUsedBlock(dst, b)
	if err != nil {
		return
	}

	needAtoms := int64(needAtoms0)
	off := h2off(handle)
	if err = a.read(b8[:], off); err != nil {
		return
	}

	switch tag := b8[0]; tag {
	default:
		dlen = int(b8[0])
	case tagUsedLong:
		dlen = m2n(int(b8[1])<<8 | int(b8[2]))
	case tagUsedRelocated:
		if err = a.free(b2h(b8[1:]), handle, false); err != nil {
			return err
		}

		dlen = 0
	case tagFreeShort, tagFreeLong:
		return &ErrINVAL{"Allocator.Realloc: invalid handle", handle}
	}

	atoms := int64(n2atoms(dlen))
retry:
	switch {
	case needAtoms < atoms:
		// in place shrink
		if err = a.writeUsedBlock(handle, cc, b); err != nil {
			return
		}

		fh, fa := handle+needAtoms, atoms-needAtoms
		sz, err := a.f.Size()
		if err != nil {
			return err
		}

		if h2off(fh)+16*fa == sz {
			return a.f.Truncate(h2off(fh))
		}

		return a.free2(fh, fa)
	case needAtoms == atoms:
		// in place replace
		return a.writeUsedBlock(handle, cc, b)
	}

	// case needAtoms > atoms:
	// in place extend or relocate
	var sz int64
	if sz, err = a.f.Size(); err != nil {
		return
	}

	off = h2off(handle)
	switch {
	case off+atoms*16 == sz:
		// relocating tail block - shortcut
		return a.writeUsedBlock(handle, cc, b)
	default:
		if off+atoms*16 < sz {
			// handle is not a tail block, check right neighbour
			rh := handle + atoms
			rtag, ratoms, p, n, e := a.nfo(rh)
			if e != nil {
				return e
			}

			if rtag == tagFreeShort || rtag == tagFreeLong {
				// Right neighbour is a free block
				if needAtoms <= atoms+ratoms {
					// can expand in place
					if err = a.unlink(rh, ratoms, p, n); err != nil {
						return
					}

					atoms += ratoms
					goto retry

				}
			}
		}
	}

	if atoms > 1 {
		if err = a.realloc(handle, nil); err != nil {
			return
		}
	}

	var newH int64
	if newH, err = a.alloc(b, cc); err != nil {
		return err
	}

	rb := bufs.GCache.Cget(16)
	defer bufs.GCache.Put(rb)
	rb[0] = tagUsedRelocated
	h2b(rb[1:], newH)
	if err = a.writeAt(rb[:], h2off(handle)); err != nil {
		return
	}

	return a.writeUsedBlock(newH, cc, b)
}

func (a *Allocator) writeAt(b []byte, off int64) (err error) {
	var n int
	if n, err = a.f.WriteAt(b, off); err != nil {
		return
	}

	if n != len(b) {
		err = io.ErrShortWrite
	}
	return
}

func (a *Allocator) write(off int64, b ...[]byte) (err error) {
	rq := 0
	for _, part := range b {
		rq += len(part)
	}
	buf := bufs.GCache.Get(rq)
	defer bufs.GCache.Put(buf)
	buf = buf[:0]
	for _, part := range b {
		buf = append(buf, part...)
	}
	return a.writeAt(buf, off)
}

func (a *Allocator) read(b []byte, off int64) (err error) {
	var rn int
	if rn, err = a.f.ReadAt(b, off); rn != len(b) {
		return &ErrILSEQ{Type: ErrOther, Off: off, More: err}
	}

	return nil
}

// nfo returns h's tag. If it's a free block then return also (s)ize (in
// atoms), (p)rev and (n)ext. If it's a used block then only (s)ize is returned
// (again in atoms). If it's a used relocate block then (n)ext is set to the
// relocation target handle.
func (a *Allocator) nfo(h int64) (tag byte, s, p, n int64, err error) {
	off := h2off(h)
	rq := int64(22)
	sz, err := a.f.Size()
	if err != nil {
		return
	}

	if off+rq >= sz {
		if rq = sz - off; rq < 15 {
			err = io.ErrUnexpectedEOF
			return
		}
	}

	buf := bufs.GCache.Get(22)
	defer bufs.GCache.Put(buf)
	if err = a.read(buf[:rq], off); err != nil {
		return
	}

	switch tag = buf[0]; tag {
	default:
		s = int64(n2atoms(int(tag)))
	case tagUsedLong:
		s = int64(n2atoms(m2n(int(buf[1])<<8 | int(buf[2]))))
	case tagFreeLong:
		if rq < 22 {
			err = io.ErrUnexpectedEOF
			return
		}

		s, p, n = b2h(buf[1:]), b2h(buf[8:]), b2h(buf[15:])
	case tagUsedRelocated:
		s, n = 1, b2h(buf[1:])
	case tagFreeShort:
		s, p, n = 1, b2h(buf[1:]), b2h(buf[8:])
	}
	return
}

// leftNfo returns nfo for h's left neighbor if h > 1 and the left neighbor is
// a free block. Otherwise all zero values are returned instead.
func (a *Allocator) leftNfo(h int64) (tag byte, s, p, n int64, err error) {
	if !(h > 1) {
		return
	}

	buf := bufs.GCache.Get(8)
	defer bufs.GCache.Put(buf)
	off := h2off(h)
	if err = a.read(buf[:], off-8); err != nil {
		return
	}

	switch tag := buf[7]; tag {
	case tagFreeShort:
		return a.nfo(h - 1)
	case tagFreeLong:
		return a.nfo(h - b2h(buf[:]))
	}
	return
}

// Set h.prev = p
func (a *Allocator) prev(h, p int64) (err error) {
	b := bufs.GCache.Get(7)
	defer bufs.GCache.Put(b)
	off := h2off(h)
	if err = a.read(b[:1], off); err != nil {
		return
	}

	switch tag := b[0]; tag {
	default:
		return &ErrILSEQ{Type: ErrExpFreeTag, Off: off, Arg: int64(tag)}
	case tagFreeShort:
		off += 1
	case tagFreeLong:
		off += 8
	}
	return a.writeAt(h2b(b[:7], p), off)
}

// Set h.next = n
func (a *Allocator) next(h, n int64) (err error) {
	b := bufs.GCache.Get(7)
	defer bufs.GCache.Put(b)
	off := h2off(h)
	if err = a.read(b[:1], off); err != nil {
		return
	}

	switch tag := b[0]; tag {
	default:
		return &ErrILSEQ{Type: ErrExpFreeTag, Off: off, Arg: int64(tag)}
	case tagFreeShort:
		off += 8
	case tagFreeLong:
		off += 15
	}
	return a.writeAt(h2b(b[:7], n), off)
}

// Make the filer image @h a free block.
func (a *Allocator) makeFree(h, atoms, prev, next int64) (err error) {
	buf := bufs.GCache.Get(22)
	defer bufs.GCache.Put(buf)
	switch {
	case atoms == 1:
		buf[0], buf[15] = tagFreeShort, tagFreeShort
		h2b(buf[1:], prev)
		h2b(buf[8:], next)
		if err = a.write(h2off(h), buf[:16]); err != nil {
			return
		}
	default:

		buf[0] = tagFreeLong
		h2b(buf[1:], atoms)
		h2b(buf[8:], prev)
		h2b(buf[15:], next)
		if err = a.write(h2off(h), buf[:22]); err != nil {
			return
		}

		h2b(buf[:], atoms)
		buf[7] = tagFreeLong
		if err = a.write(h2off(h+atoms)-8, buf[:8]); err != nil {
			return
		}
	}
	if prev != 0 {
		if err = a.next(prev, h); err != nil {
			return
		}
	}

	if next != 0 {
		err = a.prev(next, h)
	}
	return
}

func (a *Allocator) makeUsedBlock(dst []byte, b []byte) (w []byte, rqAtoms int, cc byte, err error) {
	cc = tagNotCompressed
	w = b

	var n int
	if n = len(b); n > maxRq {
		return nil, 0, 0, &ErrINVAL{"Allocator.makeUsedBlock: content size out of limits", n}
	}

	rqAtoms = n2atoms(n)
	if a.Compress && n > 14 { // attempt compression
		if dst, err = zappy.Encode(dst, b); err != nil {
			return
		}

		n2 := len(dst)
		if rqAtoms2 := n2atoms(n2); rqAtoms2 < rqAtoms { // compression saved at least a single atom
			w, n, rqAtoms, cc = dst, n2, rqAtoms2, tagCompressed
		}
	}
	return
}

func (a *Allocator) writeUsedBlock(h int64, cc byte, b []byte) (err error) {
	n := len(b)
	rq := n2atoms(n) << 4
	buf := bufs.GCache.Get(rq)
	defer bufs.GCache.Put(buf)
	switch n <= maxShort {
	case true:
		buf[0] = byte(n)
		copy(buf[1:], b)
	case false:
		m := n2m(n)
		buf[0], buf[1], buf[2] = tagUsedLong, byte(m>>8), byte(m)
		copy(buf[3:], b)
	}
	if p := n2padding(n); p != 0 {
		copy(buf[rq-1-p:], zeros[:])
	}
	buf[rq-1] = cc
	return a.writeAt(buf, h2off(h))
}

func (a *Allocator) verifyUnused(h, totalAtoms int64, tag byte, log func(error) bool, fast bool) (atoms, prev, next int64, err error) {
	switch tag {
	default:
		panic("internal error")
	case tagFreeShort:
		var b [16]byte
		off := h2off(h)
		if err = a.read(b[:], off); err != nil {
			return
		}

		if b[15] != tagFreeShort {
			err = &ErrILSEQ{Type: ErrShortFreeTailTag, Off: off, Arg: int64(b[15])}
			log(err)
			return
		}

		atoms, prev, next = 1, b2h(b[1:]), b2h(b[8:])
	case tagFreeLong:
		var b [22]byte
		off := h2off(h)
		if err = a.read(b[:], off); err != nil {
			return
		}

		atoms, prev, next = b2h(b[1:]), b2h(b[8:]), b2h(b[15:])
		if fast {
			return
		}

		if atoms < 2 {
			err = &ErrILSEQ{Type: ErrLongFreeBlkTooShort, Off: off, Arg: int64(atoms)}
			break
		}

		if h+atoms-1 > totalAtoms {
			err = &ErrILSEQ{Type: ErrLongFreeBlkTooLong, Off: off, Arg: atoms}
			break
		}

		if prev > totalAtoms {
			err = &ErrILSEQ{Type: ErrLongFreePrevBeyondEOF, Off: off, Arg: next}
			break
		}

		if next > totalAtoms {
			err = &ErrILSEQ{Type: ErrLongFreeNextBeyondEOF, Off: off, Arg: next}
			break
		}

		toff := h2off(h+atoms) - 8
		if err = a.read(b[:8], toff); err != nil {
			return
		}

		if b[7] != tag {
			err = &ErrILSEQ{Type: ErrLongFreeTailTag, Off: off, Arg: int64(b[7])}
			break
		}

		if s2 := b2h(b[:]); s2 != atoms {
			err = &ErrILSEQ{Type: ErrVerifyTailSize, Off: off, Arg: atoms, Arg2: s2}
			break
		}

	}
	if err != nil {
		log(err)
	}
	return
}

func (a *Allocator) verifyUsed(h, totalAtoms int64, tag byte, buf, ubuf []byte, log func(error) bool, fast bool) (compressed bool, dlen int, atoms, link int64, err error) {
	var (
		padding  int
		doff     int64
		padZeros [15]byte
		tailBuf  [16]byte
	)

	switch tag {
	default: // Short used
		dlen = int(tag)
		atoms = int64((dlen+1)/16) + 1
		padding = 15 - (dlen+1)%16
		doff = h2off(h) + 1
	case tagUsedLong:
		off := h2off(h) + 1
		var b2 [2]byte
		if err = a.read(b2[:], off); err != nil {
			return
		}

		dlen = m2n(int(b2[0])<<8 | int(b2[1]))
		atoms = int64((dlen+3)/16) + 1
		padding = 15 - (dlen+3)%16
		doff = h2off(h) + 3
	case tagUsedRelocated:
		dlen = 7
		atoms = 1
		padding = 7
		doff = h2off(h) + 1
	case tagFreeShort, tagFreeLong:
		panic("internal error")
	}

	if fast {
		if tag == tagUsedRelocated {
			dlen = 0
			if err = a.read(buf[:7], doff); err != nil {
				return
			}

			link = b2h(buf)
		}

		return false, dlen, atoms, link, nil
	}

	if ok := h+atoms-1 <= totalAtoms; !ok { // invalid last block
		err = &ErrILSEQ{Type: ErrVerifyUsedSpan, Off: h2off(h), Arg: atoms}
		log(err)
		return
	}

	tailsz := 1 + padding
	off := h2off(h) + 16*atoms - int64(tailsz)
	if err = a.read(tailBuf[:tailsz], off); err != nil {
		return false, 0, 0, 0, err
	}

	if ok := bytes.Equal(padZeros[:padding], tailBuf[:padding]); !ok {
		err = &ErrILSEQ{Type: ErrVerifyPadding, Off: h2off(h)}
		log(err)
		return
	}

	var cc byte
	switch cc = tailBuf[padding]; cc {
	default:
		err = &ErrILSEQ{Type: ErrTailTag, Off: h2off(h)}
		log(err)
		return
	case tagCompressed:
		compressed = true
		if tag == tagUsedRelocated {
			err = &ErrILSEQ{Type: ErrTailTag, Off: h2off(h)}
			log(err)
			return
		}

		fallthrough
	case tagNotCompressed:
		if err = a.read(buf[:dlen], doff); err != nil {
			return false, 0, 0, 0, err
		}
	}

	if cc == tagCompressed {
		if ubuf, err = zappy.Decode(ubuf, buf[:dlen]); err != nil || len(ubuf) > maxRq {
			err = &ErrILSEQ{Type: ErrDecompress, Off: h2off(h)}
			log(err)
			return
		}

		dlen = len(ubuf)
	}

	if tag == tagUsedRelocated {
		link = b2h(buf)
		if link == 0 {
			err = &ErrILSEQ{Type: ErrNullReloc, Off: h2off(h)}
			log(err)
			return
		}

		if link > totalAtoms { // invalid last block
			err = &ErrILSEQ{Type: ErrRelocBeyondEOF, Off: h2off(h), Arg: link}
			log(err)
			return
		}
	}

	return
}

var nolog = func(error) bool { return false }

// Verify attempts to find any structural errors in a Filer wrt the
// organization of it as defined by Allocator. 'bitmap' is a scratch pad for
// necessary bookkeeping and will grow to at most to Allocator's
// Filer.Size()/128 (0,78%).  Any problems found are reported to 'log' except
// non verify related errors like disk read fails etc.  If 'log' returns false
// or the error doesn't allow to (reliably) continue, the verification process
// is stopped and an error is returned from the Verify function. Passing a nil
// log works like providing a log function always returning false. Any
// non-structural errors, like for instance Filer read errors, are NOT reported
// to 'log', but returned as the Verify's return value, because Verify cannot
// proceed in such cases.  Verify returns nil only if it fully completed
// verifying Allocator's Filer without detecting any error.
//
// It is recommended to limit the number reported problems by returning false
// from 'log' after reaching some limit. Huge and corrupted DB can produce an
// overwhelming error report dataset.
//
// The verifying process will scan the whole DB at least 3 times (a trade
// between processing space and time consumed). It doesn't read the content of
// free blocks above the head/tail info bytes. If the 3rd phase detects lost
// free space, then a 4th scan (a faster one) is performed to precisely report
// all of them.
//
// If the DB/Filer to be verified is reasonably small, respective if its
// size/128 can comfortably fit within process's free memory, then it is
// recommended to consider using a MemFiler for the bit map.
//
// Statistics are returned via 'stats' if non nil. The statistics are valid
// only if Verify succeeded, ie. it didn't reported anything to log and it
// returned a nil error.
func (a *Allocator) Verify(bitmap Filer, log func(error) bool, stats *AllocStats) (err error) {
	if log == nil {
		log = nolog
	}

	n, err := bitmap.Size()
	if err != nil {
		return
	}

	if n != 0 {
		return &ErrINVAL{"Allocator.Verify: bit map initial size non zero (%d)", n}
	}

	var bits int64
	bitMask := [8]byte{1, 2, 4, 8, 16, 32, 64, 128}
	byteBuf := []byte{0}

	//DONE
	// +performance, this implementation is hopefully correct but _very_
	// naive, probably good as a prototype only. Use maybe a MemFiler
	// "cache" etc.
	// ----
	// Turns out the OS caching is as effective as it can probably get.
	bit := func(on bool, h int64) (wasOn bool, err error) {
		m := bitMask[h&7]
		off := h >> 3
		var v byte
		sz, err := bitmap.Size()
		if err != nil {
			return
		}

		if off < sz {
			if n, err := bitmap.ReadAt(byteBuf, off); n != 1 {
				return false, &ErrILSEQ{Type: ErrOther, Off: off, More: fmt.Errorf("Allocator.Verify - reading bitmap: %s", err)}
			}

			v = byteBuf[0]
		}
		switch wasOn = v&m != 0; on {
		case true:
			if !wasOn {
				v |= m
				bits++
			}
		case false:
			if wasOn {
				v ^= m
				bits--
			}
		}
		byteBuf[0] = v
		if n, err := bitmap.WriteAt(byteBuf, off); n != 1 || err != nil {
			return false, &ErrILSEQ{Type: ErrOther, Off: off, More: fmt.Errorf("Allocator.Verify - writing bitmap: %s", err)}
		}

		return
	}

	// Phase 1 - sequentially scan a.f to reliably determine block
	// boundaries. Set a bit for every block start.
	var (
		buf, ubuf       [maxRq]byte
		prevH, h, atoms int64
		wasOn           bool
		tag             byte
		st              = AllocStats{
			AllocMap: map[int64]int64{},
			FreeMap:  map[int64]int64{},
		}
		dlen int
	)

	fsz, err := a.f.Size()
	if err != nil {
		return
	}

	ok := fsz%16 == 0
	totalAtoms := (fsz - fltSz) / atomLen
	if !ok {
		err = &ErrILSEQ{Type: ErrFileSize, Name: a.f.Name(), Arg: fsz}
		log(err)
		return
	}

	st.TotalAtoms = totalAtoms
	prevTag := -1
	lastH := int64(-1)

	for h = 1; h <= totalAtoms; h += atoms {
		prevH = h // For checking last block == used

		off := h2off(h)
		if err = a.read(buf[:1], off); err != nil {
			return
		}

		switch tag = buf[0]; tag {
		default: // Short used
			fallthrough
		case tagUsedLong, tagUsedRelocated:
			var compressed bool
			if compressed, dlen, atoms, _, err = a.verifyUsed(h, totalAtoms, tag, buf[:], ubuf[:], log, false); err != nil {
				return
			}

			if compressed {
				st.Compression++
			}
			st.AllocAtoms += atoms
			switch {
			case tag == tagUsedRelocated:
				st.AllocMap[1]++
				st.Relocations++
			default:
				st.AllocMap[atoms]++
				st.AllocBytes += int64(dlen)
				st.Handles++
			}
		case tagFreeShort, tagFreeLong:
			if prevTag == tagFreeShort || prevTag == tagFreeLong {
				err = &ErrILSEQ{Type: ErrAdjacentFree, Off: h2off(lastH), Arg: off}
				log(err)
				return
			}

			if atoms, _, _, err = a.verifyUnused(h, totalAtoms, tag, log, false); err != nil {
				return
			}

			st.FreeMap[atoms]++
			st.FreeAtoms += atoms
		}

		if wasOn, err = bit(true, h); err != nil {
			return
		}

		if wasOn {
			panic("internal error")
		}

		prevTag = int(tag)
		lastH = h
	}

	if totalAtoms != 0 && (tag == tagFreeShort || tag == tagFreeLong) {
		err = &ErrILSEQ{Type: ErrFreeTailBlock, Off: h2off(prevH)}
		log(err)
		return
	}

	// Phase 2 - check used blocks, turn off the map bit for every used
	// block.
	for h = 1; h <= totalAtoms; h += atoms {
		off := h2off(h)
		if err = a.read(buf[:1], off); err != nil {
			return
		}

		var link int64
		switch tag = buf[0]; tag {
		default: // Short used
			fallthrough
		case tagUsedLong, tagUsedRelocated:
			if _, _, atoms, link, err = a.verifyUsed(h, totalAtoms, tag, buf[:], ubuf[:], log, true); err != nil {
				return
			}
		case tagFreeShort, tagFreeLong:
			if atoms, _, _, err = a.verifyUnused(h, totalAtoms, tag, log, true); err != nil {
				return
			}
		}

		turnoff := true
		switch tag {
		case tagUsedRelocated:
			if err = a.read(buf[:1], h2off(link)); err != nil {
				return
			}

			switch linkedTag := buf[0]; linkedTag {
			case tagFreeShort, tagFreeLong, tagUsedRelocated:
				err = &ErrILSEQ{Type: ErrInvalidRelocTarget, Off: off, Arg: link}
				log(err)
				return
			}

		case tagFreeShort, tagFreeLong:
			turnoff = false
		}

		if !turnoff {
			continue
		}

		if wasOn, err = bit(false, h); err != nil {
			return
		}

		if !wasOn {
			panic("internal error")
		}

	}

	// Phase 3 - using the flt check heads link to proper free blocks.  For
	// every free block, walk the list, verify the {next, prev} links and
	// turn the respective map bit off. After processing all free lists,
	// the map bits count should be zero. Otherwise there are "lost" free
	// blocks.

	var prev, next, fprev, fnext int64
	rep := a.flt

	for _, list := range rep {
		prev, next = 0, list.head
		for ; next != 0; prev, next = next, fnext {
			if wasOn, err = bit(false, next); err != nil {
				return
			}

			if !wasOn {
				err = &ErrILSEQ{Type: ErrFLT, Off: h2off(next), Arg: h}
				log(err)
				return
			}

			off := h2off(next)
			if err = a.read(buf[:1], off); err != nil {
				return
			}

			switch tag = buf[0]; tag {
			default:
				panic("internal error")
			case tagFreeShort, tagFreeLong:
				if atoms, fprev, fnext, err = a.verifyUnused(next, totalAtoms, tag, log, true); err != nil {
					return
				}

				if min := list.minSize; atoms < min {
					err = &ErrILSEQ{Type: ErrFLTSize, Off: h2off(next), Arg: atoms, Arg2: min}
					log(err)
					return
				}

				if fprev != prev {
					err = &ErrILSEQ{Type: ErrFreeChaining, Off: h2off(next)}
					log(err)
					return
				}
			}
		}

	}

	if bits == 0 { // Verify succeeded
		if stats != nil {
			*stats = st
		}
		return
	}

	// Phase 4 - if after phase 3 there are lost free blocks, report all of
	// them to 'log'
	for i := range ubuf { // setup zeros for compares
		ubuf[i] = 0
	}

	var off, lh int64
	rem, err := bitmap.Size()
	if err != nil {
		return err
	}

	for rem != 0 {
		rq := int(mathutil.MinInt64(64*1024, rem))
		var n int
		if n, err = bitmap.ReadAt(buf[:rq], off); n != rq {
			return &ErrILSEQ{Type: ErrOther, Off: off, More: fmt.Errorf("bitmap ReadAt(size %d, off %#x): %s", rq, off, err)}
		}

		if !bytes.Equal(buf[:rq], ubuf[:rq]) {
			for d, v := range buf[:rq] {
				if v != 0 {
					for i, m := range bitMask {
						if v&m != 0 {
							lh = 8*(off+int64(d)) + int64(i)
							err = &ErrILSEQ{Type: ErrLostFreeBlock, Off: h2off(lh)}
							log(err)
							return
						}
					}
				}
			}
		}

		off += int64(rq)
		rem -= int64(rq)
	}

	return
}

type fltSlot struct {
	head    int64
	minSize int64
}

func (f fltSlot) String() string {
	return fmt.Sprintf("head %#x, minSize %#x\n", f.head, f.minSize)
}

type flt [14]fltSlot

func (f *flt) init() {
	sz := 1
	for i := range *f {
		f[i].minSize, f[i].head = int64(sz), 0
		sz <<= 1
	}
	f[13].minSize = 4112
}

func (f *flt) load(fi Filer, off int64) (err error) {
	b := bufs.GCache.Get(fltSz)
	defer bufs.GCache.Put(b)
	if _, err = fi.ReadAt(b[:], off); err != nil {
		return
	}

	for i := range *f {
		off := 8*i + 1
		f[i].head = b2h(b[off:])
	}
	return
}

func (f *flt) find(rq int) (h int64) {
	switch {
	case rq < 1:
		panic(rq)
	case rq >= maxFLTRq:
		h, f[13].head = f[13].head, 0
		return
	default:
		g := f[mathutil.Log2Uint16(uint16(rq)):]
		for i := range g {
			p := &g[i]
			if rq <= int(p.minSize) {
				if h = p.head; h != 0 {
					p.head = 0
					return
				}
			}
		}
		return
	}
}

func (f *flt) head(atoms int64) (h int64) {
	switch {
	case atoms < 1:
		panic(atoms)
	case atoms >= maxFLTRq:
		return f[13].head
	default:
		lg := mathutil.Log2Uint16(uint16(atoms))
		g := f[lg:]
		for i := range g {
			if atoms < g[i+1].minSize {
				return g[i].head
			}
		}
		panic("internal error")
	}
}

func (f *flt) setHead(h, atoms int64, fi Filer) (err error) {
	switch {
	case atoms < 1:
		panic(atoms)
	case atoms >= maxFLTRq:
		b := bufs.GCache.Get(7)
		defer bufs.GCache.Put(b)
		if _, err = fi.WriteAt(h2b(b[:], h), 8*13+1); err != nil {
			return
		}

		f[13].head = h
		return
	default:
		lg := mathutil.Log2Uint16(uint16(atoms))
		g := f[lg:]
		for i := range f {
			if atoms < g[i+1].minSize {
				b := bufs.GCache.Get(7)
				defer bufs.GCache.Put(b)
				if _, err = fi.WriteAt(h2b(b[:], h), 8*int64(i+lg)+1); err != nil {
					return
				}

				g[i].head = h
				return
			}
		}
		panic("internal error")
	}
}

func (f *flt) String() string {
	a := []string{}
	for i, v := range *f {
		a = append(a, fmt.Sprintf("[%2d] %s", i, v))
	}
	return strings.Join(a, "")
}

type node struct {
	b          []byte
	h          int64
	prev, next *node
}

type cache []*node

func (c *cache) get(n int) *node {
	r, _ := c.get2(n)
	return r
}

func (c *cache) get2(n int) (r *node, isZeroed bool) {
	s := *c
	lens := len(s)
	if lens == 0 {
		return &node{b: make([]byte, n, mathutil.Min(2*n, maxBuf))}, true
	}

	i := sort.Search(lens, func(x int) bool { return len(s[x].b) >= n })
	if i == lens {
		i--
		s[i].b, isZeroed = make([]byte, n, mathutil.Min(2*n, maxBuf)), true
	}

	r = s[i]
	r.b = r.b[:n]
	copy(s[i:], s[i+1:])
	s = s[:lens-1]
	*c = s
	return
}

func (c *cache) cget(n int) (r *node) {
	r, ok := c.get2(n)
	if ok {
		return
	}

	for i := range r.b {
		r.b[i] = 0
	}
	return
}

func (c *cache) size() (sz int64) {
	for _, n := range *c {
		sz += int64(cap(n.b))
	}
	return
}

func (c *cache) put(n *node) *node {
	s := *c
	n.b = n.b[:cap(n.b)]
	lenb := len(n.b)
	lens := len(s)
	i := sort.Search(lens, func(x int) bool { return len(s[x].b) >= lenb })
	s = append(s, nil)
	copy(s[i+1:], s[i:])
	s[i] = n
	*c = s
	return n
}

type lst struct {
	front, back *node
}

func (l *lst) pushFront(n *node) *node {
	if l.front == nil {
		l.front, l.back, n.prev, n.next = n, n, nil, nil
		return n
	}

	n.prev, n.next, l.front.prev, l.front = nil, l.front, n, n
	return n
}

func (l *lst) remove(n *node) *node {
	if n.prev == nil {
		l.front = n.next
	} else {
		n.prev.next = n.next
	}
	if n.next == nil {
		l.back = n.prev
	} else {
		n.next.prev = n.prev
	}
	n.prev, n.next = nil, nil
	return n
}

func (l *lst) removeBack() *node {
	return l.remove(l.back)
}

func (l *lst) moveToFront(n *node) *node {
	return l.pushFront(l.remove(n))
}

func (l *lst) size() (sz int64) {
	for n := l.front; n != nil; n = n.next {
		sz += int64(cap(n.b))
	}
	return
}

func cacheAudit(m map[int64]*node, l *lst) (err error) {
	cnt := 0
	for h, n := range m {
		if g, e := n.h, h; g != e {
			return fmt.Errorf("cacheAudit: invalid node handle %d != %d", g, e)
		}

		if cnt, err = l.audit(n, true); err != nil {
			return
		}
	}

	if g, e := cnt, len(m); g != e {
		return fmt.Errorf("cacheAudit: invalid cache size %d != %d", g, e)
	}

	return
}

func (l *lst) audit(n *node, onList bool) (cnt int, err error) {
	if !onList && (n.prev != nil || n.next != nil) {
		return -1, fmt.Errorf("lst.audit: free node with non nil linkage")
	}

	if l.front == nil && l.back != nil || l.back == nil && l.front != nil {
		return -1, fmt.Errorf("lst.audit: one of .front/.back is nil while the other is non nil")
	}

	if l.front == l.back && l.front != nil {
		x := l.front
		if x.prev != nil || x.next != nil {
			return -1, fmt.Errorf("lst.audit: single node has non nil linkage")
		}

		if onList && x != n {
			return -1, fmt.Errorf("lst.audit: single node is alien")
		}
	}

	seen := false
	var prev *node
	x := l.front
	for x != nil {
		cnt++
		if x.prev != prev {
			return -1, fmt.Errorf("lst.audit: broken .prev linkage")
		}

		if x == n {
			seen = true
		}

		prev = x
		x = x.next
	}

	if prev != l.back {
		return -1, fmt.Errorf("lst.audit: broken .back linkage")
	}

	if onList && !seen {
		return -1, fmt.Errorf("lst.audit: node missing in list")
	}

	if !onList && seen {
		return -1, fmt.Errorf("lst.audit: node should not be on the list")
	}

	return
}
