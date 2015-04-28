// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

/*

This is an mostly (WIP) conforming implementation of the "specs" in docs.go.

The main incompletness is support for only one kind of FTL, though this table kind is still per "specs".

*/

package falloc

import (
	"bytes"
	"github.com/cznic/fileutil/storage"
	"sync"
)

// Handle is a reference to a block in a file/store.
// Handle is an uint56 wrapped in an in64, i.e. the most significant byte must be always zero.
type Handle int64

// Put puts the 7 least significant bytes of h into b. The MSB of h should be zero.
func (h Handle) Put(b []byte) {
	for ofs := 6; ofs >= 0; ofs-- {
		b[ofs] = byte(h)
		h >>= 8
	}
}

// Get gets the 7 least significant bytes of h from b. The MSB of h is zeroed.
func (h *Handle) Get(b []byte) {
	var x Handle
	for ofs := 0; ofs <= 6; ofs++ {
		x = x<<8 | Handle(b[ofs])
	}
	*h = x
}

// File is a file/store with space allocation/deallocation support.
type File struct {
	f       storage.Accessor
	atoms   int64       // current file size in atom units
	canfree int64       // only blocks >= canfree can be subject to Free()
	freetab [3857]int64 // freetab[0] is unused, freetab[1] is size 1 ptr, freetab[2] is size 2 ptr, ...
	rwm     sync.RWMutex
}

func (f *File) read(b []byte, off int64) {
	if n, err := f.f.ReadAt(b, off); n != len(b) {
		panic(&ERead{f.f.Name(), off, err})
	}
}

func (f *File) write(b []byte, off int64) {
	if n, err := f.f.WriteAt(b, off); n != len(b) {
		panic(&EWrite{f.f.Name(), off, err})
	}
}

var ( // R/O
	hdr   = []byte{0x0f, 0xf1, 0xc1, 0xa1, 0xfe, 0xa5, 0x1b, 0x1e, 0, 0, 0, 0, 0, 0, 2, 0} // free lists table @2
	empty = make([]byte, 16)
	zero  = []byte{0}
	zero7 = make([]byte, 7)
)

// New returns a new File backed by store or an error if any.
// Any existing data in store are discarded.
func New(store storage.Accessor) (f *File, err error) {
	f = &File{f: store}
	return f, storage.Mutate(store, func() (err error) {
		if err = f.f.Truncate(0); err != nil {
			return &ECreate{f.f.Name(), err}
		}

		if _, err = f.Alloc(hdr[1:]); err != nil { //TODO internal panicking versions of the exported fns.
			return
		}

		if _, err = f.Alloc(nil); err != nil { // (empty) root @1
			return
		}

		b := make([]byte, 3856*14)
		for i := 1; i <= 3856; i++ {
			Handle(i).Put(b[(i-1)*14:])
		}
		if _, err = f.Alloc(b); err != nil {
			return
		}

		f.canfree = f.atoms
		return
	})
}

// Open returns a new File backed by store or an error if any.
// Store already has to be in a valid format.
func Open(store storage.Accessor) (f *File, err error) {
	defer func() {
		if e := recover(); e != nil {
			f = nil
			err = e.(error)
		}
	}()

	fi, err := store.Stat()
	if err != nil {
		panic(&EOpen{store.Name(), err})
	}

	fs := fi.Size()
	if fs&0xf != 0 {
		panic(&ESize{store.Name(), fi.Size()})
	}

	f = &File{f: store, atoms: fs >> 4}
	b := make([]byte, len(hdr))
	f.read(b, 0)
	if !bytes.Equal(b, hdr) {
		panic(&EHeader{store.Name(), b, append([]byte{}, hdr...)})
	}

	var atoms int64
	b, atoms = f.readUsed(2)
	f.canfree = atoms + 2
	ofs := 0
	var size, p Handle
	for ofs < len(b) {
		size.Get(b[ofs:])
		ofs += 7
		p.Get(b[ofs:])
		ofs += 7
		if sz, pp := int64(size), int64(p); size == 0 || size > 3856 || (pp != 0 && pp < f.canfree) || pp<<4 > fs-16 {
			panic(&EFreeList{store.Name(), sz, pp})
		}

		f.freetab[size] = int64(p)
	}
	return
}

// Accessor returns the File's underlying Accessor.
func (f *File) Accessor() storage.Accessor {
	return f.f
}

// Close closes f and returns an error if any.
func (f *File) Close() (err error) {
	return storage.Mutate(f.Accessor(), func() (err error) {
		if err = f.f.Close(); err != nil {
			err = &EClose{f.f.Name(), err}
		}
		return
	})
}

// Root returns the handle of the DB root (top level directory, ...).
func (f *File) Root() Handle {
	return 1
}

func (f *File) readUsed(atom int64) (content []byte, atoms int64) {
	b, redirected := make([]byte, 7), false
redir:
	ofs := atom << 4
	f.read(b[:1], ofs)
	switch pre := b[0]; {
	default:
		panic(&ECorrupted{f.f.Name(), ofs})
	case pre == 0x00: // Empty block
	case pre >= 1 && pre <= 237: // Short
		content = make([]byte, pre)
		f.read(content, ofs+1)
	case pre >= 0xee && pre <= 0xfb: // Short esc
		content = make([]byte, 15+16*(pre-0xee))
		f.read(content, ofs+1)
		content[len(content)-1] += 0xfe
	case pre == 0xfc: // Long
		f.read(b[:2], ofs+1)
		n := int(b[0])<<8 + int(b[1])
		switch {
		default:
			panic(&ECorrupted{f.f.Name(), ofs + 1})
		case n >= 238 && n <= 61680: // Long non esc
			content = make([]byte, n)
			f.read(content, ofs+3)
		case n >= 61681: // Long esc
			content = make([]byte, 13+16*(n-0xf0f1))
			f.read(content, ofs+3)
			content[len(content)-1] += 0xfe
		}
	case pre == 0xfd: // redir
		if redirected {
			panic(&ECorrupted{f.f.Name(), ofs})
		}

		f.read(b[:7], ofs+1)
		(*Handle)(&atom).Get(b)
		redirected = true
		goto redir
	}
	return content, rq2Atoms(len(content))
}

func (f *File) writeUsed(b []byte, atom int64) {
	n := len(b)
	switch ofs, atoms, endmark := atom<<4, rq2Atoms(n), true; {
	default:
		panic("internal error")
	case n == 0:
		f.write(empty, ofs)
	case n <= 237:
		if (n+1)&0xf == 0 { // content end == atom end
			if v := b[n-1]; v >= 0xfe { // escape
				pre := []byte{byte((16*0xee + n - 15) >> 4)}
				f.write(pre, ofs)
				f.write(b[:n-1], ofs+1)
				f.write([]byte{v - 0xfe}, ofs+atoms<<4-1)
				return
			}
			endmark = false
		}
		// non esacpe
		pre := []byte{byte(n)}
		f.write(pre, ofs)
		f.write(b, ofs+1)
		if endmark {
			f.write(zero, ofs+atoms<<4-1) // last block byte <- used block
		}
	case n > 237 && n <= 61680:
		if (n+3)&0xf == 0 { // content end == atom end
			if v := b[n-1]; v >= 0xfe { // escape
				x := (16*0xf0f1 + n - 13) >> 4
				pre := []byte{0xFC, byte(x >> 8), byte(x)}
				f.write(pre, ofs)
				f.write(b[:n-1], ofs+3)
				f.write([]byte{v - 0xfe}, ofs+atoms<<4-1)
				return
			}
			endmark = false
		}
		// non esacpe
		pre := []byte{0xfc, byte(n >> 8), byte(n)}
		f.write(pre, ofs)
		f.write(b, ofs+3)
		if endmark {
			f.write(zero, ofs+atoms<<4-1) // last block byte <- used block
		}
	}
}

func rq2Atoms(rqbytes int) (rqatoms int64) {
	if rqbytes > 237 {
		rqbytes += 2
	}
	return int64(rqbytes>>4 + 1)
}

func (f *File) extend(b []byte) (handle int64) {
	handle = f.atoms
	f.writeUsed(b, handle)
	f.atoms += rq2Atoms(len(b))
	return
}

// Alloc stores b in a newly allocated space and returns its handle and an error if any.
func (f *File) Alloc(b []byte) (handle Handle, err error) {
	err = storage.Mutate(f.Accessor(), func() (err error) {
		rqAtoms := rq2Atoms(len(b))
		if rqAtoms > 3856 {
			return &EBadRequest{f.f.Name(), len(b)}
		}

		for foundsize, foundp := range f.freetab[rqAtoms:] {
			if foundp != 0 {
				// this works only for the current unique sizes list (except the last item!)
				size := int64(foundsize) + rqAtoms
				handle = Handle(foundp)
				if size == 3856 {
					buf := make([]byte, 7)
					f.read(buf, int64(handle)<<4+15)
					(*Handle)(&size).Get(buf)
				}
				f.delFree(int64(handle), size)
				if rqAtoms < size {
					f.addFree(int64(handle)+rqAtoms, size-rqAtoms)
				}
				f.writeUsed(b, int64(handle))
				return
			}
		}

		handle = Handle(f.extend(b))
		return
	})
	return
}

// checkLeft returns the atom size of a free bleck left adjacent to block @atom.
// If that block is not free the returned size is 0.
func (f *File) checkLeft(atom int64) (size int64) {
	if atom <= f.canfree {
		return
	}

	b := make([]byte, 7)
	fp := atom << 4
	f.read(b[:1], fp-1)
	switch last := b[0]; {
	case last <= 0xfd:
		// used block
	case last == 0xfe:
		f.read(b, fp-8)
		(*Handle)(&size).Get(b)
	case last == 0xff:
		size = 1
	}
	return
}

// getInfo returns the block @atom type and size.
func (f *File) getInfo(atom int64) (pref byte, size int64) {
	b := make([]byte, 7)
	fp := atom << 4
	f.read(b[:1], fp)
	switch pref = b[0]; {
	case pref == 0: // Empty used
		size = 1
	case pref >= 1 && pref <= 237: // Short
		size = rq2Atoms(int(pref))
	case pref >= 0xee && pref <= 0xfb: // Short esc
		size = rq2Atoms(15 + 16*int(pref-0xee))
	case pref == 0xfc: // Long
		f.read(b[:2], fp+1)
		n := int(b[0])<<8 + int(b[1])
		switch {
		default:
			panic(&ECorrupted{f.f.Name(), fp + 1})
		case n >= 238 && n <= 61680: // Long non esc
			size = rq2Atoms(n)
		case n >= 61681: // Long esc
			size = rq2Atoms(13 + 16*(n-0xf0f1))
		}
	case pref == 0xfd: // reloc
		size = 1
	case pref == 0xfe:
		f.read(b, fp+15)
		(*Handle)(&size).Get(b)
	case pref == 0xff:
		size = 1
	}
	return
}

// getSize returns the atom size of the block @atom and wheter it is free.
func (f *File) getSize(atom int64) (size int64, isFree bool) {
	var typ byte
	typ, size = f.getInfo(atom)
	isFree = typ >= 0xfe
	return
}

// checkRight returns the atom size of a free bleck right adjacent to block @atom,atoms.
// If that block is not free the returned size is 0.
func (f *File) checkRight(atom, atoms int64) (size int64) {
	if atom+atoms >= f.atoms {
		return
	}

	if sz, free := f.getSize(atom + atoms); free {
		size = sz
	}
	return
}

// delFree removes the atoms@atom free block from the free block list
func (f *File) delFree(atom, atoms int64) {
	b := make([]byte, 15)
	size := int(atoms)
	if n := len(f.freetab); atoms >= int64(n) {
		size = n - 1
	}
	fp := atom << 4
	f.read(b[1:], fp+1)
	var prev, next Handle
	prev.Get(b[1:])
	next.Get(b[8:])

	switch {
	case prev == 0 && next != 0:
		next.Put(b)
		f.write(b[:7], int64(32+3+7+(size-1)*14))
		f.write(zero7, int64(next)<<4+1)
		f.freetab[size] = int64(next)
	case prev != 0 && next == 0:
		f.write(zero7, int64(prev)<<4+8)
	case prev != 0 && next != 0:
		prev.Put(b)
		f.write(b[:7], int64(next)<<4+1)
		next.Put(b)
		f.write(b[:7], int64(prev)<<4+8)
	default: // prev == 0 && next == 0:
		f.write(zero7, int64(32+3+7+(size-1)*14))
		f.freetab[size] = 0
	}
}

// addFree adds atoms@atom to the free block lists and marks it free.
func (f *File) addFree(atom, atoms int64) {
	b := make([]byte, 7)
	size := int(atoms)
	if n := len(f.freetab); atoms >= int64(n) {
		size = n - 1
	}
	head := f.freetab[size]
	if head == 0 { // empty list
		f.makeFree(0, atom, atoms, 0)
		Handle(atom).Put(b)
		f.write(b, int64(32+3+7+(size-1)*14))
		f.freetab[size] = atom
		return
	}

	Handle(atom).Put(b)
	f.write(b, head<<4+1)            // head.prev = atom
	f.makeFree(0, atom, atoms, head) // atom.next = head
	f.write(b, int64(32+3+7+(size-1)*14))
	f.freetab[size] = atom
}

// makeFree sets up the content of a free block atoms@atom, fills the prev and next links.
func (f *File) makeFree(prev, atom, atoms, next int64) {
	b := make([]byte, 23)
	fp := atom << 4
	if atoms == 1 {
		b[0] = 0xff
		Handle(prev).Put(b[1:])
		Handle(next).Put(b[8:])
		b[15] = 0xff
		f.write(b[:16], fp)
		return
	}

	b[0] = 0xfe
	Handle(prev).Put(b[1:])
	Handle(next).Put(b[8:])
	Handle(atoms).Put(b[15:])
	f.write(b[:22], fp)
	b[22] = 0xfe
	f.write(b[15:], fp+atoms<<4-8)
}

// Read reads and return the data associated with handle and an error if any.
// Passing an invalid handle to Read may return invalid data without error.
// It's like getting garbage via passing an invalid pointer to C.memcopy().
func (f *File) Read(handle Handle) (b []byte, err error) {
	defer func() {
		if e := recover(); e != nil {
			b = nil
			err = e.(error)
		}
	}()

	switch handle {
	case 0:
		panic(ENullHandle(f.f.Name()))
	case 2:
		panic(&EHandle{f.f.Name(), handle})
	default:
		b, _ = f.readUsed(int64(handle))
	}
	return
}

// Free frees space associated with handle and returns an error if any. Passing an invalid
// handle to Free or reusing handle afterwards will probably corrupt the database or provide
// invalid data on Read. It's like corrupting memory via passing an invalid pointer to C.free()
// or reusing that pointer.
func (f *File) Free(handle Handle) (err error) {
	return storage.Mutate(f.Accessor(), func() (err error) {
		atom := int64(handle)
		atoms, isFree := f.getSize(atom)
		if isFree || atom < f.canfree {
			return &EHandle{f.f.Name(), handle}
		}

		leftFree, rightFree := f.checkLeft(atom), f.checkRight(atom, atoms)
		switch {
		case leftFree != 0 && rightFree != 0:
			f.delFree(atom-leftFree, leftFree)
			f.delFree(atom+atoms, rightFree)
			f.addFree(atom-leftFree, leftFree+atoms+rightFree)
		case leftFree != 0 && rightFree == 0:
			f.delFree(atom-leftFree, leftFree)
			if atom+atoms == f.atoms { // the left free neighbour and this block together are an empy tail
				f.atoms = atom - leftFree
				f.f.Truncate(f.atoms << 4)
				return
			}

			f.addFree(atom-leftFree, leftFree+atoms)
		case leftFree == 0 && rightFree != 0:
			f.delFree(atom+atoms, rightFree)
			f.addFree(atom, atoms+rightFree)
		default: // leftFree == 0 && rightFree == 0
			if atom+atoms < f.atoms { // isolated inner block
				f.addFree(atom, atoms)
				return
			}

			f.f.Truncate(atom << 4) // isolated tail block, shrink file
			f.atoms = atom
		}
		return
	})
}

// Realloc reallocates space associted with handle to acomodate b, returns the newhandle
// newly associated with b and an error if any. If keepHandle == true then Realloc guarantees
// newhandle == handle even if the new data are larger then the previous content associated
// with handle. If !keepHandle && newhandle != handle then reusing handle will probably corrupt
// the database.
// The above effects are like corrupting memory/data via passing an invalid pointer to C.realloc().
func (f *File) Realloc(handle Handle, b []byte, keepHandle bool) (newhandle Handle, err error) {
	err = storage.Mutate(f.Accessor(), func() (err error) {
		switch handle {
		case 0, 2:
			return &EHandle{f.f.Name(), handle}
		case 1:
			keepHandle = true
		}
		newhandle = handle
		atom, newatoms := int64(handle), rq2Atoms(len(b))
		if newatoms > 3856 {
			return &EBadRequest{f.f.Name(), len(b)}
		}

		typ, oldatoms := f.getInfo(atom)
		switch {
		default:
			return &ECorrupted{f.f.Name(), atom << 4}
		case typ <= 0xfc: // non relocated used block
			switch {
			case newatoms == oldatoms: // in place replace
				f.writeUsed(b, atom)
			case newatoms < oldatoms: // in place shrink
				rightFree := f.checkRight(atom, oldatoms)
				if rightFree > 0 { // right join
					f.delFree(atom+oldatoms, rightFree)
				}
				f.addFree(atom+newatoms, oldatoms+rightFree-newatoms)
				f.writeUsed(b, atom)
			case newatoms > oldatoms:
				if rightFree := f.checkRight(atom, oldatoms); rightFree > 0 && newatoms <= oldatoms+rightFree {
					f.delFree(atom+oldatoms, rightFree)
					if newatoms < oldatoms+rightFree {
						f.addFree(atom+newatoms, oldatoms+rightFree-newatoms)
					}
					f.writeUsed(b, atom)
					return
				}

				if !keepHandle {
					f.Free(Handle(atom))
					newhandle, err = f.Alloc(b)
					return
				}

				// reloc
				newatom, e := f.Alloc(b)
				if e != nil {
					return e
				}

				buf := make([]byte, 16)
				buf[0] = 0xfd
				Handle(newatom).Put(buf[1:])
				f.Realloc(Handle(atom), buf[1:], true)
				f.write(buf[:1], atom<<4)
			}
		case typ == 0xfd: // reloc
			var target Handle
			buf := make([]byte, 7)
			f.read(buf, atom<<4+1)
			target.Get(buf)
			switch {
			case newatoms == 1:
				f.writeUsed(b, atom)
				f.Free(target)
			default:
				if rightFree := f.checkRight(atom, 1); rightFree > 0 && newatoms <= 1+rightFree {
					f.delFree(atom+1, rightFree)
					if newatoms < 1+rightFree {
						f.addFree(atom+newatoms, 1+rightFree-newatoms)
					}
					f.writeUsed(b, atom)
					f.Free(target)
					return
				}

				newtarget, e := f.Realloc(Handle(target), b, false)
				if e != nil {
					return e
				}

				if newtarget != target {
					Handle(newtarget).Put(buf)
					f.write(buf, atom<<4+1)
				}
			}
		}
		return
	})
	return
}

// Lock locks f for writing. If the lock is already locked for reading or writing,
// Lock blocks until the lock is available. To ensure that the lock eventually becomes available,
// a blocked Lock call excludes new readers from acquiring the lock.
func (f *File) Lock() {
	f.rwm.Lock()
}

// RLock locks f for reading. If the lock is already locked for writing or there is a writer
// already waiting to release the lock, RLock blocks until the writer has released the lock.
func (f *File) RLock() {
	f.rwm.RLock()
}

// Unlock unlocks f for writing. It is a run-time error if f is not locked for writing on entry to Unlock.
//
// As with Mutexes, a locked RWMutex is not associated with a particular goroutine.
// One goroutine may RLock (Lock) f and then arrange for another goroutine to RUnlock (Unlock) it.
func (f *File) Unlock() {
	f.rwm.Unlock()
}

// RUnlock undoes a single RLock call; it does not affect other simultaneous readers.
// It is a run-time error if f is not locked for reading on entry to RUnlock.
func (f *File) RUnlock() {
	f.rwm.RUnlock()
}

// LockedAlloc wraps Alloc in a Lock/Unlock pair.
func (f *File) LockedAlloc(b []byte) (handle Handle, err error) {
	f.Lock()
	defer f.Unlock()
	return f.Alloc(b)
}

// LockedFree wraps Free in a Lock/Unlock pair.
func (f *File) LockedFree(handle Handle) (err error) {
	f.Lock()
	defer f.Unlock()
	return f.Free(handle)
}

// LockedRead wraps Read in a RLock/RUnlock pair.
func (f *File) LockedRead(handle Handle) (b []byte, err error) {
	f.RLock()
	defer f.RUnlock()
	return f.Read(handle)
}

// LockedRealloc wraps Realloc in a Lock/Unlock pair.
func (f *File) LockedRealloc(handle Handle, b []byte, keepHandle bool) (newhandle Handle, err error) {
	f.Lock()
	defer f.Unlock()
	return f.Realloc(handle, b, keepHandle)
}
