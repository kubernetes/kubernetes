//go:build linux && amd64 && !race

/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package bytecache

// Arena-resident Go maps via snapshot relocation.
//
// We never reimplement map hashing or insertion. A map built normally by the
// runtime is a valid, self-contained pointer graph: Map struct -> directory
// -> tables -> group arrays. We bit-copy that graph into the arena and rebase
// the internal pointers. Two properties make the copy remain a valid map for
// the runtime's own READ paths (mapaccess, len, range):
//
//  1. The hash seed is a field of the Map struct, so it travels with the copy.
//  2. Go's hash functions hash key *contents*, not addresses, so relocating
//     string keys into the arena does not change any hash: every ctrl byte
//     and probe sequence stays correct.
//
// Writes are a different story: mapassign/mapdelete/clear would write to the
// Map struct and groups — which live in the read-only mapping, so a buggy
// consumer's map write becomes an immediate hardware fault. That closes the
// mutation-protection hole that heap-anchored maps had.
//
// This file mirrors internal runtime layouts (Go 1.24+ Swiss maps, amd64) and
// is therefore version-locked; layout drift is caught by VerifyMapLayout-style
// round-trip checks rather than discovered in production.

import (
	"fmt"
	"reflect"
	"unsafe"
)

// rtMapType mirrors internal/abi.MapType. The leading embedded abi.Type is
// opaque padding: we only read the map-specific tail fields.
type rtMapType struct {
	_         [48]byte // abi.Type on 64-bit
	key       unsafe.Pointer
	elem      unsafe.Pointer
	group     unsafe.Pointer
	hasher    func(unsafe.Pointer, uintptr) uintptr
	groupSize uintptr
	slotSize  uintptr
	elemOff   uintptr
	flags     uint32
}

const (
	rtMapIndirectKey  = 1 << 2 // abi.MapIndirectKey
	rtMapIndirectElem = 1 << 3 // abi.MapIndirectElem
)

// rtMap mirrors internal/runtime/maps.Map, with pointer fields as uintptr so
// every store we do is a plain integer store (no GC write barriers on arena
// addresses, and no shading of stale arena bytes).
type rtMap struct {
	used              uint64
	seed              uintptr
	dirPtr            uintptr
	dirLen            int
	globalDepth       uint8
	globalShift       uint8
	writing           uint8
	tombstonePossible bool
	clearSeq          uint64
}

// rtTable mirrors internal/runtime/maps.table.
type rtTable struct {
	used       uint16
	capacity   uint16
	growthLeft uint16
	localDepth uint8
	index      int
	groupsData uintptr // groupsReference.data
	lengthMask uint64  // groupsReference.lengthMask
}

func mapTypeFor(t reflect.Type) *rtMapType {
	type iface struct{ tab, data unsafe.Pointer }
	return (*rtMapType)((*iface)(unsafe.Pointer(&t)).data)
}

// copyRaw bump-allocates and byte-copies size bytes from src.
func (c *Copier) copyRaw(src uintptr, size, align int) int {
	off := c.mustBump(size, align)
	copy(c.arena.rw[off:off+size], unsafe.Slice((*byte)(unsafe.Pointer(src)), size)) //nolint:govet // uintptr captured from memory kept live by the caller; see package doc
	return off
}

// snapshotMap relocates the map whose word is at p into the arena.
func (c *Copier) snapshotMap(t reflect.Type, p unsafe.Pointer) {
	wp := (*uintptr)(p)
	if *wp == 0 {
		return
	}
	mt := mapTypeFor(t)
	// Layout sanity checks; a Go version whose maps we don't understand
	// panics out of Copy as an error instead of corrupting memory.
	slotsBase := int(mt.groupSize) - 8*int(mt.slotSize)
	if slotsBase != 8 || mt.elemOff >= mt.slotSize {
		panic(fmt.Sprintf("unexpected swiss map layout for %v: groupSize=%d slotSize=%d elemOff=%d",
			t, mt.groupSize, mt.slotSize, mt.elemOff))
	}

	mOff := c.copyRaw(*wp, int(unsafe.Sizeof(rtMap{})), 8)
	m := (*rtMap)(c.rwAt(mOff))
	switch {
	case m.dirPtr == 0:
		// Empty map that never allocated. Nothing further to relocate.
	case m.dirLen == 0:
		// Small-map optimization: dirPtr points directly at a single group.
		m.dirPtr = c.roAddr(c.snapshotGroups(mt, t, m.dirPtr, 1))
		c.record(unsafe.Pointer(&m.dirPtr))
	default:
		// dirPtr -> [dirLen]*table; entries may repeat (extendible hashing),
		// so relocate each distinct table exactly once.
		dOff := c.copyRaw(m.dirPtr, m.dirLen*8, 8)
		dir := unsafe.Slice((*uintptr)(c.rwAt(dOff)), m.dirLen)
		moved := make(map[uintptr]uintptr, m.dirLen)
		for i, tp := range dir {
			np, ok := moved[tp]
			if !ok {
				tOff := c.copyRaw(tp, int(unsafe.Sizeof(rtTable{})), 8)
				tb := (*rtTable)(c.rwAt(tOff))
				tb.groupsData = c.roAddr(c.snapshotGroups(mt, t, tb.groupsData, int(tb.lengthMask)+1))
				c.record(unsafe.Pointer(&tb.groupsData))
				np = c.roAddr(tOff)
				moved[tp] = np
			}
			dir[i] = np
			c.record(unsafe.Pointer(&dir[i]))
		}
		m.dirPtr = c.roAddr(dOff)
		c.record(unsafe.Pointer(&m.dirPtr))
	}
	*wp = c.roAddr(mOff)
	c.record(p)
}

// snapshotGroups relocates an array of n slot groups and fixes up the keys
// and elems of every present slot. Empty/deleted slots are zeroed so no stale
// heap pointers are carried into the arena.
func (c *Copier) snapshotGroups(mt *rtMapType, t reflect.Type, data uintptr, n int) int {
	gOff := c.copyRaw(data, n*int(mt.groupSize), 8)
	kt, et := t.Key(), t.Elem()
	kPtr, ePtr := c.hasPointers(kt), c.hasPointers(et)
	kInd, eInd := mt.flags&rtMapIndirectKey != 0, mt.flags&rtMapIndirectElem != 0
	for g := range n {
		base := gOff + g*int(mt.groupSize)
		ctrls := *(*uint64)(c.rwAt(base))
		for s := range 8 {
			slotOff := base + 8 + s*int(mt.slotSize)
			if byte(ctrls>>(8*s))&0x80 != 0 { // empty or deleted
				clear(c.arena.rw[slotOff : slotOff+int(mt.slotSize)])
				continue
			}
			c.fixupSlot(kt, kPtr, kInd, slotOff)
			c.fixupSlot(et, ePtr, eInd, slotOff+int(mt.elemOff))
		}
	}
	return gOff
}

func (c *Copier) fixupSlot(t reflect.Type, hasPtr, indirect bool, off int) {
	if indirect {
		// Slot holds a pointer to the key/elem (types > 128 bytes).
		pp := (*uintptr)(c.rwAt(off))
		*pp = c.roAddr(c.clone(t, unsafe.Pointer(*pp))) //nolint:govet // uintptr captured from memory kept live by the caller; see package doc
		c.record(unsafe.Pointer(pp))
		return
	}
	if hasPtr {
		c.fixup(t, c.rwAt(off))
	}
}
