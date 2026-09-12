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

import (
	"fmt"
	"reflect"
	"time"
	"unsafe"
)

// stringRepr / sliceRepr mirror the runtime headers, but with uintptr data
// words so that all our stores into arena memory are plain integer stores:
// no typed pointer writes into the arena means no GC write barriers fire on
// non-heap addresses.
type stringRepr struct {
	data uintptr
	len  int
}

type sliceRepr struct {
	data uintptr
	len  int
	cap  int
}

var locationType = reflect.TypeFor[time.Location]()

// Copier performs a relocating deep-copy of a live Go object graph into an
// Arena, preserving native memory layout. The copy in the arena is a real Go
// object: a *T cast of its offset (in the read-only mapping) is directly
// usable, with zero decode cost.
//
// Pointers, strings, and slices are relocated into the arena, with their data
// pointers rebased to the read-only mapping. Maps cannot live in the arena —
// their internals are owned by the Go runtime — so each map is rebuilt as a
// real heap map (whose string keys/values do alias the arena) and returned as
// an "anchor": because the GC never scans arena memory, the caller must keep
// anchors alive for as long as the arena object is reachable.
//
// Assumes acyclic object graphs (true for k8s API types). *time.Location is
// shared rather than copied: in practice it points at the runtime's UTC/Local
// globals.
type Copier struct {
	arena   *Arena
	anchors []any
	hasPtr  map[reflect.Type]bool
	// arenaMaps: relocate map internals into the arena via snapshotMap
	// (version-locked to the runtime's map layout) instead of rebuilding
	// heap maps that must be anchored.
	arenaMaps bool
	// recordRelocs: remember the arena offset of every internal pointer
	// word written. A block plus its reloc list can then be relocated
	// anywhere with memcpy + patch — no reflection walk.
	recordRelocs bool
	relocs       []int32
	depth        int
}

// record notes that the pointer word at p (inside the RW mapping) holds an
// internal pointer that must be patched if the block is relocated.
func (c *Copier) record(p unsafe.Pointer) {
	if !c.recordRelocs {
		return
	}
	off := uintptr(p) - uintptr(unsafe.Pointer(&c.arena.rw[0]))
	if off >= uintptr(len(c.arena.rw)) {
		panic("bytecache: reloc recording outside arena (heap-map mode is not relocatable)")
	}
	c.relocs = append(c.relocs, int32(off))
}

func NewCopier(a *Arena) *Copier {
	return &Copier{arena: a, hasPtr: map[reflect.Type]bool{}}
}

// NewArenaMapCopier relocates maps into the arena too: no heap anchors at
// all, and map contents become hardware-read-only like everything else.
func NewArenaMapCopier(a *Arena) *Copier {
	c := NewCopier(a)
	c.arenaMaps = true
	return c
}

// Copy relocates the object of type t at src into the arena. It returns the
// arena offset of the root and the heap anchors (rebuilt maps) that must be
// kept alive alongside it. The caller must keep the source object alive until
// Copy returns (it is read through raw pointers).
func (c *Copier) Copy(t reflect.Type, src unsafe.Pointer) (off int, anchors []any, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("arena copy failed: %v", r)
		}
	}()
	c.anchors = nil
	off = c.clone(t, src)
	anchors = c.anchors
	c.anchors = nil
	return off, anchors, nil
}

// clone bump-allocates space for t, memcpys the raw bytes from src, then
// fixes up any pointer-bearing fields in place.
func (c *Copier) clone(t reflect.Type, src unsafe.Pointer) int {
	c.depth++
	defer func() { c.depth-- }()
	if c.depth > 512 {
		panic("object graph too deep (cyclic?)")
	}
	size := int(t.Size())
	off := c.mustBump(size, int(t.Align()))
	copy(c.arena.rw[off:off+size], unsafe.Slice((*byte)(src), size))
	if c.hasPointers(t) {
		c.fixup(t, c.rwAt(off))
	}
	return off
}

func (c *Copier) mustBump(size, align int) int {
	off, err := c.arena.Bump(size, align)
	if err != nil {
		panic(err)
	}
	return off
}

// rwAt is where we write during construction; roAddr is the address readers
// will use — pointers stored in the arena are rebased onto the RO mapping.
func (c *Copier) rwAt(off int) unsafe.Pointer { return unsafe.Pointer(&c.arena.rw[off]) }
func (c *Copier) roAddr(off int) uintptr      { return uintptr(unsafe.Pointer(&c.arena.ro[off])) }

// fixup rewrites the pointer-bearing parts of the value of type t at p
// (writable memory: either the RW mapping or a heap temporary for map
// entries), relocating everything it references into the arena.
func (c *Copier) fixup(t reflect.Type, p unsafe.Pointer) {
	switch t.Kind() {
	case reflect.Struct:
		for f := range t.Fields() {
			if c.hasPointers(f.Type) {
				c.fixup(f.Type, unsafe.Add(p, f.Offset))
			}
		}
	case reflect.Array:
		et := t.Elem()
		if c.hasPointers(et) {
			for i := 0; i < t.Len(); i++ {
				c.fixup(et, unsafe.Add(p, uintptr(i)*et.Size()))
			}
		}
	case reflect.String:
		h := (*stringRepr)(p)
		if h.len == 0 {
			h.data = 0
			return
		}
		off := c.mustBump(h.len, 1)
		copy(c.arena.rw[off:off+h.len], unsafe.Slice((*byte)(unsafe.Pointer(h.data)), h.len)) //nolint:govet // uintptr captured from memory kept live by the caller; see package doc
		h.data = c.roAddr(off)
		c.record(p)
	case reflect.Slice:
		h := (*sliceRepr)(p)
		if h.len == 0 {
			h.data, h.cap = 0, 0
			return
		}
		et := t.Elem()
		esz := int(et.Size())
		off := c.mustBump(esz*h.len, int(et.Align()))
		copy(c.arena.rw[off:off+esz*h.len], unsafe.Slice((*byte)(unsafe.Pointer(h.data)), esz*h.len)) //nolint:govet // uintptr captured from memory kept live by the caller; see package doc
		if c.hasPointers(et) {
			for i := 0; i < h.len; i++ {
				c.fixup(et, c.rwAt(off+i*esz))
			}
		}
		h.data, h.cap = c.roAddr(off), h.len
		c.record(p)
	case reflect.Pointer:
		pp := (*uintptr)(p)
		if *pp == 0 {
			return
		}
		et := t.Elem()
		if et == locationType {
			return // shared runtime global (time.UTC / time.Local)
		}
		*pp = c.roAddr(c.clone(et, unsafe.Pointer(*pp))) //nolint:govet // uintptr captured from memory kept live by the caller; see package doc
		c.record(p)
	case reflect.Map:
		if c.arenaMaps {
			c.snapshotMap(t, p)
			return
		}
		wp := (*uintptr)(p)
		if *wp == 0 {
			return
		}
		src := reflect.NewAt(t, p).Elem()
		dst := reflect.MakeMapWithSize(t, src.Len())
		it := src.MapRange()
		for it.Next() {
			dst.SetMapIndex(c.relocate(it.Key()), c.relocate(it.Value()))
		}
		c.anchors = append(c.anchors, dst.Interface())
		*wp = uintptr(dst.UnsafePointer())
	case reflect.Interface:
		// An interface value is (type/itab word, data word). The type word
		// points at static type metadata in the binary — kept as-is. The
		// data word is either the value itself (pointer-shaped types: *T,
		// map, ...) or a pointer to a heap box holding the value.
		words := (*[2]uintptr)(p)
		if words[0] == 0 {
			return // nil interface
		}
		v := reflect.NewAt(t, p).Elem().Elem()
		if !v.IsValid() {
			return
		}
		dt := v.Type()
		if isDirectIface(dt) {
			c.fixup(dt, unsafe.Pointer(&words[1]))
		} else if words[1] != 0 {
			words[1] = c.roAddr(c.clone(dt, unsafe.Pointer(words[1]))) //nolint:govet // uintptr captured from memory kept live by the caller; see package doc
			c.record(unsafe.Pointer(&words[1]))
		}
	default:
		// Chan, Func, UnsafePointer: not relocatable.
		panic(fmt.Sprintf("unsupported kind %v (%v)", t.Kind(), t))
	}
}

// isDirectIface reports whether values of t are stored directly in an
// interface's data word (rather than boxed behind a pointer). Reads the
// TFlagDirectIface bit from the runtime type descriptor (abi.Type.TFlag at
// offset 20 on 64-bit, bit 1<<5 as of Go 1.26) — version-locked like the map
// mirrors, and self-checked at init below. (Case in point on version drift:
// this bit lived in the Kind_ byte until recently.)
func isDirectIface(t reflect.Type) bool {
	type iface struct{ tab, data unsafe.Pointer }
	tp := (*iface)(unsafe.Pointer(&t)).data
	return *(*uint8)(unsafe.Add(tp, 20))&(1<<5) != 0
}

func init() {
	// The flag is a cached computation of "pointer-shaped": verify our read
	// of it against known types so layout drift fails loudly at startup.
	if !isDirectIface(reflect.TypeFor[*int]()) ||
		!isDirectIface(reflect.TypeFor[map[string]string]()) ||
		isDirectIface(reflect.TypeFor[string]()) ||
		isDirectIface(reflect.TypeFor[int64]()) ||
		isDirectIface(reflect.TypeFor[[]int]()) {
		panic("bytecache: abi.Type.TFlag direct-iface bit not where we expect; update copier.go mirrors for this Go version")
	}
}

// relocate returns a copy of v whose referenced data (string bytes, nested
// pointers, ...) lives in the arena; the top-level value itself stays on the
// heap, because it is about to be stored inside a heap map's buckets.
func (c *Copier) relocate(v reflect.Value) reflect.Value {
	if !c.hasPointers(v.Type()) {
		return v
	}
	tmp := reflect.New(v.Type())
	tmp.Elem().Set(v)
	c.fixup(v.Type(), tmp.UnsafePointer())
	return tmp.Elem()
}

func (c *Copier) hasPointers(t reflect.Type) bool {
	if v, ok := c.hasPtr[t]; ok {
		return v
	}
	var r bool
	switch t.Kind() {
	case reflect.Pointer, reflect.Map, reflect.String, reflect.Slice,
		reflect.Chan, reflect.Func, reflect.Interface, reflect.UnsafePointer:
		r = true
	case reflect.Array:
		r = c.hasPointers(t.Elem())
	case reflect.Struct:
		for f := range t.Fields() {
			if c.hasPointers(f.Type) {
				r = true
				break
			}
		}
	}
	c.hasPtr[t] = r
	return r
}
