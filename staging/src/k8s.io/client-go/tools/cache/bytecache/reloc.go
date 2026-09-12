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
	"unsafe"
)

// ent describes one stored object: either an arena-resident block plus its
// relocation list, or a passthrough heap object (typ == nil).
type ent struct {
	typ                         reflect.Type
	off, size, relocOff, relocN int
	obj                         interface{} // passthrough only
}

// codec encodes objects of any pointer-to-struct type into the arena and
// materializes detached heap copies back out.
type codec struct {
	arena  *Arena
	copier *Copier
}

func newCodec(a *Arena) *codec {
	c := NewArenaMapCopier(a) // self-contained blocks (maps included)
	c.recordRelocs = true
	return &codec{arena: a, copier: c}
}

func (c *codec) encode(typ reflect.Type, src unsafe.Pointer) (ent, error) {
	c.copier.relocs = c.copier.relocs[:0]
	off, anchors, err := c.copier.Copy(typ, src)
	if err != nil {
		return ent{}, err
	}
	if len(anchors) != 0 {
		return ent{}, fmt.Errorf("bytecache: unexpected heap anchors")
	}
	size := c.arena.Used() - off
	e := ent{typ: typ, off: off, size: size, relocN: len(c.copier.relocs)}
	if e.relocN > 0 {
		e.relocOff, err = c.arena.Bump(4*e.relocN, 4)
		if err != nil {
			return ent{}, err
		}
		dst := unsafe.Slice((*int32)(unsafe.Pointer(&c.arena.rw[e.relocOff])), e.relocN)
		for i, r := range c.copier.relocs {
			dst[i] = r - int32(off) // block-relative
		}
	}
	return e, nil
}

// decode materializes a detached, self-contained heap copy: one []byte
// allocation, one memcpy, and a patch of each recorded pointer word. The
// returned object holds no arena pointers; its lifetime belongs to the GC.
func (c *codec) decode(e ent) (interface{}, error) {
	block := make([]byte, e.size)
	copy(block, c.arena.ro[e.off:e.off+e.size])
	delta := uintptr(unsafe.Pointer(&block[0])) - uintptr(unsafe.Pointer(&c.arena.ro[e.off]))
	if e.relocN > 0 {
		rel := unsafe.Slice((*int32)(unsafe.Pointer(&c.arena.ro[e.relocOff])), e.relocN)
		for _, r := range rel {
			*(*uintptr)(unsafe.Pointer(&block[r])) += delta
		}
	}
	return reflect.NewAt(e.typ, unsafe.Pointer(&block[0])).Interface(), nil
}
