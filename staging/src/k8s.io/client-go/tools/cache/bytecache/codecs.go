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

// Alternative serialization codecs, selectable for comparison via
// KUBE_BYTECACHE=proto|gob (KUBE_BYTECACHE=1 or "reloc" selects the native
// relocation codec). All codecs share the arena-at-rest + hot-LRU structure;
// only the byte format and materialization cost differ:
//
//   reloc: native memory layout + relocation list; decode = memcpy + patch.
//   proto: gogo protobuf wire bytes; decode = generated Unmarshal. Only for
//          types with generated marshalers; ~5x denser at rest.
//   gob:   encoding/gob; requires no generated code, but silently drops
//          unexported fields (e.g. resource.Quantity's value) and repeats
//          type descriptors per object — expected to be disqualified by the
//          per-type self-check; included for honest comparison.

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"reflect"
)

type codecKind int

const (
	kindReloc codecKind = iota
	kindProto
	kindGob
)

type protoMessage interface {
	Marshal() ([]byte, error)
	Unmarshal([]byte) error
}

func (c *Cache) encodeProto(obj interface{}) (ent, error) {
	m, ok := obj.(protoMessage)
	if !ok {
		return ent{}, fmt.Errorf("%T has no generated proto marshaler", obj)
	}
	b, err := m.Marshal()
	if err != nil {
		return ent{}, err
	}
	off, err := c.arena.Alloc(b)
	if err != nil {
		return ent{}, err
	}
	return ent{typ: reflect.TypeOf(obj).Elem(), off: off, size: len(b)}, nil
}

func (c *Cache) decodeProto(e ent) (interface{}, error) {
	obj := reflect.New(e.typ).Interface()
	if err := obj.(protoMessage).Unmarshal(c.arena.ReadOnly(e.off, e.size)); err != nil {
		return nil, err
	}
	return obj, nil
}

func (c *Cache) encodeGob(obj interface{}) (ent, error) {
	var buf bytes.Buffer
	// A fresh encoder per object: gob is stream-oriented, so this repeats
	// the type dictionary in every entry — part of what the comparison is
	// designed to show.
	if err := gob.NewEncoder(&buf).Encode(obj); err != nil {
		return ent{}, err
	}
	off, err := c.arena.Alloc(buf.Bytes())
	if err != nil {
		return ent{}, err
	}
	return ent{typ: reflect.TypeOf(obj).Elem(), off: off, size: buf.Len()}, nil
}

func (c *Cache) decodeGob(e ent) (interface{}, error) {
	obj := reflect.New(e.typ).Interface()
	if err := gob.NewDecoder(bytes.NewReader(c.arena.ReadOnly(e.off, e.size))).Decode(obj); err != nil {
		return nil, err
	}
	return obj, nil
}
