// Copyright (c) 2013, Vastech SA (PTY) LTD. All rights reserved.
// http://github.com/gogo/protobuf/gogoproto
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package proto

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
)

func GetBoolExtension(pb extendableProto, extension *ExtensionDesc, ifnotset bool) bool {
	if reflect.ValueOf(pb).IsNil() {
		return ifnotset
	}
	value, err := GetExtension(pb, extension)
	if err != nil {
		return ifnotset
	}
	if value == nil {
		return ifnotset
	}
	if value.(*bool) == nil {
		return ifnotset
	}
	return *(value.(*bool))
}

func (this *Extension) Equal(that *Extension) bool {
	return bytes.Equal(this.enc, that.enc)
}

func (this *Extension) Compare(that *Extension) int {
	return bytes.Compare(this.enc, that.enc)
}

func SizeOfExtensionMap(m map[int32]Extension) (n int) {
	return sizeExtensionMap(m)
}

type sortableMapElem struct {
	field int32
	ext   Extension
}

func newSortableExtensionsFromMap(m map[int32]Extension) sortableExtensions {
	s := make(sortableExtensions, 0, len(m))
	for k, v := range m {
		s = append(s, &sortableMapElem{field: k, ext: v})
	}
	return s
}

type sortableExtensions []*sortableMapElem

func (this sortableExtensions) Len() int { return len(this) }

func (this sortableExtensions) Swap(i, j int) { this[i], this[j] = this[j], this[i] }

func (this sortableExtensions) Less(i, j int) bool { return this[i].field < this[j].field }

func (this sortableExtensions) String() string {
	sort.Sort(this)
	ss := make([]string, len(this))
	for i := range this {
		ss[i] = fmt.Sprintf("%d: %v", this[i].field, this[i].ext)
	}
	return "map[" + strings.Join(ss, ",") + "]"
}

func StringFromExtensionsMap(m map[int32]Extension) string {
	return newSortableExtensionsFromMap(m).String()
}

func StringFromExtensionsBytes(ext []byte) string {
	m, err := BytesToExtensionsMap(ext)
	if err != nil {
		panic(err)
	}
	return StringFromExtensionsMap(m)
}

func EncodeExtensionMap(m map[int32]Extension, data []byte) (n int, err error) {
	if err := encodeExtensionMap(m); err != nil {
		return 0, err
	}
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, int(k))
	}
	sort.Ints(keys)
	for _, k := range keys {
		n += copy(data[n:], m[int32(k)].enc)
	}
	return n, nil
}

func GetRawExtension(m map[int32]Extension, id int32) ([]byte, error) {
	if m[id].value == nil || m[id].desc == nil {
		return m[id].enc, nil
	}
	if err := encodeExtensionMap(m); err != nil {
		return nil, err
	}
	return m[id].enc, nil
}

func size(buf []byte, wire int) (int, error) {
	switch wire {
	case WireVarint:
		_, n := DecodeVarint(buf)
		return n, nil
	case WireFixed64:
		return 8, nil
	case WireBytes:
		v, n := DecodeVarint(buf)
		return int(v) + n, nil
	case WireFixed32:
		return 4, nil
	case WireStartGroup:
		offset := 0
		for {
			u, n := DecodeVarint(buf[offset:])
			fwire := int(u & 0x7)
			offset += n
			if fwire == WireEndGroup {
				return offset, nil
			}
			s, err := size(buf[offset:], wire)
			if err != nil {
				return 0, err
			}
			offset += s
		}
	}
	return 0, fmt.Errorf("proto: can't get size for unknown wire type %d", wire)
}

func BytesToExtensionsMap(buf []byte) (map[int32]Extension, error) {
	m := make(map[int32]Extension)
	i := 0
	for i < len(buf) {
		tag, n := DecodeVarint(buf[i:])
		if n <= 0 {
			return nil, fmt.Errorf("unable to decode varint")
		}
		fieldNum := int32(tag >> 3)
		wireType := int(tag & 0x7)
		l, err := size(buf[i+n:], wireType)
		if err != nil {
			return nil, err
		}
		end := i + int(l) + n
		m[int32(fieldNum)] = Extension{enc: buf[i:end]}
		i = end
	}
	return m, nil
}

func NewExtension(e []byte) Extension {
	ee := Extension{enc: make([]byte, len(e))}
	copy(ee.enc, e)
	return ee
}

func AppendExtension(e extendableProto, tag int32, buf []byte) {
	if ee, eok := e.(extensionsMap); eok {
		ext := ee.ExtensionMap()[int32(tag)] // may be missing
		ext.enc = append(ext.enc, buf...)
		ee.ExtensionMap()[int32(tag)] = ext
	} else if ee, eok := e.(extensionsBytes); eok {
		ext := ee.GetExtensions()
		*ext = append(*ext, buf...)
	}
}

func (this Extension) GoString() string {
	if this.enc == nil {
		if err := encodeExtension(&this); err != nil {
			panic(err)
		}
	}
	return fmt.Sprintf("proto.NewExtension(%#v)", this.enc)
}

func SetUnsafeExtension(pb extendableProto, fieldNum int32, value interface{}) error {
	typ := reflect.TypeOf(pb).Elem()
	ext, ok := extensionMaps[typ]
	if !ok {
		return fmt.Errorf("proto: bad extended type; %s is not extendable", typ.String())
	}
	desc, ok := ext[fieldNum]
	if !ok {
		return errors.New("proto: bad extension number; not in declared ranges")
	}
	return setExtension(pb, desc, value)
}

func GetUnsafeExtension(pb extendableProto, fieldNum int32) (interface{}, error) {
	typ := reflect.TypeOf(pb).Elem()
	ext, ok := extensionMaps[typ]
	if !ok {
		return nil, fmt.Errorf("proto: bad extended type; %s is not extendable", typ.String())
	}
	desc, ok := ext[fieldNum]
	if !ok {
		return nil, fmt.Errorf("unregistered field number %d", fieldNum)
	}
	return GetExtension(pb, desc)
}
