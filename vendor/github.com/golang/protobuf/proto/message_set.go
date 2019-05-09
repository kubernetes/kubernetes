// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2010 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
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
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
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

/*
 * Support for message sets.
 */

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"sync"
)

// errNoMessageTypeID occurs when a protocol buffer does not have a message type ID.
// A message type ID is required for storing a protocol buffer in a message set.
var errNoMessageTypeID = errors.New("proto does not have a message type ID")

// The first two types (_MessageSet_Item and messageSet)
// model what the protocol compiler produces for the following protocol message:
//   message MessageSet {
//     repeated group Item = 1 {
//       required int32 type_id = 2;
//       required string message = 3;
//     };
//   }
// That is the MessageSet wire format. We can't use a proto to generate these
// because that would introduce a circular dependency between it and this package.

type _MessageSet_Item struct {
	TypeId  *int32 `protobuf:"varint,2,req,name=type_id"`
	Message []byte `protobuf:"bytes,3,req,name=message"`
}

type messageSet struct {
	Item             []*_MessageSet_Item `protobuf:"group,1,rep"`
	XXX_unrecognized []byte
	// TODO: caching?
}

// Make sure messageSet is a Message.
var _ Message = (*messageSet)(nil)

// messageTypeIder is an interface satisfied by a protocol buffer type
// that may be stored in a MessageSet.
type messageTypeIder interface {
	MessageTypeId() int32
}

func (ms *messageSet) find(pb Message) *_MessageSet_Item {
	mti, ok := pb.(messageTypeIder)
	if !ok {
		return nil
	}
	id := mti.MessageTypeId()
	for _, item := range ms.Item {
		if *item.TypeId == id {
			return item
		}
	}
	return nil
}

func (ms *messageSet) Has(pb Message) bool {
	return ms.find(pb) != nil
}

func (ms *messageSet) Unmarshal(pb Message) error {
	if item := ms.find(pb); item != nil {
		return Unmarshal(item.Message, pb)
	}
	if _, ok := pb.(messageTypeIder); !ok {
		return errNoMessageTypeID
	}
	return nil // TODO: return error instead?
}

func (ms *messageSet) Marshal(pb Message) error {
	msg, err := Marshal(pb)
	if err != nil {
		return err
	}
	if item := ms.find(pb); item != nil {
		// reuse existing item
		item.Message = msg
		return nil
	}

	mti, ok := pb.(messageTypeIder)
	if !ok {
		return errNoMessageTypeID
	}

	mtid := mti.MessageTypeId()
	ms.Item = append(ms.Item, &_MessageSet_Item{
		TypeId:  &mtid,
		Message: msg,
	})
	return nil
}

func (ms *messageSet) Reset()         { *ms = messageSet{} }
func (ms *messageSet) String() string { return CompactTextString(ms) }
func (*messageSet) ProtoMessage()     {}

// Support for the message_set_wire_format message option.

func skipVarint(buf []byte) []byte {
	i := 0
	for ; buf[i]&0x80 != 0; i++ {
	}
	return buf[i+1:]
}

// MarshalMessageSet encodes the extension map represented by m in the message set wire format.
// It is called by generated Marshal methods on protocol buffer messages with the message_set_wire_format option.
func MarshalMessageSet(exts interface{}) ([]byte, error) {
	return marshalMessageSet(exts, false)
}

// marshaMessageSet implements above function, with the opt to turn on / off deterministic during Marshal.
func marshalMessageSet(exts interface{}, deterministic bool) ([]byte, error) {
	switch exts := exts.(type) {
	case *XXX_InternalExtensions:
		var u marshalInfo
		siz := u.sizeMessageSet(exts)
		b := make([]byte, 0, siz)
		return u.appendMessageSet(b, exts, deterministic)

	case map[int32]Extension:
		// This is an old-style extension map.
		// Wrap it in a new-style XXX_InternalExtensions.
		ie := XXX_InternalExtensions{
			p: &struct {
				mu           sync.Mutex
				extensionMap map[int32]Extension
			}{
				extensionMap: exts,
			},
		}

		var u marshalInfo
		siz := u.sizeMessageSet(&ie)
		b := make([]byte, 0, siz)
		return u.appendMessageSet(b, &ie, deterministic)

	default:
		return nil, errors.New("proto: not an extension map")
	}
}

// UnmarshalMessageSet decodes the extension map encoded in buf in the message set wire format.
// It is called by Unmarshal methods on protocol buffer messages with the message_set_wire_format option.
func UnmarshalMessageSet(buf []byte, exts interface{}) error {
	var m map[int32]Extension
	switch exts := exts.(type) {
	case *XXX_InternalExtensions:
		m = exts.extensionsWrite()
	case map[int32]Extension:
		m = exts
	default:
		return errors.New("proto: not an extension map")
	}

	ms := new(messageSet)
	if err := Unmarshal(buf, ms); err != nil {
		return err
	}
	for _, item := range ms.Item {
		id := *item.TypeId
		msg := item.Message

		// Restore wire type and field number varint, plus length varint.
		// Be careful to preserve duplicate items.
		b := EncodeVarint(uint64(id)<<3 | WireBytes)
		if ext, ok := m[id]; ok {
			// Existing data; rip off the tag and length varint
			// so we join the new data correctly.
			// We can assume that ext.enc is set because we are unmarshaling.
			o := ext.enc[len(b):]   // skip wire type and field number
			_, n := DecodeVarint(o) // calculate length of length varint
			o = o[n:]               // skip length varint
			msg = append(o, msg...) // join old data and new data
		}
		b = append(b, EncodeVarint(uint64(len(msg)))...)
		b = append(b, msg...)

		m[id] = Extension{enc: b}
	}
	return nil
}

// MarshalMessageSetJSON encodes the extension map represented by m in JSON format.
// It is called by generated MarshalJSON methods on protocol buffer messages with the message_set_wire_format option.
func MarshalMessageSetJSON(exts interface{}) ([]byte, error) {
	var m map[int32]Extension
	switch exts := exts.(type) {
	case *XXX_InternalExtensions:
		var mu sync.Locker
		m, mu = exts.extensionsRead()
		if m != nil {
			// Keep the extensions map locked until we're done marshaling to prevent
			// races between marshaling and unmarshaling the lazily-{en,de}coded
			// values.
			mu.Lock()
			defer mu.Unlock()
		}
	case map[int32]Extension:
		m = exts
	default:
		return nil, errors.New("proto: not an extension map")
	}
	var b bytes.Buffer
	b.WriteByte('{')

	// Process the map in key order for deterministic output.
	ids := make([]int32, 0, len(m))
	for id := range m {
		ids = append(ids, id)
	}
	sort.Sort(int32Slice(ids)) // int32Slice defined in text.go

	for i, id := range ids {
		ext := m[id]
		msd, ok := messageSetMap[id]
		if !ok {
			// Unknown type; we can't render it, so skip it.
			continue
		}

		if i > 0 && b.Len() > 1 {
			b.WriteByte(',')
		}

		fmt.Fprintf(&b, `"[%s]":`, msd.name)

		x := ext.value
		if x == nil {
			x = reflect.New(msd.t.Elem()).Interface()
			if err := Unmarshal(ext.enc, x.(Message)); err != nil {
				return nil, err
			}
		}
		d, err := json.Marshal(x)
		if err != nil {
			return nil, err
		}
		b.Write(d)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// UnmarshalMessageSetJSON decodes the extension map encoded in buf in JSON format.
// It is called by generated UnmarshalJSON methods on protocol buffer messages with the message_set_wire_format option.
func UnmarshalMessageSetJSON(buf []byte, exts interface{}) error {
	// Common-case fast path.
	if len(buf) == 0 || bytes.Equal(buf, []byte("{}")) {
		return nil
	}

	// This is fairly tricky, and it's not clear that it is needed.
	return errors.New("TODO: UnmarshalMessageSetJSON not yet implemented")
}

// A global registry of types that can be used in a MessageSet.

var messageSetMap = make(map[int32]messageSetDesc)

type messageSetDesc struct {
	t    reflect.Type // pointer to struct
	name string
}

// RegisterMessageSetType is called from the generated code.
func RegisterMessageSetType(m Message, fieldNum int32, name string) {
	messageSetMap[fieldNum] = messageSetDesc{
		t:    reflect.TypeOf(m),
		name: name,
	}
}
