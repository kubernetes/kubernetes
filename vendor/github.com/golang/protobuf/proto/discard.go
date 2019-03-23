// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2017 The Go Authors.  All rights reserved.
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

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
)

type generatedDiscarder interface {
	XXX_DiscardUnknown()
}

// DiscardUnknown recursively discards all unknown fields from this message
// and all embedded messages.
//
// When unmarshaling a message with unrecognized fields, the tags and values
// of such fields are preserved in the Message. This allows a later call to
// marshal to be able to produce a message that continues to have those
// unrecognized fields. To avoid this, DiscardUnknown is used to
// explicitly clear the unknown fields after unmarshaling.
//
// For proto2 messages, the unknown fields of message extensions are only
// discarded from messages that have been accessed via GetExtension.
func DiscardUnknown(m Message) {
	if m, ok := m.(generatedDiscarder); ok {
		m.XXX_DiscardUnknown()
		return
	}
	// TODO: Dynamically populate a InternalMessageInfo for legacy messages,
	// but the master branch has no implementation for InternalMessageInfo,
	// so it would be more work to replicate that approach.
	discardLegacy(m)
}

// DiscardUnknown recursively discards all unknown fields.
func (a *InternalMessageInfo) DiscardUnknown(m Message) {
	di := atomicLoadDiscardInfo(&a.discard)
	if di == nil {
		di = getDiscardInfo(reflect.TypeOf(m).Elem())
		atomicStoreDiscardInfo(&a.discard, di)
	}
	di.discard(toPointer(&m))
}

type discardInfo struct {
	typ reflect.Type

	initialized int32 // 0: only typ is valid, 1: everything is valid
	lock        sync.Mutex

	fields       []discardFieldInfo
	unrecognized field
}

type discardFieldInfo struct {
	field   field // Offset of field, guaranteed to be valid
	discard func(src pointer)
}

var (
	discardInfoMap  = map[reflect.Type]*discardInfo{}
	discardInfoLock sync.Mutex
)

func getDiscardInfo(t reflect.Type) *discardInfo {
	discardInfoLock.Lock()
	defer discardInfoLock.Unlock()
	di := discardInfoMap[t]
	if di == nil {
		di = &discardInfo{typ: t}
		discardInfoMap[t] = di
	}
	return di
}

func (di *discardInfo) discard(src pointer) {
	if src.isNil() {
		return // Nothing to do.
	}

	if atomic.LoadInt32(&di.initialized) == 0 {
		di.computeDiscardInfo()
	}

	for _, fi := range di.fields {
		sfp := src.offset(fi.field)
		fi.discard(sfp)
	}

	// For proto2 messages, only discard unknown fields in message extensions
	// that have been accessed via GetExtension.
	if em, err := extendable(src.asPointerTo(di.typ).Interface()); err == nil {
		// Ignore lock since DiscardUnknown is not concurrency safe.
		emm, _ := em.extensionsRead()
		for _, mx := range emm {
			if m, ok := mx.value.(Message); ok {
				DiscardUnknown(m)
			}
		}
	}

	if di.unrecognized.IsValid() {
		*src.offset(di.unrecognized).toBytes() = nil
	}
}

func (di *discardInfo) computeDiscardInfo() {
	di.lock.Lock()
	defer di.lock.Unlock()
	if di.initialized != 0 {
		return
	}
	t := di.typ
	n := t.NumField()

	for i := 0; i < n; i++ {
		f := t.Field(i)
		if strings.HasPrefix(f.Name, "XXX_") {
			continue
		}

		dfi := discardFieldInfo{field: toField(&f)}
		tf := f.Type

		// Unwrap tf to get its most basic type.
		var isPointer, isSlice bool
		if tf.Kind() == reflect.Slice && tf.Elem().Kind() != reflect.Uint8 {
			isSlice = true
			tf = tf.Elem()
		}
		if tf.Kind() == reflect.Ptr {
			isPointer = true
			tf = tf.Elem()
		}
		if isPointer && isSlice && tf.Kind() != reflect.Struct {
			panic(fmt.Sprintf("%v.%s cannot be a slice of pointers to primitive types", t, f.Name))
		}

		switch tf.Kind() {
		case reflect.Struct:
			switch {
			case !isPointer:
				panic(fmt.Sprintf("%v.%s cannot be a direct struct value", t, f.Name))
			case isSlice: // E.g., []*pb.T
				di := getDiscardInfo(tf)
				dfi.discard = func(src pointer) {
					sps := src.getPointerSlice()
					for _, sp := range sps {
						if !sp.isNil() {
							di.discard(sp)
						}
					}
				}
			default: // E.g., *pb.T
				di := getDiscardInfo(tf)
				dfi.discard = func(src pointer) {
					sp := src.getPointer()
					if !sp.isNil() {
						di.discard(sp)
					}
				}
			}
		case reflect.Map:
			switch {
			case isPointer || isSlice:
				panic(fmt.Sprintf("%v.%s cannot be a pointer to a map or a slice of map values", t, f.Name))
			default: // E.g., map[K]V
				if tf.Elem().Kind() == reflect.Ptr { // Proto struct (e.g., *T)
					dfi.discard = func(src pointer) {
						sm := src.asPointerTo(tf).Elem()
						if sm.Len() == 0 {
							return
						}
						for _, key := range sm.MapKeys() {
							val := sm.MapIndex(key)
							DiscardUnknown(val.Interface().(Message))
						}
					}
				} else {
					dfi.discard = func(pointer) {} // Noop
				}
			}
		case reflect.Interface:
			// Must be oneof field.
			switch {
			case isPointer || isSlice:
				panic(fmt.Sprintf("%v.%s cannot be a pointer to a interface or a slice of interface values", t, f.Name))
			default: // E.g., interface{}
				// TODO: Make this faster?
				dfi.discard = func(src pointer) {
					su := src.asPointerTo(tf).Elem()
					if !su.IsNil() {
						sv := su.Elem().Elem().Field(0)
						if sv.Kind() == reflect.Ptr && sv.IsNil() {
							return
						}
						switch sv.Type().Kind() {
						case reflect.Ptr: // Proto struct (e.g., *T)
							DiscardUnknown(sv.Interface().(Message))
						}
					}
				}
			}
		default:
			continue
		}
		di.fields = append(di.fields, dfi)
	}

	di.unrecognized = invalidField
	if f, ok := t.FieldByName("XXX_unrecognized"); ok {
		if f.Type != reflect.TypeOf([]byte{}) {
			panic("expected XXX_unrecognized to be of type []byte")
		}
		di.unrecognized = toField(&f)
	}

	atomic.StoreInt32(&di.initialized, 1)
}

func discardLegacy(m Message) {
	v := reflect.ValueOf(m)
	if v.Kind() != reflect.Ptr || v.IsNil() {
		return
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return
	}
	t := v.Type()

	for i := 0; i < v.NumField(); i++ {
		f := t.Field(i)
		if strings.HasPrefix(f.Name, "XXX_") {
			continue
		}
		vf := v.Field(i)
		tf := f.Type

		// Unwrap tf to get its most basic type.
		var isPointer, isSlice bool
		if tf.Kind() == reflect.Slice && tf.Elem().Kind() != reflect.Uint8 {
			isSlice = true
			tf = tf.Elem()
		}
		if tf.Kind() == reflect.Ptr {
			isPointer = true
			tf = tf.Elem()
		}
		if isPointer && isSlice && tf.Kind() != reflect.Struct {
			panic(fmt.Sprintf("%T.%s cannot be a slice of pointers to primitive types", m, f.Name))
		}

		switch tf.Kind() {
		case reflect.Struct:
			switch {
			case !isPointer:
				panic(fmt.Sprintf("%T.%s cannot be a direct struct value", m, f.Name))
			case isSlice: // E.g., []*pb.T
				for j := 0; j < vf.Len(); j++ {
					discardLegacy(vf.Index(j).Interface().(Message))
				}
			default: // E.g., *pb.T
				discardLegacy(vf.Interface().(Message))
			}
		case reflect.Map:
			switch {
			case isPointer || isSlice:
				panic(fmt.Sprintf("%T.%s cannot be a pointer to a map or a slice of map values", m, f.Name))
			default: // E.g., map[K]V
				tv := vf.Type().Elem()
				if tv.Kind() == reflect.Ptr && tv.Implements(protoMessageType) { // Proto struct (e.g., *T)
					for _, key := range vf.MapKeys() {
						val := vf.MapIndex(key)
						discardLegacy(val.Interface().(Message))
					}
				}
			}
		case reflect.Interface:
			// Must be oneof field.
			switch {
			case isPointer || isSlice:
				panic(fmt.Sprintf("%T.%s cannot be a pointer to a interface or a slice of interface values", m, f.Name))
			default: // E.g., test_proto.isCommunique_Union interface
				if !vf.IsNil() && f.Tag.Get("protobuf_oneof") != "" {
					vf = vf.Elem() // E.g., *test_proto.Communique_Msg
					if !vf.IsNil() {
						vf = vf.Elem()   // E.g., test_proto.Communique_Msg
						vf = vf.Field(0) // E.g., Proto struct (e.g., *T) or primitive value
						if vf.Kind() == reflect.Ptr {
							discardLegacy(vf.Interface().(Message))
						}
					}
				}
			}
		}
	}

	if vf := v.FieldByName("XXX_unrecognized"); vf.IsValid() {
		if vf.Type() != reflect.TypeOf([]byte{}) {
			panic("expected XXX_unrecognized to be of type []byte")
		}
		vf.Set(reflect.ValueOf([]byte(nil)))
	}

	// For proto2 messages, only discard unknown fields in message extensions
	// that have been accessed via GetExtension.
	if em, err := extendable(m); err == nil {
		// Ignore lock since discardLegacy is not concurrency safe.
		emm, _ := em.extensionsRead()
		for _, mx := range emm {
			if m, ok := mx.value.(Message); ok {
				discardLegacy(m)
			}
		}
	}
}
