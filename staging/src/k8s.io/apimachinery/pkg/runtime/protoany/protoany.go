/*
Copyright 2026 The Kubernetes Authors.

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

// Package protoany serializes Go interface values as google.protobuf.Any.
//
// go-to-protobuf maps interface-typed fields to google.protobuf.Any and
// rewrites the generated marshalers to call into this package. Because the
// concrete type behind an interface is only known at runtime, every concrete
// type that may be stored in such a field must be registered with Register,
// which supplies the type URL written on the wire and the constructor used
// when decoding.
package protoany

import (
	"errors"
	"fmt"
	"reflect"
	"sync"

	"google.golang.org/protobuf/encoding/protowire"
)

// Message is the method set go-to-protobuf generates for every message type.
type Message interface {
	Size() int
	MarshalToSizedBuffer(dAtA []byte) (int, error)
	Unmarshal(dAtA []byte) error
}

var (
	registryLock sync.RWMutex
	newByURL     = map[string]func() Message{}
	urlByType    = map[reflect.Type]string{}
)

// Register associates typeURL with the concrete type returned by newFn. It is
// intended to be called from init, and panics on conflicting registrations
// since those are programming errors that would otherwise corrupt the wire format.
func Register(typeURL string, newFn func() Message) {
	if len(typeURL) == 0 {
		panic("protoany: Register called with an empty type URL")
	}
	if newFn == nil {
		panic(fmt.Sprintf("protoany: Register called with a nil constructor for %q", typeURL))
	}
	t := reflect.TypeOf(newFn())
	if t == nil {
		panic(fmt.Sprintf("protoany: constructor for %q returned a nil interface", typeURL))
	}

	registryLock.Lock()
	defer registryLock.Unlock()
	if _, ok := newByURL[typeURL]; ok {
		panic(fmt.Sprintf("protoany: type URL %q is already registered", typeURL))
	}
	if existing, ok := urlByType[t]; ok {
		panic(fmt.Sprintf("protoany: type %v is already registered as %q", t, existing))
	}
	newByURL[typeURL] = newFn
	urlByType[t] = typeURL
}

// TypeURL returns the type URL registered for the concrete type of v.
func TypeURL(v any) (string, bool) {
	registryLock.RLock()
	defer registryLock.RUnlock()
	url, ok := urlByType[reflect.TypeOf(v)]
	return url, ok
}

func lookup(v any) (string, Message, error) {
	// Generated code skips nil singular fields, so nil only arrives here as an
	// element of a repeated field, which protobuf cannot represent.
	if v == nil {
		return "", nil, errors.New("protoany: cannot marshal a nil interface value; repeated google.protobuf.Any fields must not contain nil elements")
	}
	if rv := reflect.ValueOf(v); rv.Kind() == reflect.Pointer && rv.IsNil() {
		return "", nil, fmt.Errorf("protoany: cannot marshal a nil %T; set the field to a nil interface instead", v)
	}
	url, ok := TypeURL(v)
	if !ok {
		return "", nil, fmt.Errorf("protoany: type %T has no registered type URL; call protoany.Register for it before marshaling", v)
	}
	// Registration constructs a Message of this exact type, so this only fails
	// if the registered constructor lies about its result.
	msg, ok := v.(Message)
	if !ok {
		return "", nil, fmt.Errorf("protoany: type %T registered as %q does not implement protoany.Message", v, url)
	}
	return url, msg, nil
}

// Field numbers of google.protobuf.Any.
const (
	typeURLField protowire.Number = 1
	valueField   protowire.Number = 2
)

// Size returns the encoded size of v as a google.protobuf.Any. Unregistered
// types report 0; MarshalToSizedBuffer returns the corresponding error.
func Size(v any) int {
	url, msg, err := lookup(v)
	if err != nil {
		return 0
	}
	return size(url, msg.Size())
}

func size(url string, valueLen int) int {
	n := protowire.SizeTag(typeURLField) + protowire.SizeBytes(len(url))
	// proto3 omits empty bytes fields
	if valueLen > 0 {
		n += protowire.SizeTag(valueField) + protowire.SizeBytes(valueLen)
	}
	return n
}

// MarshalToSizedBuffer encodes v as a google.protobuf.Any into the tail of
// dAtA, which must have been sized using Size, and returns the bytes written.
// Writing to the tail matches the backwards-marshaling convention of the
// gogo-generated code that calls this.
func MarshalToSizedBuffer(v any, dAtA []byte) (int, error) {
	url, msg, err := lookup(v)
	if err != nil {
		return 0, err
	}
	valueLen := msg.Size()
	total := size(url, valueLen)
	start := len(dAtA) - total
	if start < 0 {
		return 0, fmt.Errorf("protoany: buffer of %d bytes is too small to marshal %T, which needs %d", len(dAtA), v, total)
	}

	// The full slice expression caps appends at len(dAtA), so the header is
	// written in place and can never spill into a reallocated buffer unnoticed.
	b := protowire.AppendTag(dAtA[start:start:len(dAtA)], typeURLField, protowire.BytesType)
	b = protowire.AppendString(b, url)
	if valueLen > 0 {
		b = protowire.AppendTag(b, valueField, protowire.BytesType)
		b = protowire.AppendVarint(b, uint64(valueLen))
		n, err := msg.MarshalToSizedBuffer(dAtA[:start+len(b)+valueLen])
		if err != nil {
			return 0, err
		}
		if n != valueLen {
			return 0, fmt.Errorf("protoany: %T reported size %d but marshaled %d bytes", v, valueLen, n)
		}
	}
	return total, nil
}

// Unmarshal decodes a google.protobuf.Any from dAtA into the registered
// concrete type and stores it in into. It is generic so generated code can
// pass a pointer to an interface-typed field without naming the interface.
func Unmarshal[T any](dAtA []byte, into *T) error {
	url, value, err := decodeAny(dAtA)
	if err != nil {
		return err
	}
	if len(url) == 0 {
		return errors.New("protoany: google.protobuf.Any has an empty type URL")
	}

	registryLock.RLock()
	newFn, ok := newByURL[url]
	registryLock.RUnlock()
	if !ok {
		return fmt.Errorf("protoany: no type registered for type URL %q; call protoany.Register for it before unmarshaling", url)
	}

	msg := newFn()
	if err := msg.Unmarshal(value); err != nil {
		return fmt.Errorf("protoany: decoding %q: %w", url, err)
	}
	typed, ok := any(msg).(T)
	if !ok {
		return fmt.Errorf("protoany: type %T registered for %q is not assignable to field of type %v", msg, url, reflect.TypeOf(into).Elem())
	}
	*into = typed
	return nil
}

// decodeAny returns the type_url and value fields of an encoded
// google.protobuf.Any, skipping unknown fields.
func decodeAny(dAtA []byte) (url string, value []byte, err error) {
	for len(dAtA) > 0 {
		num, typ, n := protowire.ConsumeTag(dAtA)
		if n < 0 {
			return "", nil, fmt.Errorf("protoany: decoding google.protobuf.Any: %w", protowire.ParseError(n))
		}
		dAtA = dAtA[n:]

		if num == typeURLField || num == valueField {
			if typ != protowire.BytesType {
				return "", nil, fmt.Errorf("protoany: wrong wire type %d for google.protobuf.Any field %d", typ, num)
			}
			b, n := protowire.ConsumeBytes(dAtA)
			if n < 0 {
				return "", nil, fmt.Errorf("protoany: decoding google.protobuf.Any field %d: %w", num, protowire.ParseError(n))
			}
			if num == typeURLField {
				url = string(b)
			} else {
				value = b
			}
			dAtA = dAtA[n:]
			continue
		}

		n = protowire.ConsumeFieldValue(num, typ, dAtA)
		if n < 0 {
			return "", nil, fmt.Errorf("protoany: skipping unknown google.protobuf.Any field %d: %w", num, protowire.ParseError(n))
		}
		dAtA = dAtA[n:]
	}
	return url, value, nil
}
