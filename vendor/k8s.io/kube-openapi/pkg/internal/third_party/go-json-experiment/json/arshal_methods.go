// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"encoding"
	"errors"
	"reflect"
)

// Interfaces for custom serialization.
var (
	jsonMarshalerV1Type   = reflect.TypeOf((*MarshalerV1)(nil)).Elem()
	jsonMarshalerV2Type   = reflect.TypeOf((*MarshalerV2)(nil)).Elem()
	jsonUnmarshalerV1Type = reflect.TypeOf((*UnmarshalerV1)(nil)).Elem()
	jsonUnmarshalerV2Type = reflect.TypeOf((*UnmarshalerV2)(nil)).Elem()
	textMarshalerType     = reflect.TypeOf((*encoding.TextMarshaler)(nil)).Elem()
	textUnmarshalerType   = reflect.TypeOf((*encoding.TextUnmarshaler)(nil)).Elem()
)

// MarshalerV1 is implemented by types that can marshal themselves.
// It is recommended that types implement MarshalerV2 unless
// the implementation is trying to avoid a hard dependency on this package.
//
// It is recommended that implementations return a buffer that is safe
// for the caller to retain and potentially mutate.
type MarshalerV1 interface {
	MarshalJSON() ([]byte, error)
}

// MarshalerV2 is implemented by types that can marshal themselves.
// It is recommended that types implement MarshalerV2 instead of MarshalerV1
// since this is both more performant and flexible.
// If a type implements both MarshalerV1 and MarshalerV2,
// then MarshalerV2 takes precedence. In such a case, both implementations
// should aim to have equivalent behavior for the default marshal options.
//
// The implementation must write only one JSON value to the Encoder and
// must not retain the pointer to Encoder.
type MarshalerV2 interface {
	MarshalNextJSON(MarshalOptions, *Encoder) error

	// TODO: Should users call the MarshalOptions.MarshalNext method or
	// should/can they call this method directly? Does it matter?
}

// UnmarshalerV1 is implemented by types that can unmarshal themselves.
// It is recommended that types implement UnmarshalerV2 unless
// the implementation is trying to avoid a hard dependency on this package.
//
// The input can be assumed to be a valid encoding of a JSON value
// if called from unmarshal functionality in this package.
// UnmarshalJSON must copy the JSON data if it is retained after returning.
// It is recommended that UnmarshalJSON implement merge semantics when
// unmarshaling into a pre-populated value.
//
// Implementations must not retain or mutate the input []byte.
type UnmarshalerV1 interface {
	UnmarshalJSON([]byte) error
}

// UnmarshalerV2 is implemented by types that can unmarshal themselves.
// It is recommended that types implement UnmarshalerV2 instead of UnmarshalerV1
// since this is both more performant and flexible.
// If a type implements both UnmarshalerV1 and UnmarshalerV2,
// then UnmarshalerV2 takes precedence. In such a case, both implementations
// should aim to have equivalent behavior for the default unmarshal options.
//
// The implementation must read only one JSON value from the Decoder.
// It is recommended that UnmarshalNextJSON implement merge semantics when
// unmarshaling into a pre-populated value.
//
// Implementations must not retain the pointer to Decoder.
type UnmarshalerV2 interface {
	UnmarshalNextJSON(UnmarshalOptions, *Decoder) error

	// TODO: Should users call the UnmarshalOptions.UnmarshalNext method or
	// should/can they call this method directly? Does it matter?
}

func makeMethodArshaler(fncs *arshaler, t reflect.Type) *arshaler {
	// Avoid injecting method arshaler on the pointer or interface version
	// to avoid ever calling the method on a nil pointer or interface receiver.
	// Let it be injected on the value receiver (which is always addressable).
	if t.Kind() == reflect.Pointer || t.Kind() == reflect.Interface {
		return fncs
	}

	// Handle custom marshaler.
	switch which, needAddr := implementsWhich(t, jsonMarshalerV2Type, jsonMarshalerV1Type, textMarshalerType); which {
	case jsonMarshalerV2Type:
		fncs.nonDefault = true
		fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
			prevDepth, prevLength := enc.tokens.depthLength()
			err := va.addrWhen(needAddr).Interface().(MarshalerV2).MarshalNextJSON(mo, enc)
			currDepth, currLength := enc.tokens.depthLength()
			if (prevDepth != currDepth || prevLength+1 != currLength) && err == nil {
				err = errors.New("must write exactly one JSON value")
			}
			if err != nil {
				err = wrapSkipFunc(err, "marshal method")
				// TODO: Avoid wrapping semantic or I/O errors.
				return &SemanticError{action: "marshal", GoType: t, Err: err}
			}
			return nil
		}
	case jsonMarshalerV1Type:
		fncs.nonDefault = true
		fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
			marshaler := va.addrWhen(needAddr).Interface().(MarshalerV1)
			val, err := marshaler.MarshalJSON()
			if err != nil {
				err = wrapSkipFunc(err, "marshal method")
				// TODO: Avoid wrapping semantic errors.
				return &SemanticError{action: "marshal", GoType: t, Err: err}
			}
			if err := enc.WriteValue(val); err != nil {
				// TODO: Avoid wrapping semantic or I/O errors.
				return &SemanticError{action: "marshal", JSONKind: RawValue(val).Kind(), GoType: t, Err: err}
			}
			return nil
		}
	case textMarshalerType:
		fncs.nonDefault = true
		fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
			marshaler := va.addrWhen(needAddr).Interface().(encoding.TextMarshaler)
			s, err := marshaler.MarshalText()
			if err != nil {
				err = wrapSkipFunc(err, "marshal method")
				// TODO: Avoid wrapping semantic errors.
				return &SemanticError{action: "marshal", JSONKind: '"', GoType: t, Err: err}
			}
			val := enc.UnusedBuffer()
			val, err = appendString(val, string(s), true, nil)
			if err != nil {
				return &SemanticError{action: "marshal", JSONKind: '"', GoType: t, Err: err}
			}
			if err := enc.WriteValue(val); err != nil {
				// TODO: Avoid wrapping syntactic or I/O errors.
				return &SemanticError{action: "marshal", JSONKind: '"', GoType: t, Err: err}
			}
			return nil
		}
	}

	// Handle custom unmarshaler.
	switch which, needAddr := implementsWhich(t, jsonUnmarshalerV2Type, jsonUnmarshalerV1Type, textUnmarshalerType); which {
	case jsonUnmarshalerV2Type:
		fncs.nonDefault = true
		fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
			prevDepth, prevLength := dec.tokens.depthLength()
			err := va.addrWhen(needAddr).Interface().(UnmarshalerV2).UnmarshalNextJSON(uo, dec)
			currDepth, currLength := dec.tokens.depthLength()
			if (prevDepth != currDepth || prevLength+1 != currLength) && err == nil {
				err = errors.New("must read exactly one JSON value")
			}
			if err != nil {
				err = wrapSkipFunc(err, "unmarshal method")
				// TODO: Avoid wrapping semantic, syntactic, or I/O errors.
				return &SemanticError{action: "unmarshal", GoType: t, Err: err}
			}
			return nil
		}
	case jsonUnmarshalerV1Type:
		fncs.nonDefault = true
		fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
			val, err := dec.ReadValue()
			if err != nil {
				return err // must be a syntactic or I/O error
			}
			unmarshaler := va.addrWhen(needAddr).Interface().(UnmarshalerV1)
			if err := unmarshaler.UnmarshalJSON(val); err != nil {
				err = wrapSkipFunc(err, "unmarshal method")
				// TODO: Avoid wrapping semantic, syntactic, or I/O errors.
				return &SemanticError{action: "unmarshal", JSONKind: val.Kind(), GoType: t, Err: err}
			}
			return nil
		}
	case textUnmarshalerType:
		fncs.nonDefault = true
		fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
			var flags valueFlags
			val, err := dec.readValue(&flags)
			if err != nil {
				return err // must be a syntactic or I/O error
			}
			if val.Kind() != '"' {
				err = errors.New("JSON value must be string type")
				return &SemanticError{action: "unmarshal", JSONKind: val.Kind(), GoType: t, Err: err}
			}
			s := unescapeStringMayCopy(val, flags.isVerbatim())
			unmarshaler := va.addrWhen(needAddr).Interface().(encoding.TextUnmarshaler)
			if err := unmarshaler.UnmarshalText(s); err != nil {
				err = wrapSkipFunc(err, "unmarshal method")
				// TODO: Avoid wrapping semantic, syntactic, or I/O errors.
				return &SemanticError{action: "unmarshal", JSONKind: val.Kind(), GoType: t, Err: err}
			}
			return nil
		}
	}

	return fncs
}

// implementsWhich is like t.Implements(ifaceType) for a list of interfaces,
// but checks whether either t or reflect.PointerTo(t) implements the interface.
// It returns the first interface type that matches and whether a value of t
// needs to be addressed first before it implements the interface.
func implementsWhich(t reflect.Type, ifaceTypes ...reflect.Type) (which reflect.Type, needAddr bool) {
	for _, ifaceType := range ifaceTypes {
		switch {
		case t.Implements(ifaceType):
			return ifaceType, false
		case reflect.PointerTo(t).Implements(ifaceType):
			return ifaceType, true
		}
	}
	return nil, false
}

// addrWhen returns va.Addr if addr is specified, otherwise it returns itself.
func (va addressableValue) addrWhen(addr bool) reflect.Value {
	if addr {
		return va.Addr()
	}
	return va.Value
}
