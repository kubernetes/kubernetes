// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"errors"
	"fmt"
	"reflect"
	"sync"
)

// SkipFunc may be returned by MarshalFuncV2 and UnmarshalFuncV2 functions.
//
// Any function that returns SkipFunc must not cause observable side effects
// on the provided Encoder or Decoder. For example, it is permissible to call
// Decoder.PeekKind, but not permissible to call Decoder.ReadToken or
// Encoder.WriteToken since such methods mutate the state.
const SkipFunc = jsonError("skip function")

// Marshalers is a list of functions that may override the marshal behavior
// of specific types. Populate MarshalOptions.Marshalers to use it.
// A nil *Marshalers is equivalent to an empty list.
type Marshalers = typedMarshalers

// NewMarshalers constructs a flattened list of marshal functions.
// If multiple functions in the list are applicable for a value of a given type,
// then those earlier in the list take precedence over those that come later.
// If a function returns SkipFunc, then the next applicable function is called,
// otherwise the default marshaling behavior is used.
//
// For example:
//
//	m1 := NewMarshalers(f1, f2)
//	m2 := NewMarshalers(f0, m1, f3)     // equivalent to m3
//	m3 := NewMarshalers(f0, f1, f2, f3) // equivalent to m2
func NewMarshalers(ms ...*Marshalers) *Marshalers {
	return newMarshalers(ms...)
}

// Unmarshalers is a list of functions that may override the unmarshal behavior
// of specific types. Populate UnmarshalOptions.Unmarshalers to use it.
// A nil *Unmarshalers is equivalent to an empty list.
type Unmarshalers = typedUnmarshalers

// NewUnmarshalers constructs a flattened list of unmarshal functions.
// If multiple functions in the list are applicable for a value of a given type,
// then those earlier in the list take precedence over those that come later.
// If a function returns SkipFunc, then the next applicable function is called,
// otherwise the default unmarshaling behavior is used.
//
// For example:
//
//	u1 := NewUnmarshalers(f1, f2)
//	u2 := NewUnmarshalers(f0, u1, f3)     // equivalent to u3
//	u3 := NewUnmarshalers(f0, f1, f2, f3) // equivalent to u2
func NewUnmarshalers(us ...*Unmarshalers) *Unmarshalers {
	return newUnmarshalers(us...)
}

type typedMarshalers = typedArshalers[MarshalOptions, Encoder]
type typedUnmarshalers = typedArshalers[UnmarshalOptions, Decoder]
type typedArshalers[Options, Coder any] struct {
	nonComparable

	fncVals  []typedArshaler[Options, Coder]
	fncCache sync.Map // map[reflect.Type]arshaler

	// fromAny reports whether any of Go types used to represent arbitrary JSON
	// (i.e., any, bool, string, float64, map[string]any, or []any) matches
	// any of the provided type-specific arshalers.
	//
	// This bit of information is needed in arshal_default.go to determine
	// whether to use the specialized logic in arshal_any.go to handle
	// the any interface type. The logic in arshal_any.go does not support
	// type-specific arshal functions, so we must avoid using that logic
	// if this is true.
	fromAny bool
}
type typedMarshaler = typedArshaler[MarshalOptions, Encoder]
type typedUnmarshaler = typedArshaler[UnmarshalOptions, Decoder]
type typedArshaler[Options, Coder any] struct {
	typ     reflect.Type
	fnc     func(Options, *Coder, addressableValue) error
	maySkip bool
}

func newMarshalers(ms ...*Marshalers) *Marshalers       { return newTypedArshalers(ms...) }
func newUnmarshalers(us ...*Unmarshalers) *Unmarshalers { return newTypedArshalers(us...) }
func newTypedArshalers[Options, Coder any](as ...*typedArshalers[Options, Coder]) *typedArshalers[Options, Coder] {
	var a typedArshalers[Options, Coder]
	for _, a2 := range as {
		if a2 != nil {
			a.fncVals = append(a.fncVals, a2.fncVals...)
			a.fromAny = a.fromAny || a2.fromAny
		}
	}
	if len(a.fncVals) == 0 {
		return nil
	}
	return &a
}

func (a *typedArshalers[Options, Coder]) lookup(fnc func(Options, *Coder, addressableValue) error, t reflect.Type) (func(Options, *Coder, addressableValue) error, bool) {
	if a == nil {
		return fnc, false
	}
	if v, ok := a.fncCache.Load(t); ok {
		if v == nil {
			return fnc, false
		}
		return v.(func(Options, *Coder, addressableValue) error), true
	}

	// Collect a list of arshalers that can be called for this type.
	// This list may be longer than 1 since some arshalers can be skipped.
	var fncs []func(Options, *Coder, addressableValue) error
	for _, fncVal := range a.fncVals {
		if !castableTo(t, fncVal.typ) {
			continue
		}
		fncs = append(fncs, fncVal.fnc)
		if !fncVal.maySkip {
			break // subsequent arshalers will never be called
		}
	}

	if len(fncs) == 0 {
		a.fncCache.Store(t, nil) // nil to indicate that no funcs found
		return fnc, false
	}

	// Construct an arshaler that may call every applicable arshaler.
	fncDefault := fnc
	fnc = func(o Options, c *Coder, v addressableValue) error {
		for _, fnc := range fncs {
			if err := fnc(o, c, v); err != SkipFunc {
				return err // may be nil or non-nil
			}
		}
		return fncDefault(o, c, v)
	}

	// Use the first stored so duplicate work can be garbage collected.
	v, _ := a.fncCache.LoadOrStore(t, fnc)
	return v.(func(Options, *Coder, addressableValue) error), true
}

// MarshalFuncV1 constructs a type-specific marshaler that
// specifies how to marshal values of type T.
// T can be any type except a named pointer.
// The function is always provided with a non-nil pointer value
// if T is an interface or pointer type.
//
// The function must marshal exactly one JSON value.
// The value of T must not be retained outside the function call.
// It may not return SkipFunc.
func MarshalFuncV1[T any](fn func(T) ([]byte, error)) *Marshalers {
	t := reflect.TypeOf((*T)(nil)).Elem()
	assertCastableTo(t, true)
	typFnc := typedMarshaler{
		typ: t,
		fnc: func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
			val, err := fn(va.castTo(t).Interface().(T))
			if err != nil {
				err = wrapSkipFunc(err, "marshal function of type func(T) ([]byte, error)")
				// TODO: Avoid wrapping semantic errors.
				return &SemanticError{action: "marshal", GoType: t, Err: err}
			}
			if err := enc.WriteValue(val); err != nil {
				// TODO: Avoid wrapping semantic or I/O errors.
				return &SemanticError{action: "marshal", JSONKind: RawValue(val).Kind(), GoType: t, Err: err}
			}
			return nil
		},
	}
	return &Marshalers{fncVals: []typedMarshaler{typFnc}, fromAny: castableToFromAny(t)}
}

// MarshalFuncV2 constructs a type-specific marshaler that
// specifies how to marshal values of type T.
// T can be any type except a named pointer.
// The function is always provided with a non-nil pointer value
// if T is an interface or pointer type.
//
// The function must marshal exactly one JSON value by calling write methods
// on the provided encoder. It may return SkipFunc such that marshaling can
// move on to the next marshal function. However, no mutable method calls may
// be called on the encoder if SkipFunc is returned.
// The pointer to Encoder and the value of T must not be retained
// outside the function call.
func MarshalFuncV2[T any](fn func(MarshalOptions, *Encoder, T) error) *Marshalers {
	t := reflect.TypeOf((*T)(nil)).Elem()
	assertCastableTo(t, true)
	typFnc := typedMarshaler{
		typ: t,
		fnc: func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
			prevDepth, prevLength := enc.tokens.depthLength()
			err := fn(mo, enc, va.castTo(t).Interface().(T))
			currDepth, currLength := enc.tokens.depthLength()
			if err == nil && (prevDepth != currDepth || prevLength+1 != currLength) {
				err = errors.New("must write exactly one JSON value")
			}
			if err != nil {
				if err == SkipFunc {
					if prevDepth == currDepth && prevLength == currLength {
						return SkipFunc
					}
					err = errors.New("must not write any JSON tokens when skipping")
				}
				// TODO: Avoid wrapping semantic or I/O errors.
				return &SemanticError{action: "marshal", GoType: t, Err: err}
			}
			return nil
		},
		maySkip: true,
	}
	return &Marshalers{fncVals: []typedMarshaler{typFnc}, fromAny: castableToFromAny(t)}
}

// UnmarshalFuncV1 constructs a type-specific unmarshaler that
// specifies how to unmarshal values of type T.
// T must be an unnamed pointer or an interface type.
// The function is always provided with a non-nil pointer value.
//
// The function must unmarshal exactly one JSON value.
// The input []byte must not be mutated.
// The input []byte and value T must not be retained outside the function call.
// It may not return SkipFunc.
func UnmarshalFuncV1[T any](fn func([]byte, T) error) *Unmarshalers {
	t := reflect.TypeOf((*T)(nil)).Elem()
	assertCastableTo(t, false)
	typFnc := typedUnmarshaler{
		typ: t,
		fnc: func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
			val, err := dec.ReadValue()
			if err != nil {
				return err // must be a syntactic or I/O error
			}
			err = fn(val, va.castTo(t).Interface().(T))
			if err != nil {
				err = wrapSkipFunc(err, "unmarshal function of type func([]byte, T) error")
				// TODO: Avoid wrapping semantic, syntactic, or I/O errors.
				return &SemanticError{action: "unmarshal", JSONKind: val.Kind(), GoType: t, Err: err}
			}
			return nil
		},
	}
	return &Unmarshalers{fncVals: []typedUnmarshaler{typFnc}, fromAny: castableToFromAny(t)}
}

// UnmarshalFuncV2 constructs a type-specific unmarshaler that
// specifies how to unmarshal values of type T.
// T must be an unnamed pointer or an interface type.
// The function is always provided with a non-nil pointer value.
//
// The function must unmarshal exactly one JSON value by calling read methods
// on the provided decoder. It may return SkipFunc such that unmarshaling can
// move on to the next unmarshal function. However, no mutable method calls may
// be called on the decoder if SkipFunc is returned.
// The pointer to Decoder and the value of T must not be retained
// outside the function call.
func UnmarshalFuncV2[T any](fn func(UnmarshalOptions, *Decoder, T) error) *Unmarshalers {
	t := reflect.TypeOf((*T)(nil)).Elem()
	assertCastableTo(t, false)
	typFnc := typedUnmarshaler{
		typ: t,
		fnc: func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
			prevDepth, prevLength := dec.tokens.depthLength()
			err := fn(uo, dec, va.castTo(t).Interface().(T))
			currDepth, currLength := dec.tokens.depthLength()
			if err == nil && (prevDepth != currDepth || prevLength+1 != currLength) {
				err = errors.New("must read exactly one JSON value")
			}
			if err != nil {
				if err == SkipFunc {
					if prevDepth == currDepth && prevLength == currLength {
						return SkipFunc
					}
					err = errors.New("must not read any JSON tokens when skipping")
				}
				// TODO: Avoid wrapping semantic, syntactic, or I/O errors.
				return &SemanticError{action: "unmarshal", GoType: t, Err: err}
			}
			return nil
		},
		maySkip: true,
	}
	return &Unmarshalers{fncVals: []typedUnmarshaler{typFnc}, fromAny: castableToFromAny(t)}
}

// assertCastableTo asserts that "to" is a valid type to be casted to.
// These are the Go types that type-specific arshalers may operate upon.
//
// Let AllTypes be the universal set of all possible Go types.
// This function generally asserts that:
//
//	len([from for from in AllTypes if castableTo(from, to)]) > 0
//
// otherwise it panics.
//
// As a special-case if marshal is false, then we forbid any non-pointer or
// non-interface type since it is almost always a bug trying to unmarshal
// into something where the end-user caller did not pass in an addressable value
// since they will not observe the mutations.
func assertCastableTo(to reflect.Type, marshal bool) {
	switch to.Kind() {
	case reflect.Interface:
		return
	case reflect.Pointer:
		// Only allow unnamed pointers to be consistent with the fact that
		// taking the address of a value produces an unnamed pointer type.
		if to.Name() == "" {
			return
		}
	default:
		// Technically, non-pointer types are permissible for unmarshal.
		// However, they are often a bug since the receiver would be immutable.
		// Thus, only allow them for marshaling.
		if marshal {
			return
		}
	}
	if marshal {
		panic(fmt.Sprintf("input type %v must be an interface type, an unnamed pointer type, or a non-pointer type", to))
	} else {
		panic(fmt.Sprintf("input type %v must be an interface type or an unnamed pointer type", to))
	}
}

// castableTo checks whether values of type "from" can be casted to type "to".
// Nil pointer or interface "from" values are never considered castable.
//
// This function must be kept in sync with addressableValue.castTo.
func castableTo(from, to reflect.Type) bool {
	switch to.Kind() {
	case reflect.Interface:
		// TODO: This breaks when ordinary interfaces can have type sets
		// since interfaces now exist where only the value form of a type (T)
		// implements the interface, but not the pointer variant (*T).
		// See https://go.dev/issue/45346.
		return reflect.PointerTo(from).Implements(to)
	case reflect.Pointer:
		// Common case for unmarshaling.
		// From must be a concrete or interface type.
		return reflect.PointerTo(from) == to
	default:
		// Common case for marshaling.
		// From must be a concrete type.
		return from == to
	}
}

// castTo casts va to the specified type.
// If the type is an interface, then the underlying type will always
// be a non-nil pointer to a concrete type.
//
// Requirement: castableTo(va.Type(), to) must hold.
func (va addressableValue) castTo(to reflect.Type) reflect.Value {
	switch to.Kind() {
	case reflect.Interface:
		return va.Addr().Convert(to)
	case reflect.Pointer:
		return va.Addr()
	default:
		return va.Value
	}
}

// castableToFromAny reports whether "to" can be casted to from any
// of the dynamic types used to represent arbitrary JSON.
func castableToFromAny(to reflect.Type) bool {
	for _, from := range []reflect.Type{anyType, boolType, stringType, float64Type, mapStringAnyType, sliceAnyType} {
		if castableTo(from, to) {
			return true
		}
	}
	return false
}

func wrapSkipFunc(err error, what string) error {
	if err == SkipFunc {
		return errors.New(what + " cannot be skipped")
	}
	return err
}
