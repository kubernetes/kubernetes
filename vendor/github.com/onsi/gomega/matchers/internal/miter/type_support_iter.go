//go:build go1.23

package miter

import (
	"reflect"
)

// HasIterators always returns false for Go versions before 1.23.
func HasIterators() bool { return true }

// IsIter returns true if the specified value is a function type that can be
// range-d over, otherwise false.
//
// We don't use reflect's CanSeq and CanSeq2 directly, as these would return
// true also for other value types that are range-able, such as integers,
// slices, et cetera. Here, we aim only at range-able (iterator) functions.
func IsIter(it any) bool {
	if it == nil { // on purpose we only test for untyped nil.
		return false
	}
	// reject all non-iterator-func values, even if they're range-able.
	t := reflect.TypeOf(it)
	if t.Kind() != reflect.Func {
		return false
	}
	return t.CanSeq() || t.CanSeq2()
}

// IterKVTypes returns the reflection types of an iterator's yield function's K
// and optional V arguments, otherwise nil K and V reflection types.
func IterKVTypes(it any) (k, v reflect.Type) {
	if it == nil {
		return
	}
	// reject all non-iterator-func values, even if they're range-able.
	t := reflect.TypeOf(it)
	if t.Kind() != reflect.Func {
		return
	}
	// get the reflection types for V, and where applicable, K.
	switch {
	case t.CanSeq():
		v = t. /*iterator fn*/ In(0). /*yield fn*/ In(0)
	case t.CanSeq2():
		yieldfn := t. /*iterator fn*/ In(0)
		k = yieldfn.In(0)
		v = yieldfn.In(1)
	}
	return
}

// IsSeq2 returns true if the passed iterator function is compatible with
// iter.Seq2, otherwise false.
//
// IsSeq2 hides the Go 1.23+ specific reflect.Type.CanSeq2 behind a facade which
// is empty for Go versions before 1.23.
func IsSeq2(it any) bool {
	if it == nil {
		return false
	}
	t := reflect.TypeOf(it)
	return t.Kind() == reflect.Func && t.CanSeq2()
}

// isNilly returns true if v is either an untyped nil, or is a nil function (not
// necessarily an iterator function).
func isNilly(v any) bool {
	if v == nil {
		return true
	}
	rv := reflect.ValueOf(v)
	return rv.Kind() == reflect.Func && rv.IsNil()
}

// IterateV loops over the elements produced by an iterator function, passing
// the elements to the specified yield function individually and stopping only
// when either the iterator function runs out of elements or the yield function
// tell us to stop it.
//
// IterateV works very much like reflect.Value.Seq but hides the Go 1.23+
// specific parts behind a facade which is empty for Go versions before 1.23, in
// order to simplify code maintenance for matchers when using older Go versions.
func IterateV(it any, yield func(v reflect.Value) bool) {
	if isNilly(it) {
		return
	}
	// reject all non-iterator-func values, even if they're range-able.
	t := reflect.TypeOf(it)
	if t.Kind() != reflect.Func || !t.CanSeq() {
		return
	}
	// Call the specified iterator function, handing it our adaptor to call the
	// specified generic reflection yield function.
	reflectedYield := reflect.MakeFunc(
		t. /*iterator fn*/ In(0),
		func(args []reflect.Value) []reflect.Value {
			return []reflect.Value{reflect.ValueOf(yield(args[0]))}
		})
	reflect.ValueOf(it).Call([]reflect.Value{reflectedYield})
}

// IterateKV loops over the key-value elements produced by an iterator function,
// passing the elements to the specified yield function individually and
// stopping only when either the iterator function runs out of elements or the
// yield function tell us to stop it.
//
// IterateKV works very much like reflect.Value.Seq2 but hides the Go 1.23+
// specific parts behind a facade which is empty for Go versions before 1.23, in
// order to simplify code maintenance for matchers when using older Go versions.
func IterateKV(it any, yield func(k, v reflect.Value) bool) {
	if isNilly(it) {
		return
	}
	// reject all non-iterator-func values, even if they're range-able.
	t := reflect.TypeOf(it)
	if t.Kind() != reflect.Func || !t.CanSeq2() {
		return
	}
	// Call the specified iterator function, handing it our adaptor to call the
	// specified generic reflection yield function.
	reflectedYield := reflect.MakeFunc(
		t. /*iterator fn*/ In(0),
		func(args []reflect.Value) []reflect.Value {
			return []reflect.Value{reflect.ValueOf(yield(args[0], args[1]))}
		})
	reflect.ValueOf(it).Call([]reflect.Value{reflectedYield})
}
