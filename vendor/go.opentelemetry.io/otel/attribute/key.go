// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package attribute

// Key represents the key part in key-value pairs. It's a string. The
// allowed character set in the key depends on the use of the key.
type Key string

// Bool returns a [KeyValue] for a bool value.
//
// If creating both a key and value at the same time, use the package-level
// [Bool] function.
func (k Key) Bool(v bool) KeyValue {
	return KeyValue{
		Key:   k,
		Value: BoolValue(v),
	}
}

// BoolSlice returns a [KeyValue] for a []bool value.
//
// Note that many observability backends are not optimized to query, index, or
// aggregate complex attribute values. Complex values may also carry
// additional performance overhead. Prefer primitive values when
// possible.
//
// If creating both a key and value at the same time, use the package-level
// [BoolSlice] function.
func (k Key) BoolSlice(v []bool) KeyValue {
	return KeyValue{
		Key:   k,
		Value: BoolSliceValue(v),
	}
}

// Int returns a [KeyValue] for an int value.
//
// It is provided as a convenience for [Key.Int64].
//
// If creating both a key and value at the same time, use the package-level [Int]
// function.
func (k Key) Int(v int) KeyValue {
	return KeyValue{
		Key:   k,
		Value: IntValue(v),
	}
}

// IntSlice returns a [KeyValue] for a []int value.
//
// It is provided as a convenience for [Key.Int64Slice].
//
// Note that many observability backends are not optimized to query, index, or
// aggregate complex attribute values. Complex values may also carry
// additional performance overhead. Prefer primitive values when
// possible.
//
// If creating both a key and value at the same time, use the package-level
// [IntSlice] function.
func (k Key) IntSlice(v []int) KeyValue {
	return KeyValue{
		Key:   k,
		Value: IntSliceValue(v),
	}
}

// Int64 returns a [KeyValue] for an int64 value.
//
// If creating both a key and value at the same time, use the package-level
// [Int64] function.
func (k Key) Int64(v int64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Int64Value(v),
	}
}

// Int64Slice returns a [KeyValue] for a []int64 value.
//
// Note that many observability backends are not optimized to query, index, or
// aggregate complex attribute values. Complex values may also carry
// additional performance overhead. Prefer primitive values when
// possible.
//
// If creating both a key and value at the same time, use the package-level
// [Int64Slice] function.
func (k Key) Int64Slice(v []int64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Int64SliceValue(v),
	}
}

// Float64 returns a [KeyValue] for a float64 value.
//
// If creating both a key and value at the same time, use the package-level
// [Float64] function.
func (k Key) Float64(v float64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Float64Value(v),
	}
}

// Float64Slice returns a [KeyValue] for a []float64 value.
//
// Note that many observability backends are not optimized to query, index, or
// aggregate complex attribute values. Complex values may also carry
// additional performance overhead. Prefer primitive values when
// possible.
//
// If creating both a key and value at the same time, use the package-level
// [Float64Slice] function.
func (k Key) Float64Slice(v []float64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Float64SliceValue(v),
	}
}

// String returns a [KeyValue] for a string value.
//
// If creating both a key and value at the same time, use the package-level
// [String] function.
func (k Key) String(v string) KeyValue {
	return KeyValue{
		Key:   k,
		Value: StringValue(v),
	}
}

// StringSlice returns a [KeyValue] for a []string value.
//
// Note that many observability backends are not optimized to query, index, or
// aggregate complex attribute values. Complex values may also carry
// additional performance overhead. Prefer primitive values when
// possible.
//
// If creating both a key and value at the same time, use the package-level
// [StringSlice] function.
func (k Key) StringSlice(v []string) KeyValue {
	return KeyValue{
		Key:   k,
		Value: StringSliceValue(v),
	}
}

// ByteSlice returns a [KeyValue] for a []byte value.
//
// Note that many observability backends are not optimized to query, index, or
// aggregate complex attribute values. Complex values may also carry
// additional performance overhead. Prefer primitive values when
// possible.
//
// If creating both a key and value at the same time, use the package-level
// [ByteSlice] function.
func (k Key) ByteSlice(v []byte) KeyValue {
	return KeyValue{
		Key:   k,
		Value: ByteSliceValue(v),
	}
}

// Slice returns a [KeyValue] for a []Value value.
//
// Note that many observability backends are not optimized to query, index, or
// aggregate complex attribute values. Complex values may also carry
// additional performance overhead. Prefer primitive values when
// possible.
//
// If creating both a key and value at the same time, use the package-level
// [Slice] function.
func (k Key) Slice(v ...Value) KeyValue {
	return KeyValue{
		Key:   k,
		Value: SliceValue(v...),
	}
}

// Map returns a [KeyValue] for a []KeyValue value.
//
// Note that many observability backends are not optimized to query, index, or
// aggregate complex attribute values. Complex values may also carry
// additional performance overhead. Prefer primitive values when
// possible.
//
// Users should avoid providing duplicate keys; many receivers handle maps
// containing duplicate keys unpredictably.
//
// The order of v is not preserved.
//
// If creating both a key and value at the same time, use the package-level [Map]
// function.
func (k Key) Map(v ...KeyValue) KeyValue {
	return KeyValue{
		Key:   k,
		Value: MapValue(v...),
	}
}

// Defined reports whether the key is not empty.
func (k Key) Defined() bool {
	return len(k) != 0
}
