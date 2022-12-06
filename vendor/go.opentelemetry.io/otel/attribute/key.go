// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package attribute // import "go.opentelemetry.io/otel/attribute"

// Key represents the key part in key-value pairs. It's a string. The
// allowed character set in the key depends on the use of the key.
type Key string

// Bool creates a KeyValue instance with a BOOL Value.
//
// If creating both a key and value at the same time, use the provided
// convenience function instead -- Bool(name, value).
func (k Key) Bool(v bool) KeyValue {
	return KeyValue{
		Key:   k,
		Value: BoolValue(v),
	}
}

// BoolSlice creates a KeyValue instance with a BOOLSLICE Value.
//
// If creating both a key and value at the same time, use the provided
// convenience function instead -- BoolSlice(name, value).
func (k Key) BoolSlice(v []bool) KeyValue {
	return KeyValue{
		Key:   k,
		Value: BoolSliceValue(v),
	}
}

// Int creates a KeyValue instance with an INT64 Value.
//
// If creating both a key and value at the same time, use the provided
// convenience function instead -- Int(name, value).
func (k Key) Int(v int) KeyValue {
	return KeyValue{
		Key:   k,
		Value: IntValue(v),
	}
}

// IntSlice creates a KeyValue instance with an INT64SLICE Value.
//
// If creating both a key and value at the same time, use the provided
// convenience function instead -- IntSlice(name, value).
func (k Key) IntSlice(v []int) KeyValue {
	return KeyValue{
		Key:   k,
		Value: IntSliceValue(v),
	}
}

// Int64 creates a KeyValue instance with an INT64 Value.
//
// If creating both a key and value at the same time, use the provided
// convenience function instead -- Int64(name, value).
func (k Key) Int64(v int64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Int64Value(v),
	}
}

// Int64Slice creates a KeyValue instance with an INT64SLICE Value.
//
// If creating both a key and value at the same time, use the provided
// convenience function instead -- Int64Slice(name, value).
func (k Key) Int64Slice(v []int64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Int64SliceValue(v),
	}
}

// Float64 creates a KeyValue instance with a FLOAT64 Value.
//
// If creating both a key and value at the same time, use the provided
// convenience function instead -- Float64(name, value).
func (k Key) Float64(v float64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Float64Value(v),
	}
}

// Float64Slice creates a KeyValue instance with a FLOAT64SLICE Value.
//
// If creating both a key and value at the same time, use the provided
// convenience function instead -- Float64(name, value).
func (k Key) Float64Slice(v []float64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Float64SliceValue(v),
	}
}

// String creates a KeyValue instance with a STRING Value.
//
// If creating both a key and value at the same time, use the provided
// convenience function instead -- String(name, value).
func (k Key) String(v string) KeyValue {
	return KeyValue{
		Key:   k,
		Value: StringValue(v),
	}
}

// StringSlice creates a KeyValue instance with a STRINGSLICE Value.
//
// If creating both a key and value at the same time, use the provided
// convenience function instead -- StringSlice(name, value).
func (k Key) StringSlice(v []string) KeyValue {
	return KeyValue{
		Key:   k,
		Value: StringSliceValue(v),
	}
}

// Defined returns true for non-empty keys.
func (k Key) Defined() bool {
	return len(k) != 0
}
