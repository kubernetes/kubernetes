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
// If creating both key and a bool value at the same time, then
// instead of calling Key(name).Bool(value) consider using a
// convenience function provided by the api/key package -
// key.Bool(name, value).
func (k Key) Bool(v bool) KeyValue {
	return KeyValue{
		Key:   k,
		Value: BoolValue(v),
	}
}

// Int64 creates a KeyValue instance with an INT64 Value.
//
// If creating both key and an int64 value at the same time, then
// instead of calling Key(name).Int64(value) consider using a
// convenience function provided by the api/key package -
// key.Int64(name, value).
func (k Key) Int64(v int64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Int64Value(v),
	}
}

// Float64 creates a KeyValue instance with a FLOAT64 Value.
//
// If creating both key and a float64 value at the same time, then
// instead of calling Key(name).Float64(value) consider using a
// convenience function provided by the api/key package -
// key.Float64(name, value).
func (k Key) Float64(v float64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Float64Value(v),
	}
}

// String creates a KeyValue instance with a STRING Value.
//
// If creating both key and a string value at the same time, then
// instead of calling Key(name).String(value) consider using a
// convenience function provided by the api/key package -
// key.String(name, value).
func (k Key) String(v string) KeyValue {
	return KeyValue{
		Key:   k,
		Value: StringValue(v),
	}
}

// Int creates a KeyValue instance with an INT64 Value.
//
// If creating both key and an int value at the same time, then
// instead of calling Key(name).Int(value) consider using a
// convenience function provided by the api/key package -
// key.Int(name, value).
func (k Key) Int(v int) KeyValue {
	return KeyValue{
		Key:   k,
		Value: IntValue(v),
	}
}

// Defined returns true for non-empty keys.
func (k Key) Defined() bool {
	return len(k) != 0
}

// Array creates a KeyValue instance with a ARRAY Value.
//
// If creating both key and a array value at the same time, then
// instead of calling Key(name).String(value) consider using a
// convenience function provided by the api/key package -
// key.Array(name, value).
func (k Key) Array(v interface{}) KeyValue {
	return KeyValue{
		Key:   k,
		Value: ArrayValue(v),
	}
}
