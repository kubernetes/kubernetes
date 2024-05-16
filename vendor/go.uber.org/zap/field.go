// Copyright (c) 2016 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package zap

import (
	"fmt"
	"math"
	"time"

	"go.uber.org/zap/internal/stacktrace"
	"go.uber.org/zap/zapcore"
)

// Field is an alias for Field. Aliasing this type dramatically
// improves the navigability of this package's API documentation.
type Field = zapcore.Field

var (
	_minTimeInt64 = time.Unix(0, math.MinInt64)
	_maxTimeInt64 = time.Unix(0, math.MaxInt64)
)

// Skip constructs a no-op field, which is often useful when handling invalid
// inputs in other Field constructors.
func Skip() Field {
	return Field{Type: zapcore.SkipType}
}

// nilField returns a field which will marshal explicitly as nil. See motivation
// in https://github.com/uber-go/zap/issues/753 . If we ever make breaking
// changes and add zapcore.NilType and zapcore.ObjectEncoder.AddNil, the
// implementation here should be changed to reflect that.
func nilField(key string) Field { return Reflect(key, nil) }

// Binary constructs a field that carries an opaque binary blob.
//
// Binary data is serialized in an encoding-appropriate format. For example,
// zap's JSON encoder base64-encodes binary blobs. To log UTF-8 encoded text,
// use ByteString.
func Binary(key string, val []byte) Field {
	return Field{Key: key, Type: zapcore.BinaryType, Interface: val}
}

// Bool constructs a field that carries a bool.
func Bool(key string, val bool) Field {
	var ival int64
	if val {
		ival = 1
	}
	return Field{Key: key, Type: zapcore.BoolType, Integer: ival}
}

// Boolp constructs a field that carries a *bool. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Boolp(key string, val *bool) Field {
	if val == nil {
		return nilField(key)
	}
	return Bool(key, *val)
}

// ByteString constructs a field that carries UTF-8 encoded text as a []byte.
// To log opaque binary blobs (which aren't necessarily valid UTF-8), use
// Binary.
func ByteString(key string, val []byte) Field {
	return Field{Key: key, Type: zapcore.ByteStringType, Interface: val}
}

// Complex128 constructs a field that carries a complex number. Unlike most
// numeric fields, this costs an allocation (to convert the complex128 to
// interface{}).
func Complex128(key string, val complex128) Field {
	return Field{Key: key, Type: zapcore.Complex128Type, Interface: val}
}

// Complex128p constructs a field that carries a *complex128. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Complex128p(key string, val *complex128) Field {
	if val == nil {
		return nilField(key)
	}
	return Complex128(key, *val)
}

// Complex64 constructs a field that carries a complex number. Unlike most
// numeric fields, this costs an allocation (to convert the complex64 to
// interface{}).
func Complex64(key string, val complex64) Field {
	return Field{Key: key, Type: zapcore.Complex64Type, Interface: val}
}

// Complex64p constructs a field that carries a *complex64. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Complex64p(key string, val *complex64) Field {
	if val == nil {
		return nilField(key)
	}
	return Complex64(key, *val)
}

// Float64 constructs a field that carries a float64. The way the
// floating-point value is represented is encoder-dependent, so marshaling is
// necessarily lazy.
func Float64(key string, val float64) Field {
	return Field{Key: key, Type: zapcore.Float64Type, Integer: int64(math.Float64bits(val))}
}

// Float64p constructs a field that carries a *float64. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Float64p(key string, val *float64) Field {
	if val == nil {
		return nilField(key)
	}
	return Float64(key, *val)
}

// Float32 constructs a field that carries a float32. The way the
// floating-point value is represented is encoder-dependent, so marshaling is
// necessarily lazy.
func Float32(key string, val float32) Field {
	return Field{Key: key, Type: zapcore.Float32Type, Integer: int64(math.Float32bits(val))}
}

// Float32p constructs a field that carries a *float32. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Float32p(key string, val *float32) Field {
	if val == nil {
		return nilField(key)
	}
	return Float32(key, *val)
}

// Int constructs a field with the given key and value.
func Int(key string, val int) Field {
	return Int64(key, int64(val))
}

// Intp constructs a field that carries a *int. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Intp(key string, val *int) Field {
	if val == nil {
		return nilField(key)
	}
	return Int(key, *val)
}

// Int64 constructs a field with the given key and value.
func Int64(key string, val int64) Field {
	return Field{Key: key, Type: zapcore.Int64Type, Integer: val}
}

// Int64p constructs a field that carries a *int64. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Int64p(key string, val *int64) Field {
	if val == nil {
		return nilField(key)
	}
	return Int64(key, *val)
}

// Int32 constructs a field with the given key and value.
func Int32(key string, val int32) Field {
	return Field{Key: key, Type: zapcore.Int32Type, Integer: int64(val)}
}

// Int32p constructs a field that carries a *int32. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Int32p(key string, val *int32) Field {
	if val == nil {
		return nilField(key)
	}
	return Int32(key, *val)
}

// Int16 constructs a field with the given key and value.
func Int16(key string, val int16) Field {
	return Field{Key: key, Type: zapcore.Int16Type, Integer: int64(val)}
}

// Int16p constructs a field that carries a *int16. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Int16p(key string, val *int16) Field {
	if val == nil {
		return nilField(key)
	}
	return Int16(key, *val)
}

// Int8 constructs a field with the given key and value.
func Int8(key string, val int8) Field {
	return Field{Key: key, Type: zapcore.Int8Type, Integer: int64(val)}
}

// Int8p constructs a field that carries a *int8. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Int8p(key string, val *int8) Field {
	if val == nil {
		return nilField(key)
	}
	return Int8(key, *val)
}

// String constructs a field with the given key and value.
func String(key string, val string) Field {
	return Field{Key: key, Type: zapcore.StringType, String: val}
}

// Stringp constructs a field that carries a *string. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Stringp(key string, val *string) Field {
	if val == nil {
		return nilField(key)
	}
	return String(key, *val)
}

// Uint constructs a field with the given key and value.
func Uint(key string, val uint) Field {
	return Uint64(key, uint64(val))
}

// Uintp constructs a field that carries a *uint. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Uintp(key string, val *uint) Field {
	if val == nil {
		return nilField(key)
	}
	return Uint(key, *val)
}

// Uint64 constructs a field with the given key and value.
func Uint64(key string, val uint64) Field {
	return Field{Key: key, Type: zapcore.Uint64Type, Integer: int64(val)}
}

// Uint64p constructs a field that carries a *uint64. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Uint64p(key string, val *uint64) Field {
	if val == nil {
		return nilField(key)
	}
	return Uint64(key, *val)
}

// Uint32 constructs a field with the given key and value.
func Uint32(key string, val uint32) Field {
	return Field{Key: key, Type: zapcore.Uint32Type, Integer: int64(val)}
}

// Uint32p constructs a field that carries a *uint32. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Uint32p(key string, val *uint32) Field {
	if val == nil {
		return nilField(key)
	}
	return Uint32(key, *val)
}

// Uint16 constructs a field with the given key and value.
func Uint16(key string, val uint16) Field {
	return Field{Key: key, Type: zapcore.Uint16Type, Integer: int64(val)}
}

// Uint16p constructs a field that carries a *uint16. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Uint16p(key string, val *uint16) Field {
	if val == nil {
		return nilField(key)
	}
	return Uint16(key, *val)
}

// Uint8 constructs a field with the given key and value.
func Uint8(key string, val uint8) Field {
	return Field{Key: key, Type: zapcore.Uint8Type, Integer: int64(val)}
}

// Uint8p constructs a field that carries a *uint8. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Uint8p(key string, val *uint8) Field {
	if val == nil {
		return nilField(key)
	}
	return Uint8(key, *val)
}

// Uintptr constructs a field with the given key and value.
func Uintptr(key string, val uintptr) Field {
	return Field{Key: key, Type: zapcore.UintptrType, Integer: int64(val)}
}

// Uintptrp constructs a field that carries a *uintptr. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Uintptrp(key string, val *uintptr) Field {
	if val == nil {
		return nilField(key)
	}
	return Uintptr(key, *val)
}

// Reflect constructs a field with the given key and an arbitrary object. It uses
// an encoding-appropriate, reflection-based function to lazily serialize nearly
// any object into the logging context, but it's relatively slow and
// allocation-heavy. Outside tests, Any is always a better choice.
//
// If encoding fails (e.g., trying to serialize a map[int]string to JSON), Reflect
// includes the error message in the final log output.
func Reflect(key string, val interface{}) Field {
	return Field{Key: key, Type: zapcore.ReflectType, Interface: val}
}

// Namespace creates a named, isolated scope within the logger's context. All
// subsequent fields will be added to the new namespace.
//
// This helps prevent key collisions when injecting loggers into sub-components
// or third-party libraries.
func Namespace(key string) Field {
	return Field{Key: key, Type: zapcore.NamespaceType}
}

// Stringer constructs a field with the given key and the output of the value's
// String method. The Stringer's String method is called lazily.
func Stringer(key string, val fmt.Stringer) Field {
	return Field{Key: key, Type: zapcore.StringerType, Interface: val}
}

// Time constructs a Field with the given key and value. The encoder
// controls how the time is serialized.
func Time(key string, val time.Time) Field {
	if val.Before(_minTimeInt64) || val.After(_maxTimeInt64) {
		return Field{Key: key, Type: zapcore.TimeFullType, Interface: val}
	}
	return Field{Key: key, Type: zapcore.TimeType, Integer: val.UnixNano(), Interface: val.Location()}
}

// Timep constructs a field that carries a *time.Time. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Timep(key string, val *time.Time) Field {
	if val == nil {
		return nilField(key)
	}
	return Time(key, *val)
}

// Stack constructs a field that stores a stacktrace of the current goroutine
// under provided key. Keep in mind that taking a stacktrace is eager and
// expensive (relatively speaking); this function both makes an allocation and
// takes about two microseconds.
func Stack(key string) Field {
	return StackSkip(key, 1) // skip Stack
}

// StackSkip constructs a field similarly to Stack, but also skips the given
// number of frames from the top of the stacktrace.
func StackSkip(key string, skip int) Field {
	// Returning the stacktrace as a string costs an allocation, but saves us
	// from expanding the zapcore.Field union struct to include a byte slice. Since
	// taking a stacktrace is already so expensive (~10us), the extra allocation
	// is okay.
	return String(key, stacktrace.Take(skip+1)) // skip StackSkip
}

// Duration constructs a field with the given key and value. The encoder
// controls how the duration is serialized.
func Duration(key string, val time.Duration) Field {
	return Field{Key: key, Type: zapcore.DurationType, Integer: int64(val)}
}

// Durationp constructs a field that carries a *time.Duration. The returned Field will safely
// and explicitly represent `nil` when appropriate.
func Durationp(key string, val *time.Duration) Field {
	if val == nil {
		return nilField(key)
	}
	return Duration(key, *val)
}

// Object constructs a field with the given key and ObjectMarshaler. It
// provides a flexible, but still type-safe and efficient, way to add map- or
// struct-like user-defined types to the logging context. The struct's
// MarshalLogObject method is called lazily.
func Object(key string, val zapcore.ObjectMarshaler) Field {
	return Field{Key: key, Type: zapcore.ObjectMarshalerType, Interface: val}
}

// Inline constructs a Field that is similar to Object, but it
// will add the elements of the provided ObjectMarshaler to the
// current namespace.
func Inline(val zapcore.ObjectMarshaler) Field {
	return zapcore.Field{
		Type:      zapcore.InlineMarshalerType,
		Interface: val,
	}
}

// Dict constructs a field containing the provided key-value pairs.
// It acts similar to [Object], but with the fields specified as arguments.
func Dict(key string, val ...Field) Field {
	return dictField(key, val)
}

// We need a function with the signature (string, T) for zap.Any.
func dictField(key string, val []Field) Field {
	return Object(key, dictObject(val))
}

type dictObject []Field

func (d dictObject) MarshalLogObject(enc zapcore.ObjectEncoder) error {
	for _, f := range d {
		f.AddTo(enc)
	}
	return nil
}

// We discovered an issue where zap.Any can cause a performance degradation
// when used in new goroutines.
//
// This happens because the compiler assigns 4.8kb (one zap.Field per arm of
// switch statement) of stack space for zap.Any when it takes the form:
//
//	switch v := v.(type) {
//	case string:
//		return String(key, v)
//	case int:
//		return Int(key, v)
//		// ...
//	default:
//		return Reflect(key, v)
//	}
//
// To avoid this, we use the type switch to assign a value to a single local variable
// and then call a function on it.
// The local variable is just a function reference so it doesn't allocate
// when converted to an interface{}.
//
// A fair bit of experimentation went into this.
// See also:
//
// - https://github.com/uber-go/zap/pull/1301
// - https://github.com/uber-go/zap/pull/1303
// - https://github.com/uber-go/zap/pull/1304
// - https://github.com/uber-go/zap/pull/1305
// - https://github.com/uber-go/zap/pull/1308
type anyFieldC[T any] func(string, T) Field

func (f anyFieldC[T]) Any(key string, val any) Field {
	v, _ := val.(T)
	// val is guaranteed to be a T, except when it's nil.
	return f(key, v)
}

// Any takes a key and an arbitrary value and chooses the best way to represent
// them as a field, falling back to a reflection-based approach only if
// necessary.
//
// Since byte/uint8 and rune/int32 are aliases, Any can't differentiate between
// them. To minimize surprises, []byte values are treated as binary blobs, byte
// values are treated as uint8, and runes are always treated as integers.
func Any(key string, value interface{}) Field {
	var c interface{ Any(string, any) Field }

	switch value.(type) {
	case zapcore.ObjectMarshaler:
		c = anyFieldC[zapcore.ObjectMarshaler](Object)
	case zapcore.ArrayMarshaler:
		c = anyFieldC[zapcore.ArrayMarshaler](Array)
	case []Field:
		c = anyFieldC[[]Field](dictField)
	case bool:
		c = anyFieldC[bool](Bool)
	case *bool:
		c = anyFieldC[*bool](Boolp)
	case []bool:
		c = anyFieldC[[]bool](Bools)
	case complex128:
		c = anyFieldC[complex128](Complex128)
	case *complex128:
		c = anyFieldC[*complex128](Complex128p)
	case []complex128:
		c = anyFieldC[[]complex128](Complex128s)
	case complex64:
		c = anyFieldC[complex64](Complex64)
	case *complex64:
		c = anyFieldC[*complex64](Complex64p)
	case []complex64:
		c = anyFieldC[[]complex64](Complex64s)
	case float64:
		c = anyFieldC[float64](Float64)
	case *float64:
		c = anyFieldC[*float64](Float64p)
	case []float64:
		c = anyFieldC[[]float64](Float64s)
	case float32:
		c = anyFieldC[float32](Float32)
	case *float32:
		c = anyFieldC[*float32](Float32p)
	case []float32:
		c = anyFieldC[[]float32](Float32s)
	case int:
		c = anyFieldC[int](Int)
	case *int:
		c = anyFieldC[*int](Intp)
	case []int:
		c = anyFieldC[[]int](Ints)
	case int64:
		c = anyFieldC[int64](Int64)
	case *int64:
		c = anyFieldC[*int64](Int64p)
	case []int64:
		c = anyFieldC[[]int64](Int64s)
	case int32:
		c = anyFieldC[int32](Int32)
	case *int32:
		c = anyFieldC[*int32](Int32p)
	case []int32:
		c = anyFieldC[[]int32](Int32s)
	case int16:
		c = anyFieldC[int16](Int16)
	case *int16:
		c = anyFieldC[*int16](Int16p)
	case []int16:
		c = anyFieldC[[]int16](Int16s)
	case int8:
		c = anyFieldC[int8](Int8)
	case *int8:
		c = anyFieldC[*int8](Int8p)
	case []int8:
		c = anyFieldC[[]int8](Int8s)
	case string:
		c = anyFieldC[string](String)
	case *string:
		c = anyFieldC[*string](Stringp)
	case []string:
		c = anyFieldC[[]string](Strings)
	case uint:
		c = anyFieldC[uint](Uint)
	case *uint:
		c = anyFieldC[*uint](Uintp)
	case []uint:
		c = anyFieldC[[]uint](Uints)
	case uint64:
		c = anyFieldC[uint64](Uint64)
	case *uint64:
		c = anyFieldC[*uint64](Uint64p)
	case []uint64:
		c = anyFieldC[[]uint64](Uint64s)
	case uint32:
		c = anyFieldC[uint32](Uint32)
	case *uint32:
		c = anyFieldC[*uint32](Uint32p)
	case []uint32:
		c = anyFieldC[[]uint32](Uint32s)
	case uint16:
		c = anyFieldC[uint16](Uint16)
	case *uint16:
		c = anyFieldC[*uint16](Uint16p)
	case []uint16:
		c = anyFieldC[[]uint16](Uint16s)
	case uint8:
		c = anyFieldC[uint8](Uint8)
	case *uint8:
		c = anyFieldC[*uint8](Uint8p)
	case []byte:
		c = anyFieldC[[]byte](Binary)
	case uintptr:
		c = anyFieldC[uintptr](Uintptr)
	case *uintptr:
		c = anyFieldC[*uintptr](Uintptrp)
	case []uintptr:
		c = anyFieldC[[]uintptr](Uintptrs)
	case time.Time:
		c = anyFieldC[time.Time](Time)
	case *time.Time:
		c = anyFieldC[*time.Time](Timep)
	case []time.Time:
		c = anyFieldC[[]time.Time](Times)
	case time.Duration:
		c = anyFieldC[time.Duration](Duration)
	case *time.Duration:
		c = anyFieldC[*time.Duration](Durationp)
	case []time.Duration:
		c = anyFieldC[[]time.Duration](Durations)
	case error:
		c = anyFieldC[error](NamedError)
	case []error:
		c = anyFieldC[[]error](Errors)
	case fmt.Stringer:
		c = anyFieldC[fmt.Stringer](Stringer)
	default:
		c = anyFieldC[any](Reflect)
	}

	return c.Any(key, value)
}
