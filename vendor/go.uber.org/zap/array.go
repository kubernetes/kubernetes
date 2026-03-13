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
	"time"

	"go.uber.org/zap/zapcore"
)

// Array constructs a field with the given key and ArrayMarshaler. It provides
// a flexible, but still type-safe and efficient, way to add array-like types
// to the logging context. The struct's MarshalLogArray method is called lazily.
func Array(key string, val zapcore.ArrayMarshaler) Field {
	return Field{Key: key, Type: zapcore.ArrayMarshalerType, Interface: val}
}

// Bools constructs a field that carries a slice of bools.
func Bools(key string, bs []bool) Field {
	return Array(key, bools(bs))
}

// ByteStrings constructs a field that carries a slice of []byte, each of which
// must be UTF-8 encoded text.
func ByteStrings(key string, bss [][]byte) Field {
	return Array(key, byteStringsArray(bss))
}

// Complex128s constructs a field that carries a slice of complex numbers.
func Complex128s(key string, nums []complex128) Field {
	return Array(key, complex128s(nums))
}

// Complex64s constructs a field that carries a slice of complex numbers.
func Complex64s(key string, nums []complex64) Field {
	return Array(key, complex64s(nums))
}

// Durations constructs a field that carries a slice of time.Durations.
func Durations(key string, ds []time.Duration) Field {
	return Array(key, durations(ds))
}

// Float64s constructs a field that carries a slice of floats.
func Float64s(key string, nums []float64) Field {
	return Array(key, float64s(nums))
}

// Float32s constructs a field that carries a slice of floats.
func Float32s(key string, nums []float32) Field {
	return Array(key, float32s(nums))
}

// Ints constructs a field that carries a slice of integers.
func Ints(key string, nums []int) Field {
	return Array(key, ints(nums))
}

// Int64s constructs a field that carries a slice of integers.
func Int64s(key string, nums []int64) Field {
	return Array(key, int64s(nums))
}

// Int32s constructs a field that carries a slice of integers.
func Int32s(key string, nums []int32) Field {
	return Array(key, int32s(nums))
}

// Int16s constructs a field that carries a slice of integers.
func Int16s(key string, nums []int16) Field {
	return Array(key, int16s(nums))
}

// Int8s constructs a field that carries a slice of integers.
func Int8s(key string, nums []int8) Field {
	return Array(key, int8s(nums))
}

// Objects constructs a field with the given key, holding a list of the
// provided objects that can be marshaled by Zap.
//
// Note that these objects must implement zapcore.ObjectMarshaler directly.
// That is, if you're trying to marshal a []Request, the MarshalLogObject
// method must be declared on the Request type, not its pointer (*Request).
// If it's on the pointer, use ObjectValues.
//
// Given an object that implements MarshalLogObject on the value receiver, you
// can log a slice of those objects with Objects like so:
//
//	type Author struct{ ... }
//	func (a Author) MarshalLogObject(enc zapcore.ObjectEncoder) error
//
//	var authors []Author = ...
//	logger.Info("loading article", zap.Objects("authors", authors))
//
// Similarly, given a type that implements MarshalLogObject on its pointer
// receiver, you can log a slice of pointers to that object with Objects like
// so:
//
//	type Request struct{ ... }
//	func (r *Request) MarshalLogObject(enc zapcore.ObjectEncoder) error
//
//	var requests []*Request = ...
//	logger.Info("sending requests", zap.Objects("requests", requests))
//
// If instead, you have a slice of values of such an object, use the
// ObjectValues constructor.
//
//	var requests []Request = ...
//	logger.Info("sending requests", zap.ObjectValues("requests", requests))
func Objects[T zapcore.ObjectMarshaler](key string, values []T) Field {
	return Array(key, objects[T](values))
}

type objects[T zapcore.ObjectMarshaler] []T

func (os objects[T]) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for _, o := range os {
		if err := arr.AppendObject(o); err != nil {
			return err
		}
	}
	return nil
}

// ObjectMarshalerPtr is a constraint that specifies that the given type
// implements zapcore.ObjectMarshaler on a pointer receiver.
type ObjectMarshalerPtr[T any] interface {
	*T
	zapcore.ObjectMarshaler
}

// ObjectValues constructs a field with the given key, holding a list of the
// provided objects, where pointers to these objects can be marshaled by Zap.
//
// Note that pointers to these objects must implement zapcore.ObjectMarshaler.
// That is, if you're trying to marshal a []Request, the MarshalLogObject
// method must be declared on the *Request type, not the value (Request).
// If it's on the value, use Objects.
//
// Given an object that implements MarshalLogObject on the pointer receiver,
// you can log a slice of those objects with ObjectValues like so:
//
//	type Request struct{ ... }
//	func (r *Request) MarshalLogObject(enc zapcore.ObjectEncoder) error
//
//	var requests []Request = ...
//	logger.Info("sending requests", zap.ObjectValues("requests", requests))
//
// If instead, you have a slice of pointers of such an object, use the Objects
// field constructor.
//
//	var requests []*Request = ...
//	logger.Info("sending requests", zap.Objects("requests", requests))
func ObjectValues[T any, P ObjectMarshalerPtr[T]](key string, values []T) Field {
	return Array(key, objectValues[T, P](values))
}

type objectValues[T any, P ObjectMarshalerPtr[T]] []T

func (os objectValues[T, P]) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range os {
		// It is necessary for us to explicitly reference the "P" type.
		// We cannot simply pass "&os[i]" to AppendObject because its type
		// is "*T", which the type system does not consider as
		// implementing ObjectMarshaler.
		// Only the type "P" satisfies ObjectMarshaler, which we have
		// to convert "*T" to explicitly.
		var p P = &os[i]
		if err := arr.AppendObject(p); err != nil {
			return err
		}
	}
	return nil
}

// Strings constructs a field that carries a slice of strings.
func Strings(key string, ss []string) Field {
	return Array(key, stringArray(ss))
}

// Stringers constructs a field with the given key, holding a list of the
// output provided by the value's String method
//
// Given an object that implements String on the value receiver, you
// can log a slice of those objects with Objects like so:
//
//	type Request struct{ ... }
//	func (a Request) String() string
//
//	var requests []Request = ...
//	logger.Info("sending requests", zap.Stringers("requests", requests))
//
// Note that these objects must implement fmt.Stringer directly.
// That is, if you're trying to marshal a []Request, the String method
// must be declared on the Request type, not its pointer (*Request).
func Stringers[T fmt.Stringer](key string, values []T) Field {
	return Array(key, stringers[T](values))
}

type stringers[T fmt.Stringer] []T

func (os stringers[T]) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for _, o := range os {
		arr.AppendString(o.String())
	}
	return nil
}

// Times constructs a field that carries a slice of time.Times.
func Times(key string, ts []time.Time) Field {
	return Array(key, times(ts))
}

// Uints constructs a field that carries a slice of unsigned integers.
func Uints(key string, nums []uint) Field {
	return Array(key, uints(nums))
}

// Uint64s constructs a field that carries a slice of unsigned integers.
func Uint64s(key string, nums []uint64) Field {
	return Array(key, uint64s(nums))
}

// Uint32s constructs a field that carries a slice of unsigned integers.
func Uint32s(key string, nums []uint32) Field {
	return Array(key, uint32s(nums))
}

// Uint16s constructs a field that carries a slice of unsigned integers.
func Uint16s(key string, nums []uint16) Field {
	return Array(key, uint16s(nums))
}

// Uint8s constructs a field that carries a slice of unsigned integers.
func Uint8s(key string, nums []uint8) Field {
	return Array(key, uint8s(nums))
}

// Uintptrs constructs a field that carries a slice of pointer addresses.
func Uintptrs(key string, us []uintptr) Field {
	return Array(key, uintptrs(us))
}

// Errors constructs a field that carries a slice of errors.
func Errors(key string, errs []error) Field {
	return Array(key, errArray(errs))
}

type bools []bool

func (bs bools) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range bs {
		arr.AppendBool(bs[i])
	}
	return nil
}

type byteStringsArray [][]byte

func (bss byteStringsArray) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range bss {
		arr.AppendByteString(bss[i])
	}
	return nil
}

type complex128s []complex128

func (nums complex128s) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendComplex128(nums[i])
	}
	return nil
}

type complex64s []complex64

func (nums complex64s) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendComplex64(nums[i])
	}
	return nil
}

type durations []time.Duration

func (ds durations) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range ds {
		arr.AppendDuration(ds[i])
	}
	return nil
}

type float64s []float64

func (nums float64s) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendFloat64(nums[i])
	}
	return nil
}

type float32s []float32

func (nums float32s) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendFloat32(nums[i])
	}
	return nil
}

type ints []int

func (nums ints) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendInt(nums[i])
	}
	return nil
}

type int64s []int64

func (nums int64s) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendInt64(nums[i])
	}
	return nil
}

type int32s []int32

func (nums int32s) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendInt32(nums[i])
	}
	return nil
}

type int16s []int16

func (nums int16s) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendInt16(nums[i])
	}
	return nil
}

type int8s []int8

func (nums int8s) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendInt8(nums[i])
	}
	return nil
}

type stringArray []string

func (ss stringArray) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range ss {
		arr.AppendString(ss[i])
	}
	return nil
}

type times []time.Time

func (ts times) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range ts {
		arr.AppendTime(ts[i])
	}
	return nil
}

type uints []uint

func (nums uints) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendUint(nums[i])
	}
	return nil
}

type uint64s []uint64

func (nums uint64s) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendUint64(nums[i])
	}
	return nil
}

type uint32s []uint32

func (nums uint32s) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendUint32(nums[i])
	}
	return nil
}

type uint16s []uint16

func (nums uint16s) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendUint16(nums[i])
	}
	return nil
}

type uint8s []uint8

func (nums uint8s) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendUint8(nums[i])
	}
	return nil
}

type uintptrs []uintptr

func (nums uintptrs) MarshalLogArray(arr zapcore.ArrayEncoder) error {
	for i := range nums {
		arr.AppendUintptr(nums[i])
	}
	return nil
}
