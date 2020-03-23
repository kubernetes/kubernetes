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

package zapcore

import (
	"time"

	"go.uber.org/zap/buffer"
)

// DefaultLineEnding defines the default line ending when writing logs.
// Alternate line endings specified in EncoderConfig can override this
// behavior.
const DefaultLineEnding = "\n"

// A LevelEncoder serializes a Level to a primitive type.
type LevelEncoder func(Level, PrimitiveArrayEncoder)

// LowercaseLevelEncoder serializes a Level to a lowercase string. For example,
// InfoLevel is serialized to "info".
func LowercaseLevelEncoder(l Level, enc PrimitiveArrayEncoder) {
	enc.AppendString(l.String())
}

// LowercaseColorLevelEncoder serializes a Level to a lowercase string and adds coloring.
// For example, InfoLevel is serialized to "info" and colored blue.
func LowercaseColorLevelEncoder(l Level, enc PrimitiveArrayEncoder) {
	s, ok := _levelToLowercaseColorString[l]
	if !ok {
		s = _unknownLevelColor.Add(l.String())
	}
	enc.AppendString(s)
}

// CapitalLevelEncoder serializes a Level to an all-caps string. For example,
// InfoLevel is serialized to "INFO".
func CapitalLevelEncoder(l Level, enc PrimitiveArrayEncoder) {
	enc.AppendString(l.CapitalString())
}

// CapitalColorLevelEncoder serializes a Level to an all-caps string and adds color.
// For example, InfoLevel is serialized to "INFO" and colored blue.
func CapitalColorLevelEncoder(l Level, enc PrimitiveArrayEncoder) {
	s, ok := _levelToCapitalColorString[l]
	if !ok {
		s = _unknownLevelColor.Add(l.CapitalString())
	}
	enc.AppendString(s)
}

// UnmarshalText unmarshals text to a LevelEncoder. "capital" is unmarshaled to
// CapitalLevelEncoder, "coloredCapital" is unmarshaled to CapitalColorLevelEncoder,
// "colored" is unmarshaled to LowercaseColorLevelEncoder, and anything else
// is unmarshaled to LowercaseLevelEncoder.
func (e *LevelEncoder) UnmarshalText(text []byte) error {
	switch string(text) {
	case "capital":
		*e = CapitalLevelEncoder
	case "capitalColor":
		*e = CapitalColorLevelEncoder
	case "color":
		*e = LowercaseColorLevelEncoder
	default:
		*e = LowercaseLevelEncoder
	}
	return nil
}

// A TimeEncoder serializes a time.Time to a primitive type.
type TimeEncoder func(time.Time, PrimitiveArrayEncoder)

// EpochTimeEncoder serializes a time.Time to a floating-point number of seconds
// since the Unix epoch.
func EpochTimeEncoder(t time.Time, enc PrimitiveArrayEncoder) {
	nanos := t.UnixNano()
	sec := float64(nanos) / float64(time.Second)
	enc.AppendFloat64(sec)
}

// EpochMillisTimeEncoder serializes a time.Time to a floating-point number of
// milliseconds since the Unix epoch.
func EpochMillisTimeEncoder(t time.Time, enc PrimitiveArrayEncoder) {
	nanos := t.UnixNano()
	millis := float64(nanos) / float64(time.Millisecond)
	enc.AppendFloat64(millis)
}

// EpochNanosTimeEncoder serializes a time.Time to an integer number of
// nanoseconds since the Unix epoch.
func EpochNanosTimeEncoder(t time.Time, enc PrimitiveArrayEncoder) {
	enc.AppendInt64(t.UnixNano())
}

// ISO8601TimeEncoder serializes a time.Time to an ISO8601-formatted string
// with millisecond precision.
func ISO8601TimeEncoder(t time.Time, enc PrimitiveArrayEncoder) {
	enc.AppendString(t.Format("2006-01-02T15:04:05.000Z0700"))
}

// UnmarshalText unmarshals text to a TimeEncoder. "iso8601" and "ISO8601" are
// unmarshaled to ISO8601TimeEncoder, "millis" is unmarshaled to
// EpochMillisTimeEncoder, and anything else is unmarshaled to EpochTimeEncoder.
func (e *TimeEncoder) UnmarshalText(text []byte) error {
	switch string(text) {
	case "iso8601", "ISO8601":
		*e = ISO8601TimeEncoder
	case "millis":
		*e = EpochMillisTimeEncoder
	case "nanos":
		*e = EpochNanosTimeEncoder
	default:
		*e = EpochTimeEncoder
	}
	return nil
}

// A DurationEncoder serializes a time.Duration to a primitive type.
type DurationEncoder func(time.Duration, PrimitiveArrayEncoder)

// SecondsDurationEncoder serializes a time.Duration to a floating-point number of seconds elapsed.
func SecondsDurationEncoder(d time.Duration, enc PrimitiveArrayEncoder) {
	enc.AppendFloat64(float64(d) / float64(time.Second))
}

// NanosDurationEncoder serializes a time.Duration to an integer number of
// nanoseconds elapsed.
func NanosDurationEncoder(d time.Duration, enc PrimitiveArrayEncoder) {
	enc.AppendInt64(int64(d))
}

// StringDurationEncoder serializes a time.Duration using its built-in String
// method.
func StringDurationEncoder(d time.Duration, enc PrimitiveArrayEncoder) {
	enc.AppendString(d.String())
}

// UnmarshalText unmarshals text to a DurationEncoder. "string" is unmarshaled
// to StringDurationEncoder, and anything else is unmarshaled to
// NanosDurationEncoder.
func (e *DurationEncoder) UnmarshalText(text []byte) error {
	switch string(text) {
	case "string":
		*e = StringDurationEncoder
	case "nanos":
		*e = NanosDurationEncoder
	default:
		*e = SecondsDurationEncoder
	}
	return nil
}

// A CallerEncoder serializes an EntryCaller to a primitive type.
type CallerEncoder func(EntryCaller, PrimitiveArrayEncoder)

// FullCallerEncoder serializes a caller in /full/path/to/package/file:line
// format.
func FullCallerEncoder(caller EntryCaller, enc PrimitiveArrayEncoder) {
	// TODO: consider using a byte-oriented API to save an allocation.
	enc.AppendString(caller.String())
}

// ShortCallerEncoder serializes a caller in package/file:line format, trimming
// all but the final directory from the full path.
func ShortCallerEncoder(caller EntryCaller, enc PrimitiveArrayEncoder) {
	// TODO: consider using a byte-oriented API to save an allocation.
	enc.AppendString(caller.TrimmedPath())
}

// UnmarshalText unmarshals text to a CallerEncoder. "full" is unmarshaled to
// FullCallerEncoder and anything else is unmarshaled to ShortCallerEncoder.
func (e *CallerEncoder) UnmarshalText(text []byte) error {
	switch string(text) {
	case "full":
		*e = FullCallerEncoder
	default:
		*e = ShortCallerEncoder
	}
	return nil
}

// A NameEncoder serializes a period-separated logger name to a primitive
// type.
type NameEncoder func(string, PrimitiveArrayEncoder)

// FullNameEncoder serializes the logger name as-is.
func FullNameEncoder(loggerName string, enc PrimitiveArrayEncoder) {
	enc.AppendString(loggerName)
}

// UnmarshalText unmarshals text to a NameEncoder. Currently, everything is
// unmarshaled to FullNameEncoder.
func (e *NameEncoder) UnmarshalText(text []byte) error {
	switch string(text) {
	case "full":
		*e = FullNameEncoder
	default:
		*e = FullNameEncoder
	}
	return nil
}

// An EncoderConfig allows users to configure the concrete encoders supplied by
// zapcore.
type EncoderConfig struct {
	// Set the keys used for each log entry. If any key is empty, that portion
	// of the entry is omitted.
	MessageKey    string `json:"messageKey" yaml:"messageKey"`
	LevelKey      string `json:"levelKey" yaml:"levelKey"`
	TimeKey       string `json:"timeKey" yaml:"timeKey"`
	NameKey       string `json:"nameKey" yaml:"nameKey"`
	CallerKey     string `json:"callerKey" yaml:"callerKey"`
	StacktraceKey string `json:"stacktraceKey" yaml:"stacktraceKey"`
	LineEnding    string `json:"lineEnding" yaml:"lineEnding"`
	// Configure the primitive representations of common complex types. For
	// example, some users may want all time.Times serialized as floating-point
	// seconds since epoch, while others may prefer ISO8601 strings.
	EncodeLevel    LevelEncoder    `json:"levelEncoder" yaml:"levelEncoder"`
	EncodeTime     TimeEncoder     `json:"timeEncoder" yaml:"timeEncoder"`
	EncodeDuration DurationEncoder `json:"durationEncoder" yaml:"durationEncoder"`
	EncodeCaller   CallerEncoder   `json:"callerEncoder" yaml:"callerEncoder"`
	// Unlike the other primitive type encoders, EncodeName is optional. The
	// zero value falls back to FullNameEncoder.
	EncodeName NameEncoder `json:"nameEncoder" yaml:"nameEncoder"`
}

// ObjectEncoder is a strongly-typed, encoding-agnostic interface for adding a
// map- or struct-like object to the logging context. Like maps, ObjectEncoders
// aren't safe for concurrent use (though typical use shouldn't require locks).
type ObjectEncoder interface {
	// Logging-specific marshalers.
	AddArray(key string, marshaler ArrayMarshaler) error
	AddObject(key string, marshaler ObjectMarshaler) error

	// Built-in types.
	AddBinary(key string, value []byte)     // for arbitrary bytes
	AddByteString(key string, value []byte) // for UTF-8 encoded bytes
	AddBool(key string, value bool)
	AddComplex128(key string, value complex128)
	AddComplex64(key string, value complex64)
	AddDuration(key string, value time.Duration)
	AddFloat64(key string, value float64)
	AddFloat32(key string, value float32)
	AddInt(key string, value int)
	AddInt64(key string, value int64)
	AddInt32(key string, value int32)
	AddInt16(key string, value int16)
	AddInt8(key string, value int8)
	AddString(key, value string)
	AddTime(key string, value time.Time)
	AddUint(key string, value uint)
	AddUint64(key string, value uint64)
	AddUint32(key string, value uint32)
	AddUint16(key string, value uint16)
	AddUint8(key string, value uint8)
	AddUintptr(key string, value uintptr)

	// AddReflected uses reflection to serialize arbitrary objects, so it's slow
	// and allocation-heavy.
	AddReflected(key string, value interface{}) error
	// OpenNamespace opens an isolated namespace where all subsequent fields will
	// be added. Applications can use namespaces to prevent key collisions when
	// injecting loggers into sub-components or third-party libraries.
	OpenNamespace(key string)
}

// ArrayEncoder is a strongly-typed, encoding-agnostic interface for adding
// array-like objects to the logging context. Of note, it supports mixed-type
// arrays even though they aren't typical in Go. Like slices, ArrayEncoders
// aren't safe for concurrent use (though typical use shouldn't require locks).
type ArrayEncoder interface {
	// Built-in types.
	PrimitiveArrayEncoder

	// Time-related types.
	AppendDuration(time.Duration)
	AppendTime(time.Time)

	// Logging-specific marshalers.
	AppendArray(ArrayMarshaler) error
	AppendObject(ObjectMarshaler) error

	// AppendReflected uses reflection to serialize arbitrary objects, so it's
	// slow and allocation-heavy.
	AppendReflected(value interface{}) error
}

// PrimitiveArrayEncoder is the subset of the ArrayEncoder interface that deals
// only in Go's built-in types. It's included only so that Duration- and
// TimeEncoders cannot trigger infinite recursion.
type PrimitiveArrayEncoder interface {
	// Built-in types.
	AppendBool(bool)
	AppendByteString([]byte) // for UTF-8 encoded bytes
	AppendComplex128(complex128)
	AppendComplex64(complex64)
	AppendFloat64(float64)
	AppendFloat32(float32)
	AppendInt(int)
	AppendInt64(int64)
	AppendInt32(int32)
	AppendInt16(int16)
	AppendInt8(int8)
	AppendString(string)
	AppendUint(uint)
	AppendUint64(uint64)
	AppendUint32(uint32)
	AppendUint16(uint16)
	AppendUint8(uint8)
	AppendUintptr(uintptr)
}

// Encoder is a format-agnostic interface for all log entry marshalers. Since
// log encoders don't need to support the same wide range of use cases as
// general-purpose marshalers, it's possible to make them faster and
// lower-allocation.
//
// Implementations of the ObjectEncoder interface's methods can, of course,
// freely modify the receiver. However, the Clone and EncodeEntry methods will
// be called concurrently and shouldn't modify the receiver.
type Encoder interface {
	ObjectEncoder

	// Clone copies the encoder, ensuring that adding fields to the copy doesn't
	// affect the original.
	Clone() Encoder

	// EncodeEntry encodes an entry and fields, along with any accumulated
	// context, into a byte buffer and returns it.
	EncodeEntry(Entry, []Field) (*buffer.Buffer, error)
}
