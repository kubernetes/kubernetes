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
	"encoding/base64"
	"encoding/json"
	"math"
	"sync"
	"time"
	"unicode/utf8"

	"go.uber.org/zap/buffer"
	"go.uber.org/zap/internal/bufferpool"
)

// For JSON-escaping; see jsonEncoder.safeAddString below.
const _hex = "0123456789abcdef"

var _jsonPool = sync.Pool{New: func() interface{} {
	return &jsonEncoder{}
}}

func getJSONEncoder() *jsonEncoder {
	return _jsonPool.Get().(*jsonEncoder)
}

func putJSONEncoder(enc *jsonEncoder) {
	if enc.reflectBuf != nil {
		enc.reflectBuf.Free()
	}
	enc.EncoderConfig = nil
	enc.buf = nil
	enc.spaced = false
	enc.openNamespaces = 0
	enc.reflectBuf = nil
	enc.reflectEnc = nil
	_jsonPool.Put(enc)
}

type jsonEncoder struct {
	*EncoderConfig
	buf            *buffer.Buffer
	spaced         bool // include spaces after colons and commas
	openNamespaces int

	// for encoding generic values by reflection
	reflectBuf *buffer.Buffer
	reflectEnc *json.Encoder
}

// NewJSONEncoder creates a fast, low-allocation JSON encoder. The encoder
// appropriately escapes all field keys and values.
//
// Note that the encoder doesn't deduplicate keys, so it's possible to produce
// a message like
//   {"foo":"bar","foo":"baz"}
// This is permitted by the JSON specification, but not encouraged. Many
// libraries will ignore duplicate key-value pairs (typically keeping the last
// pair) when unmarshaling, but users should attempt to avoid adding duplicate
// keys.
func NewJSONEncoder(cfg EncoderConfig) Encoder {
	return newJSONEncoder(cfg, false)
}

func newJSONEncoder(cfg EncoderConfig, spaced bool) *jsonEncoder {
	return &jsonEncoder{
		EncoderConfig: &cfg,
		buf:           bufferpool.Get(),
		spaced:        spaced,
	}
}

func (enc *jsonEncoder) AddArray(key string, arr ArrayMarshaler) error {
	enc.addKey(key)
	return enc.AppendArray(arr)
}

func (enc *jsonEncoder) AddObject(key string, obj ObjectMarshaler) error {
	enc.addKey(key)
	return enc.AppendObject(obj)
}

func (enc *jsonEncoder) AddBinary(key string, val []byte) {
	enc.AddString(key, base64.StdEncoding.EncodeToString(val))
}

func (enc *jsonEncoder) AddByteString(key string, val []byte) {
	enc.addKey(key)
	enc.AppendByteString(val)
}

func (enc *jsonEncoder) AddBool(key string, val bool) {
	enc.addKey(key)
	enc.AppendBool(val)
}

func (enc *jsonEncoder) AddComplex128(key string, val complex128) {
	enc.addKey(key)
	enc.AppendComplex128(val)
}

func (enc *jsonEncoder) AddDuration(key string, val time.Duration) {
	enc.addKey(key)
	enc.AppendDuration(val)
}

func (enc *jsonEncoder) AddFloat64(key string, val float64) {
	enc.addKey(key)
	enc.AppendFloat64(val)
}

func (enc *jsonEncoder) AddInt64(key string, val int64) {
	enc.addKey(key)
	enc.AppendInt64(val)
}

func (enc *jsonEncoder) resetReflectBuf() {
	if enc.reflectBuf == nil {
		enc.reflectBuf = bufferpool.Get()
		enc.reflectEnc = json.NewEncoder(enc.reflectBuf)

		// For consistency with our custom JSON encoder.
		enc.reflectEnc.SetEscapeHTML(false)
	} else {
		enc.reflectBuf.Reset()
	}
}

var nullLiteralBytes = []byte("null")

// Only invoke the standard JSON encoder if there is actually something to
// encode; otherwise write JSON null literal directly.
func (enc *jsonEncoder) encodeReflected(obj interface{}) ([]byte, error) {
	if obj == nil {
		return nullLiteralBytes, nil
	}
	enc.resetReflectBuf()
	if err := enc.reflectEnc.Encode(obj); err != nil {
		return nil, err
	}
	enc.reflectBuf.TrimNewline()
	return enc.reflectBuf.Bytes(), nil
}

func (enc *jsonEncoder) AddReflected(key string, obj interface{}) error {
	valueBytes, err := enc.encodeReflected(obj)
	if err != nil {
		return err
	}
	enc.addKey(key)
	_, err = enc.buf.Write(valueBytes)
	return err
}

func (enc *jsonEncoder) OpenNamespace(key string) {
	enc.addKey(key)
	enc.buf.AppendByte('{')
	enc.openNamespaces++
}

func (enc *jsonEncoder) AddString(key, val string) {
	enc.addKey(key)
	enc.AppendString(val)
}

func (enc *jsonEncoder) AddTime(key string, val time.Time) {
	enc.addKey(key)
	enc.AppendTime(val)
}

func (enc *jsonEncoder) AddUint64(key string, val uint64) {
	enc.addKey(key)
	enc.AppendUint64(val)
}

func (enc *jsonEncoder) AppendArray(arr ArrayMarshaler) error {
	enc.addElementSeparator()
	enc.buf.AppendByte('[')
	err := arr.MarshalLogArray(enc)
	enc.buf.AppendByte(']')
	return err
}

func (enc *jsonEncoder) AppendObject(obj ObjectMarshaler) error {
	enc.addElementSeparator()
	enc.buf.AppendByte('{')
	err := obj.MarshalLogObject(enc)
	enc.buf.AppendByte('}')
	return err
}

func (enc *jsonEncoder) AppendBool(val bool) {
	enc.addElementSeparator()
	enc.buf.AppendBool(val)
}

func (enc *jsonEncoder) AppendByteString(val []byte) {
	enc.addElementSeparator()
	enc.buf.AppendByte('"')
	enc.safeAddByteString(val)
	enc.buf.AppendByte('"')
}

func (enc *jsonEncoder) AppendComplex128(val complex128) {
	enc.addElementSeparator()
	// Cast to a platform-independent, fixed-size type.
	r, i := float64(real(val)), float64(imag(val))
	enc.buf.AppendByte('"')
	// Because we're always in a quoted string, we can use strconv without
	// special-casing NaN and +/-Inf.
	enc.buf.AppendFloat(r, 64)
	enc.buf.AppendByte('+')
	enc.buf.AppendFloat(i, 64)
	enc.buf.AppendByte('i')
	enc.buf.AppendByte('"')
}

func (enc *jsonEncoder) AppendDuration(val time.Duration) {
	cur := enc.buf.Len()
	enc.EncodeDuration(val, enc)
	if cur == enc.buf.Len() {
		// User-supplied EncodeDuration is a no-op. Fall back to nanoseconds to keep
		// JSON valid.
		enc.AppendInt64(int64(val))
	}
}

func (enc *jsonEncoder) AppendInt64(val int64) {
	enc.addElementSeparator()
	enc.buf.AppendInt(val)
}

func (enc *jsonEncoder) AppendReflected(val interface{}) error {
	valueBytes, err := enc.encodeReflected(val)
	if err != nil {
		return err
	}
	enc.addElementSeparator()
	_, err = enc.buf.Write(valueBytes)
	return err
}

func (enc *jsonEncoder) AppendString(val string) {
	enc.addElementSeparator()
	enc.buf.AppendByte('"')
	enc.safeAddString(val)
	enc.buf.AppendByte('"')
}

func (enc *jsonEncoder) AppendTime(val time.Time) {
	cur := enc.buf.Len()
	enc.EncodeTime(val, enc)
	if cur == enc.buf.Len() {
		// User-supplied EncodeTime is a no-op. Fall back to nanos since epoch to keep
		// output JSON valid.
		enc.AppendInt64(val.UnixNano())
	}
}

func (enc *jsonEncoder) AppendUint64(val uint64) {
	enc.addElementSeparator()
	enc.buf.AppendUint(val)
}

func (enc *jsonEncoder) AddComplex64(k string, v complex64) { enc.AddComplex128(k, complex128(v)) }
func (enc *jsonEncoder) AddFloat32(k string, v float32)     { enc.AddFloat64(k, float64(v)) }
func (enc *jsonEncoder) AddInt(k string, v int)             { enc.AddInt64(k, int64(v)) }
func (enc *jsonEncoder) AddInt32(k string, v int32)         { enc.AddInt64(k, int64(v)) }
func (enc *jsonEncoder) AddInt16(k string, v int16)         { enc.AddInt64(k, int64(v)) }
func (enc *jsonEncoder) AddInt8(k string, v int8)           { enc.AddInt64(k, int64(v)) }
func (enc *jsonEncoder) AddUint(k string, v uint)           { enc.AddUint64(k, uint64(v)) }
func (enc *jsonEncoder) AddUint32(k string, v uint32)       { enc.AddUint64(k, uint64(v)) }
func (enc *jsonEncoder) AddUint16(k string, v uint16)       { enc.AddUint64(k, uint64(v)) }
func (enc *jsonEncoder) AddUint8(k string, v uint8)         { enc.AddUint64(k, uint64(v)) }
func (enc *jsonEncoder) AddUintptr(k string, v uintptr)     { enc.AddUint64(k, uint64(v)) }
func (enc *jsonEncoder) AppendComplex64(v complex64)        { enc.AppendComplex128(complex128(v)) }
func (enc *jsonEncoder) AppendFloat64(v float64)            { enc.appendFloat(v, 64) }
func (enc *jsonEncoder) AppendFloat32(v float32)            { enc.appendFloat(float64(v), 32) }
func (enc *jsonEncoder) AppendInt(v int)                    { enc.AppendInt64(int64(v)) }
func (enc *jsonEncoder) AppendInt32(v int32)                { enc.AppendInt64(int64(v)) }
func (enc *jsonEncoder) AppendInt16(v int16)                { enc.AppendInt64(int64(v)) }
func (enc *jsonEncoder) AppendInt8(v int8)                  { enc.AppendInt64(int64(v)) }
func (enc *jsonEncoder) AppendUint(v uint)                  { enc.AppendUint64(uint64(v)) }
func (enc *jsonEncoder) AppendUint32(v uint32)              { enc.AppendUint64(uint64(v)) }
func (enc *jsonEncoder) AppendUint16(v uint16)              { enc.AppendUint64(uint64(v)) }
func (enc *jsonEncoder) AppendUint8(v uint8)                { enc.AppendUint64(uint64(v)) }
func (enc *jsonEncoder) AppendUintptr(v uintptr)            { enc.AppendUint64(uint64(v)) }

func (enc *jsonEncoder) Clone() Encoder {
	clone := enc.clone()
	clone.buf.Write(enc.buf.Bytes())
	return clone
}

func (enc *jsonEncoder) clone() *jsonEncoder {
	clone := getJSONEncoder()
	clone.EncoderConfig = enc.EncoderConfig
	clone.spaced = enc.spaced
	clone.openNamespaces = enc.openNamespaces
	clone.buf = bufferpool.Get()
	return clone
}

func (enc *jsonEncoder) EncodeEntry(ent Entry, fields []Field) (*buffer.Buffer, error) {
	final := enc.clone()
	final.buf.AppendByte('{')

	if final.LevelKey != "" {
		final.addKey(final.LevelKey)
		cur := final.buf.Len()
		final.EncodeLevel(ent.Level, final)
		if cur == final.buf.Len() {
			// User-supplied EncodeLevel was a no-op. Fall back to strings to keep
			// output JSON valid.
			final.AppendString(ent.Level.String())
		}
	}
	if final.TimeKey != "" {
		final.AddTime(final.TimeKey, ent.Time)
	}
	if ent.LoggerName != "" && final.NameKey != "" {
		final.addKey(final.NameKey)
		cur := final.buf.Len()
		nameEncoder := final.EncodeName

		// if no name encoder provided, fall back to FullNameEncoder for backwards
		// compatibility
		if nameEncoder == nil {
			nameEncoder = FullNameEncoder
		}

		nameEncoder(ent.LoggerName, final)
		if cur == final.buf.Len() {
			// User-supplied EncodeName was a no-op. Fall back to strings to
			// keep output JSON valid.
			final.AppendString(ent.LoggerName)
		}
	}
	if ent.Caller.Defined && final.CallerKey != "" {
		final.addKey(final.CallerKey)
		cur := final.buf.Len()
		final.EncodeCaller(ent.Caller, final)
		if cur == final.buf.Len() {
			// User-supplied EncodeCaller was a no-op. Fall back to strings to
			// keep output JSON valid.
			final.AppendString(ent.Caller.String())
		}
	}
	if final.MessageKey != "" {
		final.addKey(enc.MessageKey)
		final.AppendString(ent.Message)
	}
	if enc.buf.Len() > 0 {
		final.addElementSeparator()
		final.buf.Write(enc.buf.Bytes())
	}
	addFields(final, fields)
	final.closeOpenNamespaces()
	if ent.Stack != "" && final.StacktraceKey != "" {
		final.AddString(final.StacktraceKey, ent.Stack)
	}
	final.buf.AppendByte('}')
	if final.LineEnding != "" {
		final.buf.AppendString(final.LineEnding)
	} else {
		final.buf.AppendString(DefaultLineEnding)
	}

	ret := final.buf
	putJSONEncoder(final)
	return ret, nil
}

func (enc *jsonEncoder) truncate() {
	enc.buf.Reset()
}

func (enc *jsonEncoder) closeOpenNamespaces() {
	for i := 0; i < enc.openNamespaces; i++ {
		enc.buf.AppendByte('}')
	}
}

func (enc *jsonEncoder) addKey(key string) {
	enc.addElementSeparator()
	enc.buf.AppendByte('"')
	enc.safeAddString(key)
	enc.buf.AppendByte('"')
	enc.buf.AppendByte(':')
	if enc.spaced {
		enc.buf.AppendByte(' ')
	}
}

func (enc *jsonEncoder) addElementSeparator() {
	last := enc.buf.Len() - 1
	if last < 0 {
		return
	}
	switch enc.buf.Bytes()[last] {
	case '{', '[', ':', ',', ' ':
		return
	default:
		enc.buf.AppendByte(',')
		if enc.spaced {
			enc.buf.AppendByte(' ')
		}
	}
}

func (enc *jsonEncoder) appendFloat(val float64, bitSize int) {
	enc.addElementSeparator()
	switch {
	case math.IsNaN(val):
		enc.buf.AppendString(`"NaN"`)
	case math.IsInf(val, 1):
		enc.buf.AppendString(`"+Inf"`)
	case math.IsInf(val, -1):
		enc.buf.AppendString(`"-Inf"`)
	default:
		enc.buf.AppendFloat(val, bitSize)
	}
}

// safeAddString JSON-escapes a string and appends it to the internal buffer.
// Unlike the standard library's encoder, it doesn't attempt to protect the
// user from browser vulnerabilities or JSONP-related problems.
func (enc *jsonEncoder) safeAddString(s string) {
	for i := 0; i < len(s); {
		if enc.tryAddRuneSelf(s[i]) {
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		if enc.tryAddRuneError(r, size) {
			i++
			continue
		}
		enc.buf.AppendString(s[i : i+size])
		i += size
	}
}

// safeAddByteString is no-alloc equivalent of safeAddString(string(s)) for s []byte.
func (enc *jsonEncoder) safeAddByteString(s []byte) {
	for i := 0; i < len(s); {
		if enc.tryAddRuneSelf(s[i]) {
			i++
			continue
		}
		r, size := utf8.DecodeRune(s[i:])
		if enc.tryAddRuneError(r, size) {
			i++
			continue
		}
		enc.buf.Write(s[i : i+size])
		i += size
	}
}

// tryAddRuneSelf appends b if it is valid UTF-8 character represented in a single byte.
func (enc *jsonEncoder) tryAddRuneSelf(b byte) bool {
	if b >= utf8.RuneSelf {
		return false
	}
	if 0x20 <= b && b != '\\' && b != '"' {
		enc.buf.AppendByte(b)
		return true
	}
	switch b {
	case '\\', '"':
		enc.buf.AppendByte('\\')
		enc.buf.AppendByte(b)
	case '\n':
		enc.buf.AppendByte('\\')
		enc.buf.AppendByte('n')
	case '\r':
		enc.buf.AppendByte('\\')
		enc.buf.AppendByte('r')
	case '\t':
		enc.buf.AppendByte('\\')
		enc.buf.AppendByte('t')
	default:
		// Encode bytes < 0x20, except for the escape sequences above.
		enc.buf.AppendString(`\u00`)
		enc.buf.AppendByte(_hex[b>>4])
		enc.buf.AppendByte(_hex[b&0xF])
	}
	return true
}

func (enc *jsonEncoder) tryAddRuneError(r rune, size int) bool {
	if r == utf8.RuneError && size == 1 {
		enc.buf.AppendString(`\ufffd`)
		return true
	}
	return false
}
