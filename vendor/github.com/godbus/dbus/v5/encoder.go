package dbus

import (
	"bytes"
	"encoding/binary"
	"io"
	"reflect"
	"strings"
	"unicode/utf8"
)

// An encoder encodes values to the D-Bus wire format.
type encoder struct {
	out   io.Writer
	fds   []int
	order binary.ByteOrder
	pos   int
}

// NewEncoder returns a new encoder that writes to out in the given byte order.
func newEncoder(out io.Writer, order binary.ByteOrder, fds []int) *encoder {
	enc := newEncoderAtOffset(out, 0, order, fds)
	return enc
}

// newEncoderAtOffset returns a new encoder that writes to out in the given
// byte order. Specify the offset to initialize pos for proper alignment
// computation.
func newEncoderAtOffset(out io.Writer, offset int, order binary.ByteOrder, fds []int) *encoder {
	enc := new(encoder)
	enc.out = out
	enc.order = order
	enc.pos = offset
	enc.fds = fds
	return enc
}

// Aligns the next output to be on a multiple of n. Panics on write errors.
func (enc *encoder) align(n int) {
	pad := enc.padding(0, n)
	if pad > 0 {
		empty := make([]byte, pad)
		if _, err := enc.out.Write(empty); err != nil {
			panic(err)
		}
		enc.pos += pad
	}
}

// pad returns the number of bytes of padding, based on current position and additional offset.
// and alignment.
func (enc *encoder) padding(offset, algn int) int {
	abs := enc.pos + offset
	if abs%algn != 0 {
		newabs := (abs + algn - 1) & ^(algn - 1)
		return newabs - abs
	}
	return 0
}

// Calls binary.Write(enc.out, enc.order, v) and panics on write errors.
func (enc *encoder) binwrite(v any) {
	if err := binary.Write(enc.out, enc.order, v); err != nil {
		panic(err)
	}
}

// Encode encodes the given values to the underlying reader. All written values
// are aligned properly as required by the D-Bus spec.
func (enc *encoder) Encode(vs ...any) (err error) {
	defer func() {
		err, _ = recover().(error)
	}()
	for _, v := range vs {
		enc.encode(reflect.ValueOf(v), 0)
	}
	return nil
}

// encode encodes the given value to the writer and panics on error. depth holds
// the depth of the container nesting.
func (enc *encoder) encode(v reflect.Value, depth int) {
	if depth > 64 {
		panic(FormatError("input exceeds depth limitation"))
	}
	enc.align(alignment(v.Type()))
	switch v.Kind() {
	case reflect.Uint8:
		var b [1]byte
		b[0] = byte(v.Uint())
		if _, err := enc.out.Write(b[:]); err != nil {
			panic(err)
		}
		enc.pos++
	case reflect.Bool:
		if v.Bool() {
			enc.encode(reflect.ValueOf(uint32(1)), depth)
		} else {
			enc.encode(reflect.ValueOf(uint32(0)), depth)
		}
	case reflect.Int16:
		enc.binwrite(int16(v.Int()))
		enc.pos += 2
	case reflect.Uint16:
		enc.binwrite(uint16(v.Uint()))
		enc.pos += 2
	case reflect.Int, reflect.Int32:
		if v.Type() == unixFDType {
			fd := v.Int()
			idx := len(enc.fds)
			enc.fds = append(enc.fds, int(fd))
			enc.binwrite(uint32(idx))
		} else {
			enc.binwrite(int32(v.Int()))
		}
		enc.pos += 4
	case reflect.Uint, reflect.Uint32:
		enc.binwrite(uint32(v.Uint()))
		enc.pos += 4
	case reflect.Int64:
		enc.binwrite(v.Int())
		enc.pos += 8
	case reflect.Uint64:
		enc.binwrite(v.Uint())
		enc.pos += 8
	case reflect.Float64:
		enc.binwrite(v.Float())
		enc.pos += 8
	case reflect.String:
		str := v.String()
		if !utf8.ValidString(str) {
			panic(FormatError("input has a not-utf8 char in string"))
		}
		if strings.IndexByte(str, byte(0)) != -1 {
			panic(FormatError("input has a null char('\\000') in string"))
		}
		if v.Type() == objectPathType {
			if !ObjectPath(str).IsValid() {
				panic(FormatError("invalid object path"))
			}
		}
		enc.encode(reflect.ValueOf(uint32(len(str))), depth)
		b := make([]byte, v.Len()+1)
		copy(b, str)
		b[len(b)-1] = 0
		n, err := enc.out.Write(b)
		if err != nil {
			panic(err)
		}
		enc.pos += n
	case reflect.Ptr:
		enc.encode(v.Elem(), depth)
	case reflect.Slice, reflect.Array:
		// Lookahead offset: 4 bytes for uint32 length (with alignment),
		// plus alignment for elements.
		n := enc.padding(0, 4) + 4
		offset := enc.pos + n + enc.padding(n, alignment(v.Type().Elem()))

		var buf bytes.Buffer
		bufenc := newEncoderAtOffset(&buf, offset, enc.order, enc.fds)

		for i := 0; i < v.Len(); i++ {
			bufenc.encode(v.Index(i), depth+1)
		}

		if buf.Len() > 1<<26 {
			panic(FormatError("input exceeds array size limitation"))
		}

		enc.fds = bufenc.fds
		enc.encode(reflect.ValueOf(uint32(buf.Len())), depth)
		length := buf.Len()
		enc.align(alignment(v.Type().Elem()))
		if _, err := buf.WriteTo(enc.out); err != nil {
			panic(err)
		}
		enc.pos += length
	case reflect.Struct:
		switch t := v.Type(); t {
		case signatureType:
			str := v.Field(0)
			enc.encode(reflect.ValueOf(byte(str.Len())), depth)
			b := make([]byte, str.Len()+1)
			copy(b, str.String())
			b[len(b)-1] = 0
			n, err := enc.out.Write(b)
			if err != nil {
				panic(err)
			}
			enc.pos += n
		case variantType:
			variant := v.Interface().(Variant)
			enc.encode(reflect.ValueOf(variant.sig), depth+1)
			enc.encode(reflect.ValueOf(variant.value), depth+1)
		default:
			for i := 0; i < v.Type().NumField(); i++ {
				field := t.Field(i)
				if field.PkgPath == "" && field.Tag.Get("dbus") != "-" {
					enc.encode(v.Field(i), depth+1)
				}
			}
		}
	case reflect.Map:
		// Maps are arrays of structures, so they actually increase the depth by
		// 2.
		if !isKeyType(v.Type().Key()) {
			panic(InvalidTypeError{v.Type()})
		}
		keys := v.MapKeys()
		// Lookahead offset: 4 bytes for uint32 length (with alignment),
		// plus 8-byte alignment
		n := enc.padding(0, 4) + 4
		offset := enc.pos + n + enc.padding(n, 8)

		var buf bytes.Buffer
		bufenc := newEncoderAtOffset(&buf, offset, enc.order, enc.fds)
		for _, k := range keys {
			bufenc.align(8)
			bufenc.encode(k, depth+2)
			bufenc.encode(v.MapIndex(k), depth+2)
		}
		enc.fds = bufenc.fds
		enc.encode(reflect.ValueOf(uint32(buf.Len())), depth)
		length := buf.Len()
		enc.align(8)
		if _, err := buf.WriteTo(enc.out); err != nil {
			panic(err)
		}
		enc.pos += length
	case reflect.Interface:
		enc.encode(reflect.ValueOf(MakeVariant(v.Interface())), depth)
	default:
		panic(InvalidTypeError{v.Type()})
	}
}
