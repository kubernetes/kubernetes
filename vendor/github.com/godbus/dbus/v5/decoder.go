package dbus

import (
	"encoding/binary"
	"io"
	"reflect"
	"unsafe"
)

type decoder struct {
	in    io.Reader
	order binary.ByteOrder
	pos   int
	fds   []int

	// The following fields are used to reduce memory allocs.
	conv *stringConverter
	buf  []byte
	d    float64
	y    [1]byte
}

// newDecoder returns a new decoder that reads values from in. The input is
// expected to be in the given byte order.
func newDecoder(in io.Reader, order binary.ByteOrder, fds []int) *decoder {
	dec := new(decoder)
	dec.in = in
	dec.order = order
	dec.fds = fds
	dec.conv = newStringConverter(stringConverterBufferSize)
	return dec
}

// Reset resets the decoder to be reading from in.
func (dec *decoder) Reset(in io.Reader, order binary.ByteOrder, fds []int) {
	dec.in = in
	dec.order = order
	dec.pos = 0
	dec.fds = fds

	if dec.conv == nil {
		dec.conv = newStringConverter(stringConverterBufferSize)
	}
}

// align aligns the input to the given boundary and panics on error.
func (dec *decoder) align(n int) {
	if dec.pos%n != 0 {
		newpos := (dec.pos + n - 1) & ^(n - 1)
		dec.read2buf(newpos - dec.pos)
		dec.pos = newpos
	}
}

// Calls binary.Read(dec.in, dec.order, v) and panics on read errors.
func (dec *decoder) binread(v any) {
	if err := binary.Read(dec.in, dec.order, v); err != nil {
		panic(err)
	}
}

func (dec *decoder) Decode(sig Signature) (vs []any, err error) {
	defer func() {
		v := recover()
		if v == nil {
			return
		}
		e, ok := v.(error)
		if !ok {
			// Not ours; re-panic.
			panic(v)
		}
		if e == io.EOF || e == io.ErrUnexpectedEOF {
			e = FormatError("unexpected EOF")
		}
		err = e
	}()
	vs = make([]any, 0)
	s := sig.str
	for s != "" {
		err, rem := validSingle(s, &depthCounter{})
		if err != nil {
			return nil, err
		}
		v := dec.decode(s[:len(s)-len(rem)], 0)
		vs = append(vs, v)
		s = rem
	}
	return vs, nil
}

// read2buf reads exactly n bytes from the reader dec.in into the buffer dec.buf
// to reduce memory allocs.
// The buffer grows automatically.
func (dec *decoder) read2buf(n int) {
	var err error
	dec.buf, err = readFull(dec.in, dec.buf, n)
	if err != nil {
		panic(err)
	}
}

// maxArrayLen is the maximum array length in bytes, as defined by the D-Bus spec.
const maxArrayLen = 1 << 26

// readChunk is the size of the chunks readFull reads large inputs in.
const readChunk = 64 << 10

// readFull is like io.ReadFull, but it reads exactly n bytes from r into
// buf (reusing its capacity), and returns the resulting slice. If n is large
// and buf has insufficient capacity, buf is grown gradually as data arrives,
// so a bogus length coming from the input can't cause a huge allocation.
func readFull(r io.Reader, buf []byte, n int) ([]byte, error) {
	if n <= cap(buf) || n <= readChunk {
		if n > cap(buf) {
			buf = make([]byte, n)
		}
		buf = buf[:n]
		_, err := io.ReadFull(r, buf)
		return buf, err
	}
	buf = buf[:0]
	for len(buf) < n {
		// Read at most as much as we already have (but at least
		// readChunk), so the buffer grows geometrically.
		m := n - len(buf)
		if m > len(buf) && m > readChunk {
			m = len(buf)
			if m < readChunk {
				m = readChunk
			}
		}
		if cap(buf)-len(buf) < m {
			nb := make([]byte, len(buf), len(buf)+m)
			copy(nb, buf)
			buf = nb
		}
		k, err := io.ReadFull(r, buf[len(buf):len(buf)+m])
		buf = buf[:len(buf)+k]
		if err != nil {
			if err == io.EOF && len(buf) > 0 {
				err = io.ErrUnexpectedEOF
			}
			return buf, err
		}
	}
	return buf, nil
}

// decodeU decodes uint32 obtained from the reader dec.in.
// The goal is to reduce memory allocs.
func (dec *decoder) decodeU() uint32 {
	dec.align(4)
	dec.read2buf(4)
	dec.pos += 4
	return dec.order.Uint32(dec.buf)
}

func (dec *decoder) decode(s string, depth int) any {
	dec.align(alignment(typeFor(s)))
	switch s[0] {
	case 'y':
		if _, err := dec.in.Read(dec.y[:]); err != nil {
			panic(err)
		}
		dec.pos++
		return dec.y[0]
	case 'b':
		switch dec.decodeU() {
		case 0:
			return false
		case 1:
			return true
		default:
			panic(FormatError("invalid value for boolean"))
		}
	case 'n':
		dec.read2buf(2)
		dec.pos += 2
		return int16(dec.order.Uint16(dec.buf))
	case 'i':
		dec.read2buf(4)
		dec.pos += 4
		return int32(dec.order.Uint32(dec.buf))
	case 'x':
		dec.read2buf(8)
		dec.pos += 8
		return int64(dec.order.Uint64(dec.buf))
	case 'q':
		dec.read2buf(2)
		dec.pos += 2
		return dec.order.Uint16(dec.buf)
	case 'u':
		return dec.decodeU()
	case 't':
		dec.read2buf(8)
		dec.pos += 8
		return dec.order.Uint64(dec.buf)
	case 'd':
		dec.binread(&dec.d)
		dec.pos += 8
		return dec.d
	case 's':
		length := dec.decodeU()
		p := int(length) + 1
		dec.read2buf(p)
		dec.pos += p
		return dec.conv.String(dec.buf[:len(dec.buf)-1])
	case 'o':
		return ObjectPath(dec.decode("s", depth).(string))
	case 'g':
		length := dec.decode("y", depth).(byte)
		p := int(length) + 1
		dec.read2buf(p)
		dec.pos += p
		sig, err := ParseSignature(
			dec.conv.String(dec.buf[:len(dec.buf)-1]),
		)
		if err != nil {
			panic(err)
		}
		return sig
	case 'v':
		if depth >= 64 {
			panic(FormatError("input exceeds container depth limit"))
		}
		var variant Variant
		sig := dec.decode("g", depth).(Signature)
		if len(sig.str) == 0 {
			panic(FormatError("variant signature is empty"))
		}
		err, rem := validSingle(sig.str, &depthCounter{})
		if err != nil {
			panic(err)
		}
		if rem != "" {
			panic(FormatError("variant signature has multiple types"))
		}
		variant.sig = sig
		variant.value = dec.decode(sig.str, depth+1)
		return variant
	case 'h':
		idx := dec.decodeU()
		if int(idx) < len(dec.fds) {
			return UnixFD(dec.fds[idx])
		}
		return UnixFDIndex(idx)
	case 'a':
		if len(s) > 1 && s[1] == '{' {
			ksig := s[2:3]
			vsig := s[3 : len(s)-1]
			v := reflect.MakeMap(reflect.MapOf(typeFor(ksig), typeFor(vsig)))
			if depth >= 63 {
				panic(FormatError("input exceeds container depth limit"))
			}
			length := dec.decodeU()
			if length > maxArrayLen {
				panic(FormatError("input exceeds array size limitation"))
			}
			// Even for empty maps, the correct padding must be included
			dec.align(8)
			spos := dec.pos
			for dec.pos < spos+int(length) {
				dec.align(8)
				if !isKeyType(v.Type().Key()) {
					panic(InvalidTypeError{v.Type()})
				}
				kv := dec.decode(ksig, depth+2)
				vv := dec.decode(vsig, depth+2)
				v.SetMapIndex(reflect.ValueOf(kv), reflect.ValueOf(vv))
			}
			return v.Interface()
		}
		if depth >= 64 {
			panic(FormatError("input exceeds container depth limit"))
		}
		sig := s[1:]
		length := dec.decodeU()
		if length > maxArrayLen {
			panic(FormatError("input exceeds array size limitation"))
		}
		// capacity can be determined only for fixed-size element types
		var capacity int
		if s := sigByteSize(sig); s != 0 {
			// Do not trust the length too much, as the input
			// might be shorter; append will grow the slice.
			capacity = int(length)
			if capacity > readChunk {
				capacity = readChunk
			}
			capacity /= s
		}
		v := reflect.MakeSlice(reflect.SliceOf(typeFor(sig)), 0, capacity)
		// Even for empty arrays, the correct padding must be included
		align := alignment(typeFor(s[1:]))
		if len(s) > 1 && s[1] == '(' {
			// Special case for arrays of structs
			// structs decode as a slice of interface{} values
			// but the dbus alignment does not match this
			align = 8
		}
		dec.align(align)
		spos := dec.pos
		for dec.pos < spos+int(length) {
			ev := dec.decode(s[1:], depth+1)
			v = reflect.Append(v, reflect.ValueOf(ev))
		}
		return v.Interface()
	case '(':
		if depth >= 64 {
			panic(FormatError("input exceeds container depth limit"))
		}
		dec.align(8)
		v := make([]any, 0)
		s = s[1 : len(s)-1]
		for s != "" {
			err, rem := validSingle(s, &depthCounter{})
			if err != nil {
				panic(err)
			}
			ev := dec.decode(s[:len(s)-len(rem)], depth+1)
			v = append(v, ev)
			s = rem
		}
		return v
	default:
		panic(SignatureError{Sig: s})
	}
}

// sigByteSize tries to calculates size of the given signature in bytes.
//
// It returns zero when it can't, for example when it contains non-fixed size
// types such as strings, maps and arrays that require reading of the transmitted
// data, for that we would need to implement the unread method for Decoder first.
func sigByteSize(sig string) int {
	var total int
	for offset := 0; offset < len(sig); {
		switch sig[offset] {
		case 'y':
			total += 1
			offset += 1
		case 'n', 'q':
			total += 2
			offset += 1
		case 'b', 'i', 'u', 'h':
			total += 4
			offset += 1
		case 'x', 't', 'd':
			total += 8
			offset += 1
		case '(':
			i := 1
			depth := 1
			for i < len(sig[offset:]) && depth != 0 {
				switch sig[offset+i] {
				case '(':
					depth++
				case ')':
					depth--
				}
				i++
			}
			s := sigByteSize(sig[offset+1 : offset+i-1])
			if s == 0 {
				return 0
			}
			total += s
			offset += i
		default:
			return 0
		}
	}
	return total
}

// A FormatError is an error in the wire format.
type FormatError string

func (e FormatError) Error() string {
	return "dbus: wire format error: " + string(e)
}

// stringConverterBufferSize defines the recommended buffer size of 4KB.
// It showed good results in a benchmark when decoding 35KB message,
// see https://github.com/marselester/systemd#testing.
const stringConverterBufferSize = 4096

func newStringConverter(capacity int) *stringConverter {
	return &stringConverter{
		buf:    make([]byte, 0, capacity),
		offset: 0,
	}
}

// stringConverter converts bytes to strings with less allocs.
// The idea is to accumulate bytes in a buffer with specified capacity
// and create strings with unsafe package using bytes from a buffer.
// For example, 10 "fizz" strings written to a 40-byte buffer
// will result in 1 alloc instead of 10.
//
// Once a buffer is filled, a new one is created with the same capacity.
// Old buffers will be eventually GC-ed
// with no side effects to the returned strings.
type stringConverter struct {
	// buf is a temporary buffer where decoded strings are batched.
	buf []byte
	// offset is a buffer position where the last string was written.
	offset int
}

// String converts bytes to a string.
func (c *stringConverter) String(b []byte) string {
	n := len(b)
	if n == 0 {
		return ""
	}
	// Must allocate because a string doesn't fit into the buffer.
	if n > cap(c.buf) {
		return string(b)
	}

	if len(c.buf)+n > cap(c.buf) {
		c.buf = make([]byte, 0, cap(c.buf))
		c.offset = 0
	}
	c.buf = append(c.buf, b...)

	b = c.buf[c.offset:]
	s := toString(b)
	c.offset += n
	return s
}

// toString converts a byte slice to a string without allocating.
func toString(b []byte) string {
	return unsafe.String(&b[0], len(b))
}
