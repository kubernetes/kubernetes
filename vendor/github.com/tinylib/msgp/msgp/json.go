package msgp

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"io"
	"strconv"
	"unicode/utf8"
)

var (
	null = []byte("null")
	hex  = []byte("0123456789abcdef")
)

var defuns [_maxtype]func(jsWriter, *Reader) (int, error)

// note: there is an initialization loop if
// this isn't set up during init()
func init() {
	// since none of these functions are inline-able,
	// there is not much of a penalty to the indirect
	// call. however, this is best expressed as a jump-table...
	defuns = [_maxtype]func(jsWriter, *Reader) (int, error){
		StrType:        rwString,
		BinType:        rwBytes,
		MapType:        rwMap,
		ArrayType:      rwArray,
		Float64Type:    rwFloat64,
		Float32Type:    rwFloat32,
		BoolType:       rwBool,
		IntType:        rwInt,
		UintType:       rwUint,
		NilType:        rwNil,
		ExtensionType:  rwExtension,
		Complex64Type:  rwExtension,
		Complex128Type: rwExtension,
		TimeType:       rwTime,
	}
}

// this is the interface
// used to write json
type jsWriter interface {
	io.Writer
	io.ByteWriter
	WriteString(string) (int, error)
}

// CopyToJSON reads MessagePack from 'src' and copies it
// as JSON to 'dst' until EOF.
func CopyToJSON(dst io.Writer, src io.Reader) (n int64, err error) {
	r := NewReader(src)
	n, err = r.WriteToJSON(dst)
	freeR(r)
	return
}

// WriteToJSON translates MessagePack from 'r' and writes it as
// JSON to 'w' until the underlying reader returns io.EOF. It returns
// the number of bytes written, and an error if it stopped before EOF.
func (r *Reader) WriteToJSON(w io.Writer) (n int64, err error) {
	var j jsWriter
	var bf *bufio.Writer
	if jsw, ok := w.(jsWriter); ok {
		j = jsw
	} else {
		bf = bufio.NewWriter(w)
		j = bf
	}
	var nn int
	for err == nil {
		nn, err = rwNext(j, r)
		n += int64(nn)
	}
	if err != io.EOF {
		if bf != nil {
			bf.Flush()
		}
		return
	}
	err = nil
	if bf != nil {
		err = bf.Flush()
	}
	return
}

func rwNext(w jsWriter, src *Reader) (int, error) {
	t, err := src.NextType()
	if err != nil {
		return 0, err
	}
	return defuns[t](w, src)
}

func rwMap(dst jsWriter, src *Reader) (n int, err error) {
	var comma bool
	var sz uint32
	var field []byte

	sz, err = src.ReadMapHeader()
	if err != nil {
		return
	}

	if sz == 0 {
		return dst.WriteString("{}")
	}

	err = dst.WriteByte('{')
	if err != nil {
		return
	}
	n++
	var nn int
	for i := uint32(0); i < sz; i++ {
		if comma {
			err = dst.WriteByte(',')
			if err != nil {
				return
			}
			n++
		}

		field, err = src.ReadMapKeyPtr()
		if err != nil {
			return
		}
		nn, err = rwquoted(dst, field)
		n += nn
		if err != nil {
			return
		}

		err = dst.WriteByte(':')
		if err != nil {
			return
		}
		n++
		nn, err = rwNext(dst, src)
		n += nn
		if err != nil {
			return
		}
		if !comma {
			comma = true
		}
	}

	err = dst.WriteByte('}')
	if err != nil {
		return
	}
	n++
	return
}

func rwArray(dst jsWriter, src *Reader) (n int, err error) {
	err = dst.WriteByte('[')
	if err != nil {
		return
	}
	var sz uint32
	var nn int
	sz, err = src.ReadArrayHeader()
	if err != nil {
		return
	}
	comma := false
	for i := uint32(0); i < sz; i++ {
		if comma {
			err = dst.WriteByte(',')
			if err != nil {
				return
			}
			n++
		}
		nn, err = rwNext(dst, src)
		n += nn
		if err != nil {
			return
		}
		comma = true
	}

	err = dst.WriteByte(']')
	if err != nil {
		return
	}
	n++
	return
}

func rwNil(dst jsWriter, src *Reader) (int, error) {
	err := src.ReadNil()
	if err != nil {
		return 0, err
	}
	return dst.Write(null)
}

func rwFloat32(dst jsWriter, src *Reader) (int, error) {
	f, err := src.ReadFloat32()
	if err != nil {
		return 0, err
	}
	src.scratch = strconv.AppendFloat(src.scratch[:0], float64(f), 'f', -1, 64)
	return dst.Write(src.scratch)
}

func rwFloat64(dst jsWriter, src *Reader) (int, error) {
	f, err := src.ReadFloat64()
	if err != nil {
		return 0, err
	}
	src.scratch = strconv.AppendFloat(src.scratch[:0], f, 'f', -1, 32)
	return dst.Write(src.scratch)
}

func rwInt(dst jsWriter, src *Reader) (int, error) {
	i, err := src.ReadInt64()
	if err != nil {
		return 0, err
	}
	src.scratch = strconv.AppendInt(src.scratch[:0], i, 10)
	return dst.Write(src.scratch)
}

func rwUint(dst jsWriter, src *Reader) (int, error) {
	u, err := src.ReadUint64()
	if err != nil {
		return 0, err
	}
	src.scratch = strconv.AppendUint(src.scratch[:0], u, 10)
	return dst.Write(src.scratch)
}

func rwBool(dst jsWriter, src *Reader) (int, error) {
	b, err := src.ReadBool()
	if err != nil {
		return 0, err
	}
	if b {
		return dst.WriteString("true")
	}
	return dst.WriteString("false")
}

func rwTime(dst jsWriter, src *Reader) (int, error) {
	t, err := src.ReadTime()
	if err != nil {
		return 0, err
	}
	bts, err := t.MarshalJSON()
	if err != nil {
		return 0, err
	}
	return dst.Write(bts)
}

func rwExtension(dst jsWriter, src *Reader) (n int, err error) {
	et, err := src.peekExtensionType()
	if err != nil {
		return 0, err
	}

	// registered extensions can override
	// the JSON encoding
	if j, ok := extensionReg[et]; ok {
		var bts []byte
		e := j()
		err = src.ReadExtension(e)
		if err != nil {
			return
		}
		bts, err = json.Marshal(e)
		if err != nil {
			return
		}
		return dst.Write(bts)
	}

	e := RawExtension{}
	e.Type = et
	err = src.ReadExtension(&e)
	if err != nil {
		return
	}

	var nn int
	err = dst.WriteByte('{')
	if err != nil {
		return
	}
	n++

	nn, err = dst.WriteString(`"type:"`)
	n += nn
	if err != nil {
		return
	}

	src.scratch = strconv.AppendInt(src.scratch[0:0], int64(e.Type), 10)
	nn, err = dst.Write(src.scratch)
	n += nn
	if err != nil {
		return
	}

	nn, err = dst.WriteString(`,"data":"`)
	n += nn
	if err != nil {
		return
	}

	enc := base64.NewEncoder(base64.StdEncoding, dst)

	nn, err = enc.Write(e.Data)
	n += nn
	if err != nil {
		return
	}
	err = enc.Close()
	if err != nil {
		return
	}
	nn, err = dst.WriteString(`"}`)
	n += nn
	return
}

func rwString(dst jsWriter, src *Reader) (n int, err error) {
	var p []byte
	p, err = src.R.Peek(1)
	if err != nil {
		return
	}
	lead := p[0]
	var read int

	if isfixstr(lead) {
		read = int(rfixstr(lead))
		src.R.Skip(1)
		goto write
	}

	switch lead {
	case mstr8:
		p, err = src.R.Next(2)
		if err != nil {
			return
		}
		read = int(uint8(p[1]))
	case mstr16:
		p, err = src.R.Next(3)
		if err != nil {
			return
		}
		read = int(big.Uint16(p[1:]))
	case mstr32:
		p, err = src.R.Next(5)
		if err != nil {
			return
		}
		read = int(big.Uint32(p[1:]))
	default:
		err = badPrefix(StrType, lead)
		return
	}
write:
	p, err = src.R.Next(read)
	if err != nil {
		return
	}
	n, err = rwquoted(dst, p)
	return
}

func rwBytes(dst jsWriter, src *Reader) (n int, err error) {
	var nn int
	err = dst.WriteByte('"')
	if err != nil {
		return
	}
	n++
	src.scratch, err = src.ReadBytes(src.scratch[:0])
	if err != nil {
		return
	}
	enc := base64.NewEncoder(base64.StdEncoding, dst)
	nn, err = enc.Write(src.scratch)
	n += nn
	if err != nil {
		return
	}
	err = enc.Close()
	if err != nil {
		return
	}
	err = dst.WriteByte('"')
	if err != nil {
		return
	}
	n++
	return
}

// Below (c) The Go Authors, 2009-2014
// Subject to the BSD-style license found at http://golang.org
//
// see: encoding/json/encode.go:(*encodeState).stringbytes()
func rwquoted(dst jsWriter, s []byte) (n int, err error) {
	var nn int
	err = dst.WriteByte('"')
	if err != nil {
		return
	}
	n++
	start := 0
	for i := 0; i < len(s); {
		if b := s[i]; b < utf8.RuneSelf {
			if 0x20 <= b && b != '\\' && b != '"' && b != '<' && b != '>' && b != '&' {
				i++
				continue
			}
			if start < i {
				nn, err = dst.Write(s[start:i])
				n += nn
				if err != nil {
					return
				}
			}
			switch b {
			case '\\', '"':
				err = dst.WriteByte('\\')
				if err != nil {
					return
				}
				n++
				err = dst.WriteByte(b)
				if err != nil {
					return
				}
				n++
			case '\n':
				err = dst.WriteByte('\\')
				if err != nil {
					return
				}
				n++
				err = dst.WriteByte('n')
				if err != nil {
					return
				}
				n++
			case '\r':
				err = dst.WriteByte('\\')
				if err != nil {
					return
				}
				n++
				err = dst.WriteByte('r')
				if err != nil {
					return
				}
				n++
			default:
				nn, err = dst.WriteString(`\u00`)
				n += nn
				if err != nil {
					return
				}
				err = dst.WriteByte(hex[b>>4])
				if err != nil {
					return
				}
				n++
				err = dst.WriteByte(hex[b&0xF])
				if err != nil {
					return
				}
				n++
			}
			i++
			start = i
			continue
		}
		c, size := utf8.DecodeRune(s[i:])
		if c == utf8.RuneError && size == 1 {
			if start < i {
				nn, err = dst.Write(s[start:i])
				n += nn
				if err != nil {
					return
				}
				nn, err = dst.WriteString(`\ufffd`)
				n += nn
				if err != nil {
					return
				}
				i += size
				start = i
				continue
			}
		}
		if c == '\u2028' || c == '\u2029' {
			if start < i {
				nn, err = dst.Write(s[start:i])
				n += nn
				if err != nil {
					return
				}
				nn, err = dst.WriteString(`\u202`)
				n += nn
				if err != nil {
					return
				}
				err = dst.WriteByte(hex[c&0xF])
				if err != nil {
					return
				}
				n++
			}
		}
		i += size
	}
	if start < len(s) {
		nn, err = dst.Write(s[start:])
		n += nn
		if err != nil {
			return
		}
	}
	err = dst.WriteByte('"')
	if err != nil {
		return
	}
	n++
	return
}
