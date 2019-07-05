// Package jwriter contains a JSON writer.
package jwriter

import (
	"io"
	"strconv"
	"unicode/utf8"

	"github.com/mailru/easyjson/buffer"
)

// Flags describe various encoding options. The behavior may be actually implemented in the encoder, but
// Flags field in Writer is used to set and pass them around.
type Flags int

const (
	NilMapAsEmpty   Flags = 1 << iota // Encode nil map as '{}' rather than 'null'.
	NilSliceAsEmpty                   // Encode nil slice as '[]' rather than 'null'.
)

// Writer is a JSON writer.
type Writer struct {
	Flags Flags

	Error        error
	Buffer       buffer.Buffer
	NoEscapeHTML bool
}

// Size returns the size of the data that was written out.
func (w *Writer) Size() int {
	return w.Buffer.Size()
}

// DumpTo outputs the data to given io.Writer, resetting the buffer.
func (w *Writer) DumpTo(out io.Writer) (written int, err error) {
	return w.Buffer.DumpTo(out)
}

// BuildBytes returns writer data as a single byte slice. You can optionally provide one byte slice
// as argument that it will try to reuse.
func (w *Writer) BuildBytes(reuse ...[]byte) ([]byte, error) {
	if w.Error != nil {
		return nil, w.Error
	}

	return w.Buffer.BuildBytes(reuse...), nil
}

// ReadCloser returns an io.ReadCloser that can be used to read the data.
// ReadCloser also resets the buffer.
func (w *Writer) ReadCloser() (io.ReadCloser, error) {
	if w.Error != nil {
		return nil, w.Error
	}

	return w.Buffer.ReadCloser(), nil
}

// RawByte appends raw binary data to the buffer.
func (w *Writer) RawByte(c byte) {
	w.Buffer.AppendByte(c)
}

// RawByte appends raw binary data to the buffer.
func (w *Writer) RawString(s string) {
	w.Buffer.AppendString(s)
}

// Raw appends raw binary data to the buffer or sets the error if it is given. Useful for
// calling with results of MarshalJSON-like functions.
func (w *Writer) Raw(data []byte, err error) {
	switch {
	case w.Error != nil:
		return
	case err != nil:
		w.Error = err
	case len(data) > 0:
		w.Buffer.AppendBytes(data)
	default:
		w.RawString("null")
	}
}

// RawText encloses raw binary data in quotes and appends in to the buffer.
// Useful for calling with results of MarshalText-like functions.
func (w *Writer) RawText(data []byte, err error) {
	switch {
	case w.Error != nil:
		return
	case err != nil:
		w.Error = err
	case len(data) > 0:
		w.String(string(data))
	default:
		w.RawString("null")
	}
}

// Base64Bytes appends data to the buffer after base64 encoding it
func (w *Writer) Base64Bytes(data []byte) {
	if data == nil {
		w.Buffer.AppendString("null")
		return
	}
	w.Buffer.AppendByte('"')
	w.base64(data)
	w.Buffer.AppendByte('"')
}

func (w *Writer) Uint8(n uint8) {
	w.Buffer.EnsureSpace(3)
	w.Buffer.Buf = strconv.AppendUint(w.Buffer.Buf, uint64(n), 10)
}

func (w *Writer) Uint16(n uint16) {
	w.Buffer.EnsureSpace(5)
	w.Buffer.Buf = strconv.AppendUint(w.Buffer.Buf, uint64(n), 10)
}

func (w *Writer) Uint32(n uint32) {
	w.Buffer.EnsureSpace(10)
	w.Buffer.Buf = strconv.AppendUint(w.Buffer.Buf, uint64(n), 10)
}

func (w *Writer) Uint(n uint) {
	w.Buffer.EnsureSpace(20)
	w.Buffer.Buf = strconv.AppendUint(w.Buffer.Buf, uint64(n), 10)
}

func (w *Writer) Uint64(n uint64) {
	w.Buffer.EnsureSpace(20)
	w.Buffer.Buf = strconv.AppendUint(w.Buffer.Buf, n, 10)
}

func (w *Writer) Int8(n int8) {
	w.Buffer.EnsureSpace(4)
	w.Buffer.Buf = strconv.AppendInt(w.Buffer.Buf, int64(n), 10)
}

func (w *Writer) Int16(n int16) {
	w.Buffer.EnsureSpace(6)
	w.Buffer.Buf = strconv.AppendInt(w.Buffer.Buf, int64(n), 10)
}

func (w *Writer) Int32(n int32) {
	w.Buffer.EnsureSpace(11)
	w.Buffer.Buf = strconv.AppendInt(w.Buffer.Buf, int64(n), 10)
}

func (w *Writer) Int(n int) {
	w.Buffer.EnsureSpace(21)
	w.Buffer.Buf = strconv.AppendInt(w.Buffer.Buf, int64(n), 10)
}

func (w *Writer) Int64(n int64) {
	w.Buffer.EnsureSpace(21)
	w.Buffer.Buf = strconv.AppendInt(w.Buffer.Buf, n, 10)
}

func (w *Writer) Uint8Str(n uint8) {
	w.Buffer.EnsureSpace(3)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendUint(w.Buffer.Buf, uint64(n), 10)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) Uint16Str(n uint16) {
	w.Buffer.EnsureSpace(5)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendUint(w.Buffer.Buf, uint64(n), 10)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) Uint32Str(n uint32) {
	w.Buffer.EnsureSpace(10)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendUint(w.Buffer.Buf, uint64(n), 10)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) UintStr(n uint) {
	w.Buffer.EnsureSpace(20)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendUint(w.Buffer.Buf, uint64(n), 10)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) Uint64Str(n uint64) {
	w.Buffer.EnsureSpace(20)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendUint(w.Buffer.Buf, n, 10)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) UintptrStr(n uintptr) {
	w.Buffer.EnsureSpace(20)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendUint(w.Buffer.Buf, uint64(n), 10)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) Int8Str(n int8) {
	w.Buffer.EnsureSpace(4)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendInt(w.Buffer.Buf, int64(n), 10)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) Int16Str(n int16) {
	w.Buffer.EnsureSpace(6)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendInt(w.Buffer.Buf, int64(n), 10)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) Int32Str(n int32) {
	w.Buffer.EnsureSpace(11)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendInt(w.Buffer.Buf, int64(n), 10)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) IntStr(n int) {
	w.Buffer.EnsureSpace(21)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendInt(w.Buffer.Buf, int64(n), 10)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) Int64Str(n int64) {
	w.Buffer.EnsureSpace(21)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendInt(w.Buffer.Buf, n, 10)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) Float32(n float32) {
	w.Buffer.EnsureSpace(20)
	w.Buffer.Buf = strconv.AppendFloat(w.Buffer.Buf, float64(n), 'g', -1, 32)
}

func (w *Writer) Float32Str(n float32) {
	w.Buffer.EnsureSpace(20)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendFloat(w.Buffer.Buf, float64(n), 'g', -1, 32)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) Float64(n float64) {
	w.Buffer.EnsureSpace(20)
	w.Buffer.Buf = strconv.AppendFloat(w.Buffer.Buf, n, 'g', -1, 64)
}

func (w *Writer) Float64Str(n float64) {
	w.Buffer.EnsureSpace(20)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
	w.Buffer.Buf = strconv.AppendFloat(w.Buffer.Buf, float64(n), 'g', -1, 64)
	w.Buffer.Buf = append(w.Buffer.Buf, '"')
}

func (w *Writer) Bool(v bool) {
	w.Buffer.EnsureSpace(5)
	if v {
		w.Buffer.Buf = append(w.Buffer.Buf, "true"...)
	} else {
		w.Buffer.Buf = append(w.Buffer.Buf, "false"...)
	}
}

const chars = "0123456789abcdef"

func isNotEscapedSingleChar(c byte, escapeHTML bool) bool {
	// Note: might make sense to use a table if there are more chars to escape. With 4 chars
	// it benchmarks the same.
	if escapeHTML {
		return c != '<' && c != '>' && c != '&' && c != '\\' && c != '"' && c >= 0x20 && c < utf8.RuneSelf
	} else {
		return c != '\\' && c != '"' && c >= 0x20 && c < utf8.RuneSelf
	}
}

func (w *Writer) String(s string) {
	w.Buffer.AppendByte('"')

	// Portions of the string that contain no escapes are appended as
	// byte slices.

	p := 0 // last non-escape symbol

	for i := 0; i < len(s); {
		c := s[i]

		if isNotEscapedSingleChar(c, !w.NoEscapeHTML) {
			// single-width character, no escaping is required
			i++
			continue
		} else if c < utf8.RuneSelf {
			// single-with character, need to escape
			w.Buffer.AppendString(s[p:i])
			switch c {
			case '\t':
				w.Buffer.AppendString(`\t`)
			case '\r':
				w.Buffer.AppendString(`\r`)
			case '\n':
				w.Buffer.AppendString(`\n`)
			case '\\':
				w.Buffer.AppendString(`\\`)
			case '"':
				w.Buffer.AppendString(`\"`)
			default:
				w.Buffer.AppendString(`\u00`)
				w.Buffer.AppendByte(chars[c>>4])
				w.Buffer.AppendByte(chars[c&0xf])
			}

			i++
			p = i
			continue
		}

		// broken utf
		runeValue, runeWidth := utf8.DecodeRuneInString(s[i:])
		if runeValue == utf8.RuneError && runeWidth == 1 {
			w.Buffer.AppendString(s[p:i])
			w.Buffer.AppendString(`\ufffd`)
			i++
			p = i
			continue
		}

		// jsonp stuff - tab separator and line separator
		if runeValue == '\u2028' || runeValue == '\u2029' {
			w.Buffer.AppendString(s[p:i])
			w.Buffer.AppendString(`\u202`)
			w.Buffer.AppendByte(chars[runeValue&0xf])
			i += runeWidth
			p = i
			continue
		}
		i += runeWidth
	}
	w.Buffer.AppendString(s[p:])
	w.Buffer.AppendByte('"')
}

const encode = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
const padChar = '='

func (w *Writer) base64(in []byte) {

	if len(in) == 0 {
		return
	}

	w.Buffer.EnsureSpace(((len(in)-1)/3 + 1) * 4)

	si := 0
	n := (len(in) / 3) * 3

	for si < n {
		// Convert 3x 8bit source bytes into 4 bytes
		val := uint(in[si+0])<<16 | uint(in[si+1])<<8 | uint(in[si+2])

		w.Buffer.Buf = append(w.Buffer.Buf, encode[val>>18&0x3F], encode[val>>12&0x3F], encode[val>>6&0x3F], encode[val&0x3F])

		si += 3
	}

	remain := len(in) - si
	if remain == 0 {
		return
	}

	// Add the remaining small block
	val := uint(in[si+0]) << 16
	if remain == 2 {
		val |= uint(in[si+1]) << 8
	}

	w.Buffer.Buf = append(w.Buffer.Buf, encode[val>>18&0x3F], encode[val>>12&0x3F])

	switch remain {
	case 2:
		w.Buffer.Buf = append(w.Buffer.Buf, encode[val>>6&0x3F], byte(padChar))
	case 1:
		w.Buffer.Buf = append(w.Buffer.Buf, byte(padChar), byte(padChar))
	}
}
