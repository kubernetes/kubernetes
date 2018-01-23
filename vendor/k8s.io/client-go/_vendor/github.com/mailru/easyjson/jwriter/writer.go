// Package jwriter contains a JSON writer.
package jwriter

import (
	"io"
	"strconv"
	"unicode/utf8"

	"github.com/mailru/easyjson/buffer"
)

// Writer is a JSON writer.
type Writer struct {
	Error  error
	Buffer buffer.Buffer
}

// Size returns the size of the data that was written out.
func (w *Writer) Size() int {
	return w.Buffer.Size()
}

// DumpTo outputs the data to given io.Writer, resetting the buffer.
func (w *Writer) DumpTo(out io.Writer) (written int, err error) {
	return w.Buffer.DumpTo(out)
}

// BuildBytes returns writer data as a single byte slice.
func (w *Writer) BuildBytes() ([]byte, error) {
	if w.Error != nil {
		return nil, w.Error
	}

	return w.Buffer.BuildBytes(), nil
}

// RawByte appends raw binary data to the buffer.
func (w *Writer) RawByte(c byte) {
	w.Buffer.AppendByte(c)
}

// RawByte appends raw binary data to the buffer.
func (w *Writer) RawString(s string) {
	w.Buffer.AppendString(s)
}

// RawByte appends raw binary data to the buffer or sets the error if it is given. Useful for
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

func (w *Writer) Float64(n float64) {
	w.Buffer.EnsureSpace(20)
	w.Buffer.Buf = strconv.AppendFloat(w.Buffer.Buf, n, 'g', -1, 64)
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

func (w *Writer) String(s string) {
	w.Buffer.AppendByte('"')

	// Portions of the string that contain no escapes are appended as
	// byte slices.

	p := 0 // last non-escape symbol

	for i := 0; i < len(s); {
		// single-with character
		if c := s[i]; c < utf8.RuneSelf {
			var escape byte
			switch c {
			case '\t':
				escape = 't'
			case '\r':
				escape = 'r'
			case '\n':
				escape = 'n'
			case '\\':
				escape = '\\'
			case '"':
				escape = '"'
			case '<', '>':
				// do nothing
			default:
				if c >= 0x20 {
					// no escaping is required
					i++
					continue
				}
			}
			if escape != 0 {
				w.Buffer.AppendString(s[p:i])
				w.Buffer.AppendByte('\\')
				w.Buffer.AppendByte(escape)
			} else {
				w.Buffer.AppendString(s[p:i])
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
