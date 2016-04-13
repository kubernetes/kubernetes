package jsonlog

import (
	"bytes"
	"unicode/utf8"
)

// JSONLogBytes is based on JSONLog.
// It allows marshalling JSONLog from Log as []byte
// and an already marshalled Created timestamp.
type JSONLogBytes struct {
	Log     []byte `json:"log,omitempty"`
	Stream  string `json:"stream,omitempty"`
	Created string `json:"time"`
}

// MarshalJSONBuf is based on the same method from JSONLog
// It has been modified to take into account the necessary changes.
func (mj *JSONLogBytes) MarshalJSONBuf(buf *bytes.Buffer) error {
	var first = true

	buf.WriteString(`{`)
	if len(mj.Log) != 0 {
		first = false
		buf.WriteString(`"log":`)
		ffjson_WriteJsonBytesAsString(buf, mj.Log)
	}
	if len(mj.Stream) != 0 {
		if first == true {
			first = false
		} else {
			buf.WriteString(`,`)
		}
		buf.WriteString(`"stream":`)
		ffjson_WriteJsonString(buf, mj.Stream)
	}
	if first == true {
		first = false
	} else {
		buf.WriteString(`,`)
	}
	buf.WriteString(`"time":`)
	buf.WriteString(mj.Created)
	buf.WriteString(`}`)
	return nil
}

// This is based on ffjson_WriteJsonString. It has been changed
// to accept a string passed as a slice of bytes.
func ffjson_WriteJsonBytesAsString(buf *bytes.Buffer, s []byte) {
	const hex = "0123456789abcdef"

	buf.WriteByte('"')
	start := 0
	for i := 0; i < len(s); {
		if b := s[i]; b < utf8.RuneSelf {
			if 0x20 <= b && b != '\\' && b != '"' && b != '<' && b != '>' && b != '&' {
				i++
				continue
			}
			if start < i {
				buf.Write(s[start:i])
			}
			switch b {
			case '\\', '"':
				buf.WriteByte('\\')
				buf.WriteByte(b)
			case '\n':
				buf.WriteByte('\\')
				buf.WriteByte('n')
			case '\r':
				buf.WriteByte('\\')
				buf.WriteByte('r')
			default:

				buf.WriteString(`\u00`)
				buf.WriteByte(hex[b>>4])
				buf.WriteByte(hex[b&0xF])
			}
			i++
			start = i
			continue
		}
		c, size := utf8.DecodeRune(s[i:])
		if c == utf8.RuneError && size == 1 {
			if start < i {
				buf.Write(s[start:i])
			}
			buf.WriteString(`\ufffd`)
			i += size
			start = i
			continue
		}

		if c == '\u2028' || c == '\u2029' {
			if start < i {
				buf.Write(s[start:i])
			}
			buf.WriteString(`\u202`)
			buf.WriteByte(hex[c&0xF])
			i += size
			start = i
			continue
		}
		i += size
	}
	if start < len(s) {
		buf.Write(s[start:])
	}
	buf.WriteByte('"')
}
