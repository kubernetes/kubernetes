package jsonlog // import "github.com/docker/docker/daemon/logger/jsonfilelog/jsonlog"

import (
	"bytes"
	"encoding/json"
	"time"
	"unicode/utf8"
)

// JSONLogs marshals encoded JSONLog objects
type JSONLogs struct {
	Log     []byte    `json:"log,omitempty"`
	Stream  string    `json:"stream,omitempty"`
	Created time.Time `json:"time"`

	// json-encoded bytes
	RawAttrs json.RawMessage `json:"attrs,omitempty"`
}

// MarshalJSONBuf is an optimized JSON marshaller that avoids reflection
// and unnecessary allocation.
func (mj *JSONLogs) MarshalJSONBuf(buf *bytes.Buffer) error {
	var first = true

	buf.WriteString(`{`)
	if len(mj.Log) != 0 {
		first = false
		buf.WriteString(`"log":`)
		ffjsonWriteJSONBytesAsString(buf, mj.Log)
	}
	if len(mj.Stream) != 0 {
		if first {
			first = false
		} else {
			buf.WriteString(`,`)
		}
		buf.WriteString(`"stream":`)
		ffjsonWriteJSONBytesAsString(buf, []byte(mj.Stream))
	}
	if len(mj.RawAttrs) > 0 {
		if first {
			first = false
		} else {
			buf.WriteString(`,`)
		}
		buf.WriteString(`"attrs":`)
		buf.Write(mj.RawAttrs)
	}
	if !first {
		buf.WriteString(`,`)
	}

	created, err := fastTimeMarshalJSON(mj.Created)
	if err != nil {
		return err
	}

	buf.WriteString(`"time":`)
	buf.WriteString(created)
	buf.WriteString(`}`)
	return nil
}

func ffjsonWriteJSONBytesAsString(buf *bytes.Buffer, s []byte) {
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
