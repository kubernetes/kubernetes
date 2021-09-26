package test

import (
	"bytes"
	"encoding"
	"encoding/base64"
	"strings"
)

func init() {
	testCases = append(testCases,
		(*StringTextMarshaler)(nil),
	)
}

// StringTextMarshaler TEST ONLY
type StringTextMarshaler string

func (m StringTextMarshaler) encode(str string) string {
	buf := bytes.Buffer{}
	b64 := base64.NewEncoder(base64.StdEncoding, &buf)
	if _, err := b64.Write([]byte(str)); err != nil {
		panic(err)
	}
	if err := b64.Close(); err != nil {
		panic(err)
	}
	return buf.String()
}

func (m StringTextMarshaler) decode(str string) string {
	if len(str) == 0 {
		return ""
	}
	b64 := base64.NewDecoder(base64.StdEncoding, strings.NewReader(str))
	bs := make([]byte, len(str))
	if n, err := b64.Read(bs); err != nil {
		panic(err)
	} else {
		bs = bs[:n]
	}
	return string(bs)
}

// MarshalText TEST ONLY
func (m StringTextMarshaler) MarshalText() ([]byte, error) {
	return []byte(`MANUAL__` + m.encode(string(m))), nil
}

// UnmarshalText TEST ONLY
func (m *StringTextMarshaler) UnmarshalText(text []byte) error {
	*m = StringTextMarshaler(m.decode(strings.TrimPrefix(string(text), "MANUAL__")))
	return nil
}

var _ encoding.TextMarshaler = *new(StringTextMarshaler)
var _ encoding.TextUnmarshaler = new(StringTextMarshaler)
