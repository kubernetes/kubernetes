package test

import (
	"bytes"
	"encoding"
	"encoding/base64"
	"strings"
)

func init() {
	testCases = append(testCases,
		(*structTextMarshaler)(nil),
		(*structTextMarshalerAlias)(nil),
		(*struct {
			S string
			M structTextMarshaler
			I int8
		})(nil),
		(*struct {
			S string
			M structTextMarshalerAlias
			I int8
		})(nil),
	)
}

type structTextMarshaler struct {
	X string
}

func (m structTextMarshaler) encode(str string) string {
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

func (m structTextMarshaler) decode(str string) string {
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

func (m structTextMarshaler) MarshalText() ([]byte, error) {
	return []byte(`MANUAL__` + m.encode(m.X)), nil
}

func (m *structTextMarshaler) UnmarshalText(text []byte) error {
	m.X = m.decode(strings.TrimPrefix(string(text), "MANUAL__"))
	return nil
}

var _ encoding.TextMarshaler = structTextMarshaler{}
var _ encoding.TextUnmarshaler = &structTextMarshaler{}

type structTextMarshalerAlias structTextMarshaler
