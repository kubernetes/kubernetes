package test

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"
)

type structMarshaler struct {
	X string
}

func (m structMarshaler) encode(str string) string {
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

func (m structMarshaler) decode(str string) string {
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

func (m structMarshaler) MarshalJSON() ([]byte, error) {
	return []byte(`"MANUAL__` + m.encode(m.X) + `"`), nil
}

func (m *structMarshaler) UnmarshalJSON(text []byte) error {
	m.X = m.decode(strings.TrimPrefix(strings.Trim(string(text), `"`), "MANUAL__"))
	return nil
}

var _ json.Marshaler = structMarshaler{}
var _ json.Unmarshaler = &structMarshaler{}

type structMarshalerAlias structMarshaler

func init() {
	testCases = append(testCases,
		(*structMarshaler)(nil),
		(*structMarshalerAlias)(nil),
		(*struct {
			S string
			M structMarshaler
			I int8
		})(nil),
		(*struct {
			S string
			M structMarshalerAlias
			I int8
		})(nil),
	)
}
