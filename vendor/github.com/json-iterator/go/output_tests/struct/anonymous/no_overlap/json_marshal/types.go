package test

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"
)

// MarshalerForTest TEST ONLY
type MarshalerForTest string

func encode(str string) string {
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

func decode(str string) string {
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

// MarshalJSON TEST ONLY
func (m MarshalerForTest) MarshalJSON() ([]byte, error) {
	return []byte(`"MANUAL__` + encode(string(m)) + `"`), nil
}

// UnmarshalJSON TEST ONLY
func (m *MarshalerForTest) UnmarshalJSON(text []byte) error {
	*m = MarshalerForTest(decode(strings.TrimPrefix(strings.Trim(string(text), `"`), "MANUAL__")))
	return nil
}

var _ json.Marshaler = *new(MarshalerForTest)
var _ json.Unmarshaler = new(MarshalerForTest)

type typeForTest struct {
	F1 float64
	MarshalerForTest
	F2 int32
}
