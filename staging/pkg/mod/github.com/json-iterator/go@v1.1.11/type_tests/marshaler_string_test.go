package test

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"
)

type StringMarshaler string

func (m StringMarshaler) encode(str string) string {
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

func (m StringMarshaler) decode(str string) string {
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

func (m StringMarshaler) MarshalJSON() ([]byte, error) {
	return []byte(`"MANUAL__` + m.encode(string(m)) + `"`), nil
}

func (m *StringMarshaler) UnmarshalJSON(text []byte) error {
	*m = StringMarshaler(m.decode(strings.TrimPrefix(strings.Trim(string(text), `"`), "MANUAL__")))
	return nil
}

var _ json.Marshaler = *new(StringMarshaler)
var _ json.Unmarshaler = new(StringMarshaler)

func init() {
	testCases = append(testCases, (*StringMarshaler)(nil))
}
