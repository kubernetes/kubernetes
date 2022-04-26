package toml

import (
	"encoding"
	"io"
)

// Deprecated: use encoding.TextMarshaler
type TextMarshaler encoding.TextMarshaler

// Deprecated: use encoding.TextUnmarshaler
type TextUnmarshaler encoding.TextUnmarshaler

// Deprecated: use MetaData.PrimitiveDecode.
func PrimitiveDecode(primValue Primitive, v interface{}) error {
	md := MetaData{decoded: make(map[string]struct{})}
	return md.unify(primValue.undecoded, rvalue(v))
}

// Deprecated: use NewDecoder(reader).Decode(&value).
func DecodeReader(r io.Reader, v interface{}) (MetaData, error) { return NewDecoder(r).Decode(v) }
