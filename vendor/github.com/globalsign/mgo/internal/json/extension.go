package json

import (
	"reflect"
)

// Extension holds a set of additional rules to be used when unmarshaling
// strict JSON or JSON-like content.
type Extension struct {
	funcs  map[string]funcExt
	consts map[string]interface{}
	keyed  map[string]func([]byte) (interface{}, error)
	encode map[reflect.Type]func(v interface{}) ([]byte, error)

	unquotedKeys   bool
	trailingCommas bool
}

type funcExt struct {
	key  string
	args []string
}

// Extend changes the decoder behavior to consider the provided extension.
func (dec *Decoder) Extend(ext *Extension) { dec.d.ext = *ext }

// Extend changes the encoder behavior to consider the provided extension.
func (enc *Encoder) Extend(ext *Extension) { enc.ext = *ext }

// Extend includes in e the extensions defined in ext.
func (e *Extension) Extend(ext *Extension) {
	for name, fext := range ext.funcs {
		e.DecodeFunc(name, fext.key, fext.args...)
	}
	for name, value := range ext.consts {
		e.DecodeConst(name, value)
	}
	for key, decode := range ext.keyed {
		e.DecodeKeyed(key, decode)
	}
	for typ, encode := range ext.encode {
		if e.encode == nil {
			e.encode = make(map[reflect.Type]func(v interface{}) ([]byte, error))
		}
		e.encode[typ] = encode
	}
}

// DecodeFunc defines a function call that may be observed inside JSON content.
// A function with the provided name will be unmarshaled as the document
// {key: {args[0]: ..., args[N]: ...}}.
func (e *Extension) DecodeFunc(name string, key string, args ...string) {
	if e.funcs == nil {
		e.funcs = make(map[string]funcExt)
	}
	e.funcs[name] = funcExt{key, args}
}

// DecodeConst defines a constant name that may be observed inside JSON content
// and will be decoded with the provided value.
func (e *Extension) DecodeConst(name string, value interface{}) {
	if e.consts == nil {
		e.consts = make(map[string]interface{})
	}
	e.consts[name] = value
}

// DecodeKeyed defines a key that when observed as the first element inside a
// JSON document triggers the decoding of that document via the provided
// decode function.
func (e *Extension) DecodeKeyed(key string, decode func(data []byte) (interface{}, error)) {
	if e.keyed == nil {
		e.keyed = make(map[string]func([]byte) (interface{}, error))
	}
	e.keyed[key] = decode
}

// DecodeUnquotedKeys defines whether to accept map keys that are unquoted strings.
func (e *Extension) DecodeUnquotedKeys(accept bool) {
	e.unquotedKeys = accept
}

// DecodeTrailingCommas defines whether to accept trailing commas in maps and arrays.
func (e *Extension) DecodeTrailingCommas(accept bool) {
	e.trailingCommas = accept
}

// EncodeType registers a function to encode values with the same type of the
// provided sample.
func (e *Extension) EncodeType(sample interface{}, encode func(v interface{}) ([]byte, error)) {
	if e.encode == nil {
		e.encode = make(map[reflect.Type]func(v interface{}) ([]byte, error))
	}
	e.encode[reflect.TypeOf(sample)] = encode
}
