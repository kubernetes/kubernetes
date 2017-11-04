package jsoniter

import (
	"bytes"
	"io"
)

// RawMessage to make replace json with jsoniter
type RawMessage []byte

// Unmarshal adapts to json/encoding Unmarshal API
//
// Unmarshal parses the JSON-encoded data and stores the result in the value pointed to by v.
// Refer to https://godoc.org/encoding/json#Unmarshal for more information
func Unmarshal(data []byte, v interface{}) error {
	return ConfigDefault.Unmarshal(data, v)
}

func lastNotSpacePos(data []byte) int {
	for i := len(data) - 1; i >= 0; i-- {
		if data[i] != ' ' && data[i] != '\t' && data[i] != '\r' && data[i] != '\n' {
			return i + 1
		}
	}
	return 0
}

// UnmarshalFromString convenient method to read from string instead of []byte
func UnmarshalFromString(str string, v interface{}) error {
	return ConfigDefault.UnmarshalFromString(str, v)
}

// Get quick method to get value from deeply nested JSON structure
func Get(data []byte, path ...interface{}) Any {
	return ConfigDefault.Get(data, path...)
}

// Marshal adapts to json/encoding Marshal API
//
// Marshal returns the JSON encoding of v, adapts to json/encoding Marshal API
// Refer to https://godoc.org/encoding/json#Marshal for more information
func Marshal(v interface{}) ([]byte, error) {
	return ConfigDefault.Marshal(v)
}

// MarshalIndent same as json.MarshalIndent. Prefix is not supported.
func MarshalIndent(v interface{}, prefix, indent string) ([]byte, error) {
	return ConfigDefault.MarshalIndent(v, prefix, indent)
}

// MarshalToString convenient method to write as string instead of []byte
func MarshalToString(v interface{}) (string, error) {
	return ConfigDefault.MarshalToString(v)
}

// NewDecoder adapts to json/stream NewDecoder API.
//
// NewDecoder returns a new decoder that reads from r.
//
// Instead of a json/encoding Decoder, an Decoder is returned
// Refer to https://godoc.org/encoding/json#NewDecoder for more information
func NewDecoder(reader io.Reader) *Decoder {
	return ConfigDefault.NewDecoder(reader)
}

// Decoder reads and decodes JSON values from an input stream.
// Decoder provides identical APIs with json/stream Decoder (Token() and UseNumber() are in progress)
type Decoder struct {
	iter *Iterator
}

// Decode decode JSON into interface{}
func (adapter *Decoder) Decode(obj interface{}) error {
	adapter.iter.ReadVal(obj)
	err := adapter.iter.Error
	if err == io.EOF {
		return nil
	}
	return adapter.iter.Error
}

// More is there more?
func (adapter *Decoder) More() bool {
	return adapter.iter.head != adapter.iter.tail
}

// Buffered remaining buffer
func (adapter *Decoder) Buffered() io.Reader {
	remaining := adapter.iter.buf[adapter.iter.head:adapter.iter.tail]
	return bytes.NewReader(remaining)
}

// UseNumber for number JSON element, use float64 or json.NumberValue (alias of string)
func (adapter *Decoder) UseNumber() {
	origCfg := adapter.iter.cfg.configBeforeFrozen
	origCfg.UseNumber = true
	adapter.iter.cfg = origCfg.Froze().(*frozenConfig)
}

// NewEncoder same as json.NewEncoder
func NewEncoder(writer io.Writer) *Encoder {
	return ConfigDefault.NewEncoder(writer)
}

// Encoder same as json.Encoder
type Encoder struct {
	stream *Stream
}

// Encode encode interface{} as JSON to io.Writer
func (adapter *Encoder) Encode(val interface{}) error {
	adapter.stream.WriteVal(val)
	adapter.stream.Flush()
	return adapter.stream.Error
}

// SetIndent set the indention. Prefix is not supported
func (adapter *Encoder) SetIndent(prefix, indent string) {
	adapter.stream.cfg.indentionStep = len(indent)
}

// SetEscapeHTML escape html by default, set to false to disable
func (adapter *Encoder) SetEscapeHTML(escapeHTML bool) {
	config := adapter.stream.cfg.configBeforeFrozen
	config.EscapeHTML = escapeHTML
	adapter.stream.cfg = config.Froze().(*frozenConfig)
}
