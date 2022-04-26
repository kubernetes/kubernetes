package json

import (
	"encoding/json"
)

// Codec implements the encoding.Encoder and encoding.Decoder interfaces for JSON encoding.
type Codec struct{}

func (Codec) Encode(v interface{}) ([]byte, error) {
	// TODO: expose prefix and indent in the Codec as setting?
	return json.MarshalIndent(v, "", "  ")
}

func (Codec) Decode(b []byte, v interface{}) error {
	return json.Unmarshal(b, v)
}
