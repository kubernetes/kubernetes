package yaml

import "gopkg.in/yaml.v2"

// Codec implements the encoding.Encoder and encoding.Decoder interfaces for YAML encoding.
type Codec struct{}

func (Codec) Encode(v interface{}) ([]byte, error) {
	return yaml.Marshal(v)
}

func (Codec) Decode(b []byte, v interface{}) error {
	return yaml.Unmarshal(b, v)
}
