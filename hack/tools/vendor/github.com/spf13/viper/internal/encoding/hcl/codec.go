package hcl

import (
	"bytes"
	"encoding/json"

	"github.com/hashicorp/hcl"
	"github.com/hashicorp/hcl/hcl/printer"
)

// Codec implements the encoding.Encoder and encoding.Decoder interfaces for HCL encoding.
// TODO: add printer config to the codec?
type Codec struct{}

func (Codec) Encode(v interface{}) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}

	// TODO: use printer.Format? Is the trailing newline an issue?

	ast, err := hcl.Parse(string(b))
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer

	err = printer.Fprint(&buf, ast.Node)
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func (Codec) Decode(b []byte, v interface{}) error {
	return hcl.Unmarshal(b, v)
}
