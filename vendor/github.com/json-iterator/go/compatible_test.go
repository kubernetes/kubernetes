package jsoniter

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

// Standard Encoder has trailing newline.
func TestEncoderHasTrailingNewline(t *testing.T) {
	should := require.New(t)
	var buf, stdbuf bytes.Buffer
	enc := ConfigCompatibleWithStandardLibrary.NewEncoder(&buf)
	enc.Encode(1)
	stdenc := json.NewEncoder(&stdbuf)
	stdenc.Encode(1)
	should.Equal(stdbuf.Bytes(), buf.Bytes())
}

// Non-nil but empty map should be ignored.
func TestOmitempty(t *testing.T) {
	o := struct {
		A           string            `json:"a,omitempty"`
		B           string            `json:"b,omitempty"`
		Annotations map[string]string `json:"annotations,omitempty"`
	}{
		A:           "a",
		B:           "b",
		Annotations: map[string]string{},
	}
	should := require.New(t)
	var buf, stdbuf bytes.Buffer
	enc := ConfigCompatibleWithStandardLibrary.NewEncoder(&buf)
	enc.Encode(o)
	stdenc := json.NewEncoder(&stdbuf)
	stdenc.Encode(o)
	should.Equal(string(stdbuf.Bytes()), string(buf.Bytes()))
}
