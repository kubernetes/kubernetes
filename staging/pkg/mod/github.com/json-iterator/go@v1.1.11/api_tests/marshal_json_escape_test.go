package test

import (
	"bytes"
	"encoding/json"
	"testing"

	jsoniter "github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
)

var marshalConfig = jsoniter.Config{
	EscapeHTML:             false,
	SortMapKeys:            true,
	ValidateJsonRawMessage: true,
}.Froze()

type Container struct {
	Bar interface{}
}

func (c *Container) MarshalJSON() ([]byte, error) {
	return marshalConfig.Marshal(&c.Bar)
}

func TestEncodeEscape(t *testing.T) {
	should := require.New(t)

	container := &Container{
		Bar: []string{"123<ab>", "ooo"},
	}
	out, err := marshalConfig.Marshal(container)
	should.Nil(err)
	bufout := string(out)

	var stdbuf bytes.Buffer
	stdenc := json.NewEncoder(&stdbuf)
	stdenc.SetEscapeHTML(false)
	err = stdenc.Encode(container)
	should.Nil(err)
	stdout := string(stdbuf.Bytes())
	if stdout[len(stdout)-1:] == "\n" {
		stdout = stdout[:len(stdout)-1]
	}

	should.Equal(stdout, bufout)
}
