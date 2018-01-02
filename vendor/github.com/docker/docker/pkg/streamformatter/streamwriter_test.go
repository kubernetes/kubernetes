package streamformatter

import (
	"testing"

	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamWriterStdout(t *testing.T) {
	buffer := &bytes.Buffer{}
	content := "content"
	sw := NewStdoutWriter(buffer)
	size, err := sw.Write([]byte(content))

	require.NoError(t, err)
	assert.Equal(t, len(content), size)

	expected := `{"stream":"content"}` + streamNewline
	assert.Equal(t, expected, buffer.String())
}

func TestStreamWriterStderr(t *testing.T) {
	buffer := &bytes.Buffer{}
	content := "content"
	sw := NewStderrWriter(buffer)
	size, err := sw.Write([]byte(content))

	require.NoError(t, err)
	assert.Equal(t, len(content), size)

	expected := `{"stream":"\u001b[91mcontent\u001b[0m"}` + streamNewline
	assert.Equal(t, expected, buffer.String())
}
