package streamformatter

import (
	"encoding/json"
	"io"

	"github.com/docker/docker/pkg/jsonmessage"
)

type streamWriter struct {
	io.Writer
	lineFormat func([]byte) string
}

func (sw *streamWriter) Write(buf []byte) (int, error) {
	formattedBuf := sw.format(buf)
	n, err := sw.Writer.Write(formattedBuf)
	if n != len(formattedBuf) {
		return n, io.ErrShortWrite
	}
	return len(buf), err
}

func (sw *streamWriter) format(buf []byte) []byte {
	msg := &jsonmessage.JSONMessage{Stream: sw.lineFormat(buf)}
	b, err := json.Marshal(msg)
	if err != nil {
		return FormatError(err)
	}
	return appendNewline(b)
}

// NewStdoutWriter returns a writer which formats the output as json message
// representing stdout lines
func NewStdoutWriter(out io.Writer) io.Writer {
	return &streamWriter{Writer: out, lineFormat: func(buf []byte) string {
		return string(buf)
	}}
}

// NewStderrWriter returns a writer which formats the output as json message
// representing stderr lines
func NewStderrWriter(out io.Writer) io.Writer {
	return &streamWriter{Writer: out, lineFormat: func(buf []byte) string {
		return "\033[91m" + string(buf) + "\033[0m"
	}}
}
