package libcontainer

import (
	"fmt"
	"io"
	"text/template"
	"time"

	"github.com/opencontainers/runc/libcontainer/stacktrace"
)

var errorTemplate = template.Must(template.New("error").Parse(`Timestamp: {{.Timestamp}}
Code: {{.ECode}}
{{if .Message }}
Message: {{.Message}}
{{end}}
Frames:{{range $i, $frame := .Stack.Frames}}
---
{{$i}}: {{$frame.Function}}
Package: {{$frame.Package}}
File: {{$frame.File}}@{{$frame.Line}}{{end}}
`))

func newGenericError(err error, c ErrorCode) Error {
	if le, ok := err.(Error); ok {
		return le
	}
	gerr := &genericError{
		Timestamp: time.Now(),
		Err:       err,
		ECode:     c,
		Stack:     stacktrace.Capture(1),
	}
	if err != nil {
		gerr.Message = err.Error()
	}
	return gerr
}

func newSystemError(err error) Error {
	if le, ok := err.(Error); ok {
		return le
	}
	gerr := &genericError{
		Timestamp: time.Now(),
		Err:       err,
		ECode:     SystemError,
		Stack:     stacktrace.Capture(1),
	}
	if err != nil {
		gerr.Message = err.Error()
	}
	return gerr
}

type genericError struct {
	Timestamp time.Time
	ECode     ErrorCode
	Err       error `json:"-"`
	Message   string
	Stack     stacktrace.Stacktrace
}

func (e *genericError) Error() string {
	return fmt.Sprintf("[%d] %s: %s", e.ECode, e.ECode, e.Message)
}

func (e *genericError) Code() ErrorCode {
	return e.ECode
}

func (e *genericError) Detail(w io.Writer) error {
	return errorTemplate.Execute(w, e)
}
