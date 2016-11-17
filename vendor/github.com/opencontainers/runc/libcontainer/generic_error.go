package libcontainer

import (
	"fmt"
	"io"
	"text/template"
	"time"

	"github.com/opencontainers/runc/libcontainer/stacktrace"
)

type syncType uint8

const (
	procReady syncType = iota
	procError
	procRun
	procHooks
	procResume
)

type syncT struct {
	Type syncType `json:"type"`
}

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
	return createSystemError(err, "")
}

func newSystemErrorWithCausef(err error, cause string, v ...interface{}) Error {
	return createSystemError(err, fmt.Sprintf(cause, v...))
}

func newSystemErrorWithCause(err error, cause string) Error {
	return createSystemError(err, cause)
}

// createSystemError creates the specified error with the correct number of
// stack frames skipped. This is only to be called by the other functions for
// formatting the error.
func createSystemError(err error, cause string) Error {
	gerr := &genericError{
		Timestamp: time.Now(),
		Err:       err,
		ECode:     SystemError,
		Cause:     cause,
		Stack:     stacktrace.Capture(2),
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
	Cause     string
	Message   string
	Stack     stacktrace.Stacktrace
}

func (e *genericError) Error() string {
	if e.Cause == "" {
		return e.Message
	}
	frame := e.Stack.Frames[0]
	return fmt.Sprintf("%s:%d: %s caused %q", frame.File, frame.Line, e.Cause, e.Message)
}

func (e *genericError) Code() ErrorCode {
	return e.ECode
}

func (e *genericError) Detail(w io.Writer) error {
	return errorTemplate.Execute(w, e)
}
