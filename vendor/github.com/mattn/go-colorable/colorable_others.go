//go:build !windows && !appengine
// +build !windows,!appengine

package colorable

import (
	"io"
	"os"

	_ "github.com/mattn/go-isatty"
)

// NewColorable returns new instance of Writer which handles escape sequence.
func NewColorable(file *os.File) io.Writer {
	if file == nil {
		panic("nil passed instead of *os.File to NewColorable()")
	}

	return file
}

// NewColorableStdout returns new instance of Writer which handles escape sequence for stdout.
func NewColorableStdout() io.Writer {
	return os.Stdout
}

// NewColorableStderr returns new instance of Writer which handles escape sequence for stderr.
func NewColorableStderr() io.Writer {
	return os.Stderr
}

// EnableColorsStdout enable colors if possible.
func EnableColorsStdout(enabled *bool) func() {
	if enabled != nil {
		*enabled = true
	}
	return func() {}
}
