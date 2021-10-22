// +build codegen

package api

import (
	"fmt"
	"io"
	"sync"
)

var debugLogger *logger
var initDebugLoggerOnce sync.Once

// logger provides a basic logging
type logger struct {
	w io.Writer
}

// LogDebug initialize's the debug logger for the components in the api
// package to log debug lines to.
//
// Panics if called multiple times.
//
// Must be used prior to any model loading or code gen.
func LogDebug(w io.Writer) {
	var initialized bool
	initDebugLoggerOnce.Do(func() {
		debugLogger = &logger{
			w: w,
		}
		initialized = true
	})

	if !initialized && debugLogger != nil {
		panic("LogDebug called multiple times. Can only be called once")
	}
}

// Logf logs using the fmt printf pattern. Appends a new line to the end of the
// logged statement.
func (l *logger) Logf(format string, args ...interface{}) {
	if l == nil {
		return
	}
	fmt.Fprintf(l.w, format+"\n", args...)
}

// Logln logs using the fmt println pattern.
func (l *logger) Logln(args ...interface{}) {
	if l == nil {
		return
	}
	fmt.Fprintln(l.w, args...)
}
