// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package log

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/hashicorp/errwrap"
)

// Logger is an extended version of the golang Logger to support structured errors.
type Logger struct {
	debug bool
	*log.Logger
}

// New creates a new Logger with no Log flags set.
func New(out io.Writer, prefix string, debug bool) *Logger {
	l := &Logger{
		debug:  debug,
		Logger: log.New(out, prefix, 0),
	}
	l.SetFlags(0)
	return l
}

// NewLogSet returns a set of Loggers for commonly used output streams: errors,
// diagnostics, stdout. The error and stdout streams should generally never be
// suppressed. diagnostic can be suppressed by setting the output to
// ioutil.Discard. If an output destination is not needed, one can simply
// discard it by assigning it to '_'.
func NewLogSet(prefix string, debug bool) (stderr, diagnostic, stdout *Logger) {
	stderr = New(os.Stderr, prefix, debug)
	diagnostic = New(os.Stderr, prefix, debug)
	// Debug not used for stdout.
	stdout = New(os.Stdout, prefix, false)

	return stderr, diagnostic, stdout
}

// SetDebug sets the debug flag to the value of b
func (l *Logger) SetDebug(b bool) { l.debug = b }

// SetFlags is a wrapper around log.SetFlags that adds and removes, ": " to and
// from a prefix. This is needed because ": " is only added by golang's log
// package if either of the Lshortfile or Llongfile flags are set.
func (l *Logger) SetFlags(flag int) {
	l.Logger.SetFlags(flag)

	// Only proceed if we've actually got a prefix
	if l.Prefix() == "" {
		return
	}

	const clnSpc = ": "
	if flag&(log.Lshortfile|log.Llongfile) != 0 {
		l.SetPrefix(strings.TrimSuffix(l.Prefix(), clnSpc))
	} else {
		l.SetPrefix(l.Prefix() + clnSpc)
	}
}

func (l *Logger) formatErr(e error, msg string) string {
	// Get a list of accumulated errors
	var errors []error
	errwrap.Walk(e, func(err error) {
		errors = append(errors, err)
	})

	var buf bytes.Buffer
	buf.WriteString(msg)

	if !l.debug {
		if len(msg) > 0 {
			buf.WriteString(": ")
		}
		buf.WriteString(errors[len(errors)-1].Error())
	} else {
		for i, childErr := range errors {
			buf.WriteString(fmt.Sprintf("\n%s%s%v", strings.Repeat("  ", i+1), "└─", childErr))
		}
	}

	return strings.TrimSuffix(buf.String(), "\n")
}

// PrintE prints the msg and its error message(s).
func (l *Logger) PrintE(msg string, e error) {
	l.Print(l.formatErr(e, msg))
}

// Error is a convenience function for printing errors without a message.
func (l *Logger) Error(e error) {
	l.Print(l.formatErr(e, ""))
}

// Errorf is a convenience function for formatting and printing errors.
func (l *Logger) Errorf(format string, a ...interface{}) {
	l.Print(l.formatErr(fmt.Errorf(format, a...), ""))
}

// FatalE prints a string and error then calls os.Exit(254).
func (l *Logger) FatalE(msg string, e error) {
	l.Print(l.formatErr(e, msg))
	os.Exit(254)
}

// Fatalf prints an error then calls os.Exit(254).
func (l *Logger) Fatalf(format string, a ...interface{}) {
	l.Print(l.formatErr(fmt.Errorf(format, a...), ""))
	os.Exit(254)
}

// PanicE prints a string and error then calls panic.
func (l *Logger) PanicE(msg string, e error) {
	l.Panic(l.formatErr(e, msg))
}
