package log

import (
	"fmt"
	"os"

	"github.com/fatih/color"
)

type Level uint8

const (
	ErrorLevel Level = iota
	WarnLevel
	DebugLevel
)

var (
	level              = WarnLevel
	out   stringWriter = os.Stderr
)

// TODO: replace with io.StringWriter once support for go1.11 is dropped.
type stringWriter interface {
	WriteString(s string) (n int, err error)
}

// SetLevel for the global logger.
func SetLevel(l Level) {
	level = l
}

// Warnf prints the message to stderr, with a yellow WARN prefix.
func Warnf(format string, args ...interface{}) {
	if level < WarnLevel {
		return
	}
	out.WriteString(color.YellowString("WARN "))
	out.WriteString(fmt.Sprintf(format, args...))
	out.WriteString("\n")
}

// Debugf prints the message to stderr, with no prefix.
func Debugf(format string, args ...interface{}) {
	if level < DebugLevel {
		return
	}
	out.WriteString(fmt.Sprintf(format, args...))
	out.WriteString("\n")
}

// Errorf prints the message to stderr, with a red ERROR prefix.
func Errorf(format string, args ...interface{}) {
	if level < ErrorLevel {
		return
	}
	out.WriteString(color.RedString("ERROR "))
	out.WriteString(fmt.Sprintf(format, args...))
	out.WriteString("\n")
}

// Error prints the message to stderr, with a red ERROR prefix.
func Error(msg string) {
	if level < ErrorLevel {
		return
	}
	out.WriteString(color.RedString("ERROR "))
	out.WriteString(msg)
	out.WriteString("\n")
}
