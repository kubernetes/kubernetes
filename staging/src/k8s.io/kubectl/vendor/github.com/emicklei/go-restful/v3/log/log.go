package log

import (
	stdlog "log"
	"os"
)

// StdLogger corresponds to a minimal subset of the interface satisfied by stdlib log.Logger
type StdLogger interface {
	Print(v ...interface{})
	Printf(format string, v ...interface{})
}

var Logger StdLogger

func init() {
	// default Logger
	SetLogger(stdlog.New(os.Stderr, "[restful] ", stdlog.LstdFlags|stdlog.Lshortfile))
}

// SetLogger sets the logger for this package
func SetLogger(customLogger StdLogger) {
	Logger = customLogger
}

// Print delegates to the Logger
func Print(v ...interface{}) {
	Logger.Print(v...)
}

// Printf delegates to the Logger
func Printf(format string, v ...interface{}) {
	Logger.Printf(format, v...)
}
