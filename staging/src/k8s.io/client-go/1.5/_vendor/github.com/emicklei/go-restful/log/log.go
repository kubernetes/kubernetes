package log

import (
	stdlog "log"
	"os"
)

// Logger corresponds to a minimal subset of the interface satisfied by stdlib log.Logger
type StdLogger interface {
	Print(v ...interface{})
	Printf(format string, v ...interface{})
}

var Logger StdLogger

func init() {
	// default Logger
	SetLogger(stdlog.New(os.Stderr, "[restful] ", stdlog.LstdFlags|stdlog.Lshortfile))
}

func SetLogger(customLogger StdLogger) {
	Logger = customLogger
}

func Print(v ...interface{}) {
	Logger.Print(v...)
}

func Printf(format string, v ...interface{}) {
	Logger.Printf(format, v...)
}
