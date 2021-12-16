package utils

import (
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

// LogLevel of quic-go
type LogLevel uint8

const (
	// LogLevelNothing disables
	LogLevelNothing LogLevel = iota
	// LogLevelError enables err logs
	LogLevelError
	// LogLevelInfo enables info logs (e.g. packets)
	LogLevelInfo
	// LogLevelDebug enables debug logs (e.g. packet contents)
	LogLevelDebug
)

const logEnv = "QUIC_GO_LOG_LEVEL"

// A Logger logs.
type Logger interface {
	SetLogLevel(LogLevel)
	SetLogTimeFormat(format string)
	WithPrefix(prefix string) Logger
	Debug() bool

	Errorf(format string, args ...interface{})
	Infof(format string, args ...interface{})
	Debugf(format string, args ...interface{})
}

// DefaultLogger is used by quic-go for logging.
var DefaultLogger Logger

type defaultLogger struct {
	prefix string

	logLevel   LogLevel
	timeFormat string
}

var _ Logger = &defaultLogger{}

// SetLogLevel sets the log level
func (l *defaultLogger) SetLogLevel(level LogLevel) {
	l.logLevel = level
}

// SetLogTimeFormat sets the format of the timestamp
// an empty string disables the logging of timestamps
func (l *defaultLogger) SetLogTimeFormat(format string) {
	log.SetFlags(0) // disable timestamp logging done by the log package
	l.timeFormat = format
}

// Debugf logs something
func (l *defaultLogger) Debugf(format string, args ...interface{}) {
	if l.logLevel == LogLevelDebug {
		l.logMessage(format, args...)
	}
}

// Infof logs something
func (l *defaultLogger) Infof(format string, args ...interface{}) {
	if l.logLevel >= LogLevelInfo {
		l.logMessage(format, args...)
	}
}

// Errorf logs something
func (l *defaultLogger) Errorf(format string, args ...interface{}) {
	if l.logLevel >= LogLevelError {
		l.logMessage(format, args...)
	}
}

func (l *defaultLogger) logMessage(format string, args ...interface{}) {
	var pre string

	if len(l.timeFormat) > 0 {
		pre = time.Now().Format(l.timeFormat) + " "
	}
	if len(l.prefix) > 0 {
		pre += l.prefix + " "
	}
	log.Printf(pre+format, args...)
}

func (l *defaultLogger) WithPrefix(prefix string) Logger {
	if len(l.prefix) > 0 {
		prefix = l.prefix + " " + prefix
	}
	return &defaultLogger{
		logLevel:   l.logLevel,
		timeFormat: l.timeFormat,
		prefix:     prefix,
	}
}

// Debug returns true if the log level is LogLevelDebug
func (l *defaultLogger) Debug() bool {
	return l.logLevel == LogLevelDebug
}

func init() {
	DefaultLogger = &defaultLogger{}
	DefaultLogger.SetLogLevel(readLoggingEnv())
}

func readLoggingEnv() LogLevel {
	switch strings.ToLower(os.Getenv(logEnv)) {
	case "":
		return LogLevelNothing
	case "debug":
		return LogLevelDebug
	case "info":
		return LogLevelInfo
	case "error":
		return LogLevelError
	default:
		fmt.Fprintln(os.Stderr, "invalid quic-go log level, see https://github.com/lucas-clemente/quic-go/wiki/Logging")
		return LogLevelNothing
	}
}
