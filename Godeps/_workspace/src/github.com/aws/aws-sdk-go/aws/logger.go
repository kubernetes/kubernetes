package aws

import (
	"log"
	"os"
)

// A LogLevelType defines the level logging should be performed at. Used to instruct
// the SDK which statements should be logged.
type LogLevelType uint

// LogLevel returns the pointer to a LogLevel. Should be used to workaround
// not being able to take the address of a non-composite literal.
func LogLevel(l LogLevelType) *LogLevelType {
	return &l
}

// Value returns the LogLevel value or the default value LogOff if the LogLevel
// is nil. Safe to use on nil value LogLevelTypes.
func (l *LogLevelType) Value() LogLevelType {
	if l != nil {
		return *l
	}
	return LogOff
}

// Matches returns true if the v LogLevel is enabled by this LogLevel. Should be
// used with logging sub levels. Is safe to use on nil value LogLevelTypes. If
// LogLevel is nill, will default to LogOff comparison.
func (l *LogLevelType) Matches(v LogLevelType) bool {
	c := l.Value()
	return c&v == v
}

// AtLeast returns true if this LogLevel is at least high enough to satisfies v.
// Is safe to use on nil value LogLevelTypes. If LogLevel is nill, will default
// to LogOff comparison.
func (l *LogLevelType) AtLeast(v LogLevelType) bool {
	c := l.Value()
	return c >= v
}

const (
	// LogOff states that no logging should be performed by the SDK. This is the
	// default state of the SDK, and should be use to disable all logging.
	LogOff LogLevelType = iota * 0x1000

	// LogDebug state that debug output should be logged by the SDK. This should
	// be used to inspect request made and responses received.
	LogDebug
)

// Debug Logging Sub Levels
const (
	// LogDebugWithSigning states that the SDK should log request signing and
	// presigning events. This should be used to log the signing details of
	// requests for debugging. Will also enable LogDebug.
	LogDebugWithSigning LogLevelType = LogDebug | (1 << iota)

	// LogDebugWithHTTPBody states the SDK should log HTTP request and response
	// HTTP bodys in addition to the headers and path. This should be used to
	// see the body content of requests and responses made while using the SDK
	// Will also enable LogDebug.
	LogDebugWithHTTPBody

	// LogDebugWithRequestRetries states the SDK should log when service requests will
	// be retried. This should be used to log when you want to log when service
	// requests are being retried. Will also enable LogDebug.
	LogDebugWithRequestRetries

	// LogDebugWithRequestErrors states the SDK should log when service requests fail
	// to build, send, validate, or unmarshal.
	LogDebugWithRequestErrors
)

// A Logger is a minimalistic interface for the SDK to log messages to. Should
// be used to provide custom logging writers for the SDK to use.
type Logger interface {
	Log(...interface{})
}

// NewDefaultLogger returns a Logger which will write log messages to stdout, and
// use same formatting runes as the stdlib log.Logger
func NewDefaultLogger() Logger {
	return &defaultLogger{
		logger: log.New(os.Stdout, "", log.LstdFlags),
	}
}

// A defaultLogger provides a minimalistic logger satisfying the Logger interface.
type defaultLogger struct {
	logger *log.Logger
}

// Log logs the parameters to the stdlib logger. See log.Println.
func (l defaultLogger) Log(args ...interface{}) {
	l.logger.Println(args...)
}
