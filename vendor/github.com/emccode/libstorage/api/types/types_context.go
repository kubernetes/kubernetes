package types

import (
	log "github.com/Sirupsen/logrus"
	"golang.org/x/net/context"
)

// Level is a log level.
type Level log.Level

// These are the different logging levels.
const (
	// PanicLevel level, highest level of severity. Logs and then calls panic
	// with the message passed to Debug, Info, ...
	PanicLevel Level = Level(log.PanicLevel) + iota

	// FatalLevel level. Logs and then calls `os.Exit(1)`. It will exit even
	// if the logging level is set to Panic.
	FatalLevel

	// ErrorLevel level. Logs. Used for errors that should definitely be noted.
	// Commonly used for hooks to send errors to an error tracking service.
	ErrorLevel

	// WarnLevel level. Non-critical entries that deserve eyes.
	WarnLevel

	// InfoLevel level. General operational entries about what's going on
	// inside the application.
	InfoLevel

	// DebugLevel level. Usually only enabled when debugging. Very verbose
	// logging.
	DebugLevel

	// TraceLevel level. An even more verbose levle of logging than DebugLevel.
	TraceLevel
)

// Context is a libStorage context.
type Context interface {
	context.Context
	log.FieldLogger

	// WithValue returns a copy of parent in which the value associated with
	// key is val.
	WithValue(key, value interface{}) Context

	// Join joins this context with another, such that value lookups will first
	// first check the current context, and if no such value exist, a lookup
	// will be performed against the right side.
	Join(ctx context.Context) Context
}

// ContextLoggerFieldAware is used by types that will be logged by the
// Context logger. The key/value pair returned by the type is then emitted
// as part of  the Context's log entry.
type ContextLoggerFieldAware interface {

	// ContextLoggerField is the fields that is logged as part of a Context's
	// log entry.
	ContextLoggerField() (string, interface{})
}

// ContextLoggerFieldsAware is used by types that will be logged by the
// Context logger. The fields returned by the type are then emitted as part of
// the Context's log entry.
type ContextLoggerFieldsAware interface {

	// ContextLoggerFields are the fields that are logged as part of a
	// Context's log entry.
	ContextLoggerFields() map[string]interface{}
}
