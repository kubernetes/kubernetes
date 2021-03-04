/*
Copyright 2019 The logr Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// This design derives from Dave Cheney's blog:
//     http://dave.cheney.net/2015/11/05/lets-talk-about-logging
//
// This is a BETA grade API.  Until there is a significant 2nd implementation,
// I don't really know how it will change.

// Package logr defines abstract interfaces for logging.  Packages can depend on
// these interfaces and callers can implement logging in whatever way is
// appropriate.
//
// Usage
//
// Logging is done using a Logger.  Loggers can have name prefixes and named
// values attached, so that all log messages logged with that Logger have some
// base context associated.
//
// The term "key" is used to refer to the name associated with a particular
// value, to disambiguate it from the general Logger name.
//
// For instance, suppose we're trying to reconcile the state of an object, and
// we want to log that we've made some decision.
//
// With the traditional log package, we might write:
//   log.Printf("decided to set field foo to value %q for object %s/%s",
//       targetValue, object.Namespace, object.Name)
//
// With logr's structured logging, we'd write:
//   // elsewhere in the file, set up the logger to log with the prefix of
//   // "reconcilers", and the named value target-type=Foo, for extra context.
//   log := mainLogger.WithName("reconcilers").WithValues("target-type", "Foo")
//
//   // later on...
//   log.Info("setting foo on object", "value", targetValue, "object", object)
//
// Depending on our logging implementation, we could then make logging decisions
// based on field values (like only logging such events for objects in a certain
// namespace), or copy the structured information into a structured log store.
//
// For logging errors, Logger has a method called Error.  Suppose we wanted to
// log an error while reconciling.  With the traditional log package, we might
// write:
//   log.Errorf("unable to reconcile object %s/%s: %v", object.Namespace, object.Name, err)
//
// With logr, we'd instead write:
//   // assuming the above setup for log
//   log.Error(err, "unable to reconcile object", "object", object)
//
// This functions similarly to:
//   log.Info("unable to reconcile object", "error", err, "object", object)
//
// However, it ensures that a standard key for the error value ("error") is used
// across all error logging.  Furthermore, certain implementations may choose to
// attach additional information (such as stack traces) on calls to Error, so
// it's preferred to use Error to log errors.
//
// Parts of a log line
//
// Each log message from a Logger has four types of context:
// logger name, log verbosity, log message, and the named values.
//
// The Logger name consists of a series of name "segments" added by successive
// calls to WithName.  These name segments will be joined in some way by the
// underlying implementation.  It is strongly recommended that name segments
// contain simple identifiers (letters, digits, and hyphen), and do not contain
// characters that could muddle the log output or confuse the joining operation
// (e.g.  whitespace, commas, periods, slashes, brackets, quotes, etc).
//
// Log verbosity represents how little a log matters.  Level zero, the default,
// matters most.  Increasing levels matter less and less.  Try to avoid lots of
// different verbosity levels, and instead provide useful keys, logger names,
// and log messages for users to filter on.  It's illegal to pass a log level
// below zero.
//
// The log message consists of a constant message attached to the log line.
// This should generally be a simple description of what's occurring, and should
// never be a format string.
//
// Variable information can then be attached using named values (key/value
// pairs).  Keys are arbitrary strings, while values may be any Go value.
//
// Key Naming Conventions
//
// Keys are not strictly required to conform to any specification or regex, but
// it is recommended that they:
//   * be human-readable and meaningful (not auto-generated or simple ordinals)
//   * be constant (not dependent on input data)
//   * contain only printable characters
//   * not contain whitespace or punctuation
//
// These guidelines help ensure that log data is processed properly regardless
// of the log implementation.  For example, log implementations will try to
// output JSON data or will store data for later database (e.g. SQL) queries.
//
// While users are generally free to use key names of their choice, it's
// generally best to avoid using the following keys, as they're frequently used
// by implementations:
//
//   * `"caller"`: the calling information (file/line) of a particular log line.
//   * `"error"`: the underlying error value in the `Error` method.
//   * `"level"`: the log level.
//   * `"logger"`: the name of the associated logger.
//   * `"msg"`: the log message.
//   * `"stacktrace"`: the stack trace associated with a particular log line or
//                     error (often from the `Error` message).
//   * `"ts"`: the timestamp for a log line.
//
// Implementations are encouraged to make use of these keys to represent the
// above concepts, when necessary (for example, in a pure-JSON output form, it
// would be necessary to represent at least message and timestamp as ordinary
// named values).
//
// Implementations may choose to give callers access to the underlying
// logging implementation.  The recommended pattern for this is:
//   // Underlier exposes access to the underlying logging implementation.
//   // Since callers only have a logr.Logger, they have to know which
//   // implementation is in use, so this interface is less of an abstraction
//   // and more of way to test type conversion.
//   type Underlier interface {
//       GetUnderlying() <underlying-type>
//   }
package logr

import (
	"context"
)

// TODO: consider adding back in format strings if they're really needed
// TODO: consider other bits of zap/zapcore functionality like ObjectMarshaller (for arbitrary objects)
// TODO: consider other bits of glog functionality like Flush, OutputStats

// Logger represents the ability to log messages, both errors and not.
type Logger interface {
	// Enabled tests whether this Logger is enabled.  For example, commandline
	// flags might be used to set the logging verbosity and disable some info
	// logs.
	Enabled() bool

	// Info logs a non-error message with the given key/value pairs as context.
	//
	// The msg argument should be used to add some constant description to
	// the log line.  The key/value pairs can then be used to add additional
	// variable information.  The key/value pairs should alternate string
	// keys and arbitrary values.
	Info(msg string, keysAndValues ...interface{})

	// Error logs an error, with the given message and key/value pairs as context.
	// It functions similarly to calling Info with the "error" named value, but may
	// have unique behavior, and should be preferred for logging errors (see the
	// package documentations for more information).
	//
	// The msg field should be used to add context to any underlying error,
	// while the err field should be used to attach the actual error that
	// triggered this log line, if present.
	Error(err error, msg string, keysAndValues ...interface{})

	// V returns an Logger value for a specific verbosity level, relative to
	// this Logger.  In other words, V values are additive.  V higher verbosity
	// level means a log message is less important.  It's illegal to pass a log
	// level less than zero.
	V(level int) Logger

	// WithValues adds some key-value pairs of context to a logger.
	// See Info for documentation on how key/value pairs work.
	WithValues(keysAndValues ...interface{}) Logger

	// WithName adds a new element to the logger's name.
	// Successive calls with WithName continue to append
	// suffixes to the logger's name.  It's strongly recommended
	// that name segments contain only letters, digits, and hyphens
	// (see the package documentation for more information).
	WithName(name string) Logger
}

// InfoLogger provides compatibility with code that relies on the v0.1.0
// interface.
//
// Deprecated: InfoLogger is an artifact of early versions of this API.  New
// users should never use it and existing users should use Logger instead. This
// will be removed in a future release.
type InfoLogger = Logger

type contextKey struct{}

// FromContext returns a Logger constructed from ctx or nil if no
// logger details are found.
func FromContext(ctx context.Context) Logger {
	if v, ok := ctx.Value(contextKey{}).(Logger); ok {
		return v
	}

	return nil
}

// FromContextOrDiscard returns a Logger constructed from ctx or a Logger
// that discards all messages if no logger details are found.
func FromContextOrDiscard(ctx context.Context) Logger {
	if v, ok := ctx.Value(contextKey{}).(Logger); ok {
		return v
	}

	return Discard()
}

// NewContext returns a new context derived from ctx that embeds the Logger.
func NewContext(ctx context.Context, l Logger) context.Context {
	return context.WithValue(ctx, contextKey{}, l)
}

// CallDepthLogger represents a Logger that knows how to climb the call stack
// to identify the original call site and can offset the depth by a specified
// number of frames.  This is useful for users who have helper functions
// between the "real" call site and the actual calls to Logger methods.
// Implementations that log information about the call site (such as file,
// function, or line) would otherwise log information about the intermediate
// helper functions.
//
// This is an optional interface and implementations are not required to
// support it.
type CallDepthLogger interface {
	Logger

	// WithCallDepth returns a Logger that will offset the call stack by the
	// specified number of frames when logging call site information.  If depth
	// is 0 the attribution should be to the direct caller of this method.  If
	// depth is 1 the attribution should skip 1 call frame, and so on.
	// Successive calls to this are additive.
	WithCallDepth(depth int) Logger
}

// WithCallDepth returns a Logger that will offset the call stack by the
// specified number of frames when logging call site information, if possible.
// This is useful for users who have helper functions between the "real" call
// site and the actual calls to Logger methods.  If depth is 0 the attribution
// should be to the direct caller of this function.  If depth is 1 the
// attribution should skip 1 call frame, and so on.  Successive calls to this
// are additive.
//
// If the underlying log implementation supports the CallDepthLogger interface,
// the WithCallDepth method will be called and the result returned.  If the
// implementation does not support CallDepthLogger, the original Logger will be
// returned.
//
// Callers which care about whether this was supported or not should test for
// CallDepthLogger support themselves.
func WithCallDepth(logger Logger, depth int) Logger {
	if decorator, ok := logger.(CallDepthLogger); ok {
		return decorator.WithCallDepth(depth)
	}
	return logger
}
