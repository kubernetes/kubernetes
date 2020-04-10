// Package logr defines abstract interfaces for logging.  Packages can depend on
// these interfaces and callers can implement logging in whatever way is
// appropriate.
//
// This design derives from Dave Cheney's blog:
//     http://dave.cheney.net/2015/11/05/lets-talk-about-logging
//
// This is a BETA grade API.  Until there is a significant 2nd implementation,
// I don't really know how it will change.
//
// The logging specifically makes it non-trivial to use format strings, to encourage
// attaching structured information instead of unstructured format strings.
//
// Usage
//
// Logging is done using a Logger.  Loggers can have name prefixes and named values
// attached, so that all log messages logged with that Logger have some base context
// associated.
//
// The term "key" is used to refer to the name associated with a particular value, to
// disambiguate it from the general Logger name.
//
// For instance, suppose we're trying to reconcile the state of an object, and we want
// to log that we've made some decision.
//
// With the traditional log package, we might write
//  log.Printf(
//      "decided to set field foo to value %q for object %s/%s",
//       targetValue, object.Namespace, object.Name)
//
// With logr's structured logging, we'd write
//  // elsewhere in the file, set up the logger to log with the prefix of "reconcilers",
//  // and the named value target-type=Foo, for extra context.
//  log := mainLogger.WithName("reconcilers").WithValues("target-type", "Foo")
//
//  // later on...
//  log.Info("setting field foo on object", "value", targetValue, "object", object)
//
// Depending on our logging implementation, we could then make logging decisions based on field values
// (like only logging such events for objects in a certain namespace), or copy the structured
// information into a structured log store.
//
// For logging errors, Logger has a method called Error.  Suppose we wanted to log an
// error while reconciling.  With the traditional log package, we might write
//   log.Errorf("unable to reconcile object %s/%s: %v", object.Namespace, object.Name, err)
//
// With logr, we'd instead write
//   // assuming the above setup for log
//   log.Error(err, "unable to reconcile object", "object", object)
//
// This functions similarly to:
//   log.Info("unable to reconcile object", "error", err, "object", object)
//
// However, it ensures that a standard key for the error value ("error") is used across all
// error logging.  Furthermore, certain implementations may choose to attach additional
// information (such as stack traces) on calls to Error, so it's preferred to use Error
// to log errors.
//
// Parts of a log line
//
// Each log message from a Logger has four types of context:
// logger name, log verbosity, log message, and the named values.
//
// The Logger name constists of a series of name "segments" added by successive calls to WithName.
// These name segments will be joined in some way by the underlying implementation.  It is strongly
// reccomended that name segements contain simple identifiers (letters, digits, and hyphen), and do
// not contain characters that could muddle the log output or confuse the joining operation (e.g.
// whitespace, commas, periods, slashes, brackets, quotes, etc).
//
// Log verbosity represents how little a log matters.  Level zero, the default, matters most.
// Increasing levels matter less and less.  Try to avoid lots of different verbosity levels,
// and instead provide useful keys, logger names, and log messages for users to filter on.
// It's illegal to pass a log level below zero.
//
// The log message consists of a constant message attached to the the log line.  This
// should generally be a simple description of what's occuring, and should never be a format string.
//
// Variable information can then be attached using named values (key/value pairs).  Keys are arbitrary
// strings, while values may be any Go value.
//
// Key Naming Conventions
//
// While users are generally free to use key names of their choice, it's generally best to avoid
// using the following keys, as they're frequently used by implementations:
//
// - `"error"`: the underlying error value in the `Error` method.
// - `"stacktrace"`: the stack trace associated with a particular log line or error
//                   (often from the `Error` message).
// - `"caller"`: the calling information (file/line) of a particular log line.
// - `"msg"`: the log message.
// - `"level"`: the log level.
// - `"ts"`: the timestamp for a log line.
//
// Implementations are encouraged to make use of these keys to represent the above
// concepts, when neccessary (for example, in a pure-JSON output form, it would be
// necessary to represent at least message and timestamp as ordinary named values).
package logr

// TODO: consider adding back in format strings if they're really needed
// TODO: consider other bits of zap/zapcore functionality like ObjectMarshaller (for arbitrary objects)
// TODO: consider other bits of glog functionality like Flush, InfoDepth, OutputStats

// InfoLogger represents the ability to log non-error messages, at a particular verbosity.
type InfoLogger interface {
	// Info logs a non-error message with the given key/value pairs as context.
	//
	// The msg argument should be used to add some constant description to
	// the log line.  The key/value pairs can then be used to add additional
	// variable information.  The key/value pairs should alternate string
	// keys and arbitrary values.
	Info(msg string, keysAndValues ...interface{})

	// Enabled tests whether this InfoLogger is enabled.  For example,
	// commandline flags might be used to set the logging verbosity and disable
	// some info logs.
	Enabled() bool
}

// Logger represents the ability to log messages, both errors and not.
type Logger interface {
	// All Loggers implement InfoLogger.  Calling InfoLogger methods directly on
	// a Logger value is equivalent to calling them on a V(0) InfoLogger.  For
	// example, logger.Info() produces the same result as logger.V(0).Info.
	InfoLogger

	// Error logs an error, with the given message and key/value pairs as context.
	// It functions similarly to calling Info with the "error" named value, but may
	// have unique behavior, and should be preferred for logging errors (see the
	// package documentations for more information).
	//
	// The msg field should be used to add context to any underlying error,
	// while the err field should be used to attach the actual error that
	// triggered this log line, if present.
	Error(err error, msg string, keysAndValues ...interface{})

	// V returns an InfoLogger value for a specific verbosity level.  A higher
	// verbosity level means a log message is less important.  It's illegal to
	// pass a log level less than zero.
	V(level int) InfoLogger

	// WithValues adds some key-value pairs of context to a logger.
	// See Info for documentation on how key/value pairs work.
	WithValues(keysAndValues ...interface{}) Logger

	// WithName adds a new element to the logger's name.
	// Successive calls with WithName continue to append
	// suffixes to the logger's name.  It's strongly reccomended
	// that name segments contain only letters, digits, and hyphens
	// (see the package documentation for more information).
	WithName(name string) Logger
}
