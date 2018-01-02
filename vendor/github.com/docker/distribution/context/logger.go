package context

import (
	"fmt"

	"github.com/sirupsen/logrus"
	"runtime"
)

// Logger provides a leveled-logging interface.
type Logger interface {
	// standard logger methods
	Print(args ...interface{})
	Printf(format string, args ...interface{})
	Println(args ...interface{})

	Fatal(args ...interface{})
	Fatalf(format string, args ...interface{})
	Fatalln(args ...interface{})

	Panic(args ...interface{})
	Panicf(format string, args ...interface{})
	Panicln(args ...interface{})

	// Leveled methods, from logrus
	Debug(args ...interface{})
	Debugf(format string, args ...interface{})
	Debugln(args ...interface{})

	Error(args ...interface{})
	Errorf(format string, args ...interface{})
	Errorln(args ...interface{})

	Info(args ...interface{})
	Infof(format string, args ...interface{})
	Infoln(args ...interface{})

	Warn(args ...interface{})
	Warnf(format string, args ...interface{})
	Warnln(args ...interface{})
}

// WithLogger creates a new context with provided logger.
func WithLogger(ctx Context, logger Logger) Context {
	return WithValue(ctx, "logger", logger)
}

// GetLoggerWithField returns a logger instance with the specified field key
// and value without affecting the context. Extra specified keys will be
// resolved from the context.
func GetLoggerWithField(ctx Context, key, value interface{}, keys ...interface{}) Logger {
	return getLogrusLogger(ctx, keys...).WithField(fmt.Sprint(key), value)
}

// GetLoggerWithFields returns a logger instance with the specified fields
// without affecting the context. Extra specified keys will be resolved from
// the context.
func GetLoggerWithFields(ctx Context, fields map[interface{}]interface{}, keys ...interface{}) Logger {
	// must convert from interface{} -> interface{} to string -> interface{} for logrus.
	lfields := make(logrus.Fields, len(fields))
	for key, value := range fields {
		lfields[fmt.Sprint(key)] = value
	}

	return getLogrusLogger(ctx, keys...).WithFields(lfields)
}

// GetLogger returns the logger from the current context, if present. If one
// or more keys are provided, they will be resolved on the context and
// included in the logger. While context.Value takes an interface, any key
// argument passed to GetLogger will be passed to fmt.Sprint when expanded as
// a logging key field. If context keys are integer constants, for example,
// its recommended that a String method is implemented.
func GetLogger(ctx Context, keys ...interface{}) Logger {
	return getLogrusLogger(ctx, keys...)
}

// GetLogrusLogger returns the logrus logger for the context. If one more keys
// are provided, they will be resolved on the context and included in the
// logger. Only use this function if specific logrus functionality is
// required.
func getLogrusLogger(ctx Context, keys ...interface{}) *logrus.Entry {
	var logger *logrus.Entry

	// Get a logger, if it is present.
	loggerInterface := ctx.Value("logger")
	if loggerInterface != nil {
		if lgr, ok := loggerInterface.(*logrus.Entry); ok {
			logger = lgr
		}
	}

	if logger == nil {
		fields := logrus.Fields{}

		// Fill in the instance id, if we have it.
		instanceID := ctx.Value("instance.id")
		if instanceID != nil {
			fields["instance.id"] = instanceID
		}

		fields["go.version"] = runtime.Version()
		// If no logger is found, just return the standard logger.
		logger = logrus.StandardLogger().WithFields(fields)
	}

	fields := logrus.Fields{}
	for _, key := range keys {
		v := ctx.Value(key)
		if v != nil {
			fields[fmt.Sprint(key)] = v
		}
	}

	return logger.WithFields(fields)
}
