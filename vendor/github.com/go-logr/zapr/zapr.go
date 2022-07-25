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

// Copyright 2018 Solly Ross
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

// Package zapr defines an implementation of the github.com/go-logr/logr
// interfaces built on top of Zap (go.uber.org/zap).
//
// Usage
//
// A new logr.Logger can be constructed from an existing zap.Logger using
// the NewLogger function:
//
//  log := zapr.NewLogger(someZapLogger)
//
// Implementation Details
//
// For the most part, concepts in Zap correspond directly with those in
// logr.
//
// Unlike Zap, all fields *must* be in the form of sugared fields --
// it's illegal to pass a strongly-typed Zap field in a key position
// to any of the log methods.
//
// Levels in logr correspond to custom debug levels in Zap.  Any given level
// in logr is represents by its inverse in zap (`zapLevel = -1*logrLevel`).
// For example V(2) is equivalent to log level -2 in Zap, while V(1) is
// equivalent to Zap's DebugLevel.
package zapr

import (
	"fmt"

	"github.com/go-logr/logr"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// NB: right now, we always use the equivalent of sugared logging.
// This is necessary, since logr doesn't define non-suggared types,
// and using zap-specific non-suggared types would make uses tied
// directly to Zap.

// zapLogger is a logr.Logger that uses Zap to log.  The level has already been
// converted to a Zap level, which is to say that `logrLevel = -1*zapLevel`.
type zapLogger struct {
	// NB: this looks very similar to zap.SugaredLogger, but
	// deals with our desire to have multiple verbosity levels.
	l *zap.Logger

	// numericLevelKey controls whether the numeric logr level is
	// added to each Info log message and with which key.
	numericLevelKey string

	// errorKey is the field name used for the error in
	// Logger.Error calls.
	errorKey string

	// allowZapFields enables logging of strongly-typed Zap
	// fields. It is off by default because it breaks
	// implementation agnosticism.
	allowZapFields bool

	// panicMessages enables log messages for invalid log calls
	// that explain why a call was invalid (for example,
	// non-string key). This is enabled by default.
	panicMessages bool
}

const (
	// noLevel tells handleFields to not inject a numeric log level field.
	noLevel = -1
)

// handleFields converts a bunch of arbitrary key-value pairs into Zap fields.  It takes
// additional pre-converted Zap fields, for use with automatically attached fields, like
// `error`.
func (zl *zapLogger) handleFields(lvl int, args []interface{}, additional ...zap.Field) []zap.Field {
	injectNumericLevel := zl.numericLevelKey != "" && lvl != noLevel

	// a slightly modified version of zap.SugaredLogger.sweetenFields
	if len(args) == 0 {
		// fast-return if we have no suggared fields and no "v" field.
		if !injectNumericLevel {
			return additional
		}
		// Slightly slower fast path when we need to inject "v".
		return append(additional, zap.Int(zl.numericLevelKey, lvl))
	}

	// unlike Zap, we can be pretty sure users aren't passing structured
	// fields (since logr has no concept of that), so guess that we need a
	// little less space.
	numFields := len(args)/2 + len(additional)
	if injectNumericLevel {
		numFields++
	}
	fields := make([]zap.Field, 0, numFields)
	if injectNumericLevel {
		fields = append(fields, zap.Int(zl.numericLevelKey, lvl))
	}
	for i := 0; i < len(args); {
		// Check just in case for strongly-typed Zap fields,
		// which might be illegal (since it breaks
		// implementation agnosticism). If disabled, we can
		// give a better error message.
		if field, ok := args[i].(zap.Field); ok {
			if zl.allowZapFields {
				fields = append(fields, field)
				i++
				continue
			}
			if zl.panicMessages {
				zl.l.WithOptions(zap.AddCallerSkip(1)).DPanic("strongly-typed Zap Field passed to logr", zapIt("zap field", args[i]))
			}
			break
		}

		// make sure this isn't a mismatched key
		if i == len(args)-1 {
			if zl.panicMessages {
				zl.l.WithOptions(zap.AddCallerSkip(1)).DPanic("odd number of arguments passed as key-value pairs for logging", zapIt("ignored key", args[i]))
			}
			break
		}

		// process a key-value pair,
		// ensuring that the key is a string
		key, val := args[i], args[i+1]
		keyStr, isString := key.(string)
		if !isString {
			// if the key isn't a string, DPanic and stop logging
			if zl.panicMessages {
				zl.l.WithOptions(zap.AddCallerSkip(1)).DPanic("non-string key argument passed to logging, ignoring all later arguments", zapIt("invalid key", key))
			}
			break
		}

		fields = append(fields, zapIt(keyStr, val))
		i += 2
	}

	return append(fields, additional...)
}

func zapIt(field string, val interface{}) zap.Field {
	// Handle types that implement logr.Marshaler: log the replacement
	// object instead of the original one.
	if marshaler, ok := val.(logr.Marshaler); ok {
		field, val = invokeMarshaler(field, marshaler)
	}
	return zap.Any(field, val)
}

func invokeMarshaler(field string, m logr.Marshaler) (f string, ret interface{}) {
	defer func() {
		if r := recover(); r != nil {
			ret = fmt.Sprintf("PANIC=%s", r)
			f = field + "Error"
		}
	}()
	return field, m.MarshalLog()
}

func (zl *zapLogger) Init(ri logr.RuntimeInfo) {
	zl.l = zl.l.WithOptions(zap.AddCallerSkip(ri.CallDepth))
}

// Zap levels are int8 - make sure we stay in bounds.  logr itself should
// ensure we never get negative values.
func toZapLevel(lvl int) zapcore.Level {
	if lvl > 127 {
		lvl = 127
	}
	// zap levels are inverted.
	return 0 - zapcore.Level(lvl)
}

func (zl zapLogger) Enabled(lvl int) bool {
	return zl.l.Core().Enabled(toZapLevel(lvl))
}

func (zl *zapLogger) Info(lvl int, msg string, keysAndVals ...interface{}) {
	if checkedEntry := zl.l.Check(toZapLevel(lvl), msg); checkedEntry != nil {
		checkedEntry.Write(zl.handleFields(lvl, keysAndVals)...)
	}
}

func (zl *zapLogger) Error(err error, msg string, keysAndVals ...interface{}) {
	if checkedEntry := zl.l.Check(zap.ErrorLevel, msg); checkedEntry != nil {
		checkedEntry.Write(zl.handleFields(noLevel, keysAndVals, zap.NamedError(zl.errorKey, err))...)
	}
}

func (zl *zapLogger) WithValues(keysAndValues ...interface{}) logr.LogSink {
	newLogger := *zl
	newLogger.l = zl.l.With(zl.handleFields(noLevel, keysAndValues)...)
	return &newLogger
}

func (zl *zapLogger) WithName(name string) logr.LogSink {
	newLogger := *zl
	newLogger.l = zl.l.Named(name)
	return &newLogger
}

func (zl *zapLogger) WithCallDepth(depth int) logr.LogSink {
	newLogger := *zl
	newLogger.l = zl.l.WithOptions(zap.AddCallerSkip(depth))
	return &newLogger
}

// Underlier exposes access to the underlying logging implementation.  Since
// callers only have a logr.Logger, they have to know which implementation is
// in use, so this interface is less of an abstraction and more of way to test
// type conversion.
type Underlier interface {
	GetUnderlying() *zap.Logger
}

func (zl *zapLogger) GetUnderlying() *zap.Logger {
	return zl.l
}

// NewLogger creates a new logr.Logger using the given Zap Logger to log.
func NewLogger(l *zap.Logger) logr.Logger {
	return NewLoggerWithOptions(l)
}

// NewLoggerWithOptions creates a new logr.Logger using the given Zap Logger to
// log and applies additional options.
func NewLoggerWithOptions(l *zap.Logger, opts ...Option) logr.Logger {
	// creates a new logger skipping one level of callstack
	log := l.WithOptions(zap.AddCallerSkip(1))
	zl := &zapLogger{
		l: log,
	}
	zl.errorKey = "error"
	zl.panicMessages = true
	for _, option := range opts {
		option(zl)
	}
	return logr.New(zl)
}

// Option is one additional parameter for NewLoggerWithOptions.
type Option func(*zapLogger)

// LogInfoLevel controls whether a numeric log level is added to
// Info log message. The empty string disables this, a non-empty
// string is the key for the additional field. Errors and
// internal panic messages do not have a log level and thus
// are always logged without this extra field.
func LogInfoLevel(key string) Option {
	return func(zl *zapLogger) {
		zl.numericLevelKey = key
	}
}

// ErrorKey replaces the default "error" field name used for the error
// in Logger.Error calls.
func ErrorKey(key string) Option {
	return func(zl *zapLogger) {
		zl.errorKey = key
	}
}

// AllowZapFields controls whether strongly-typed Zap fields may
// be passed instead of a key/value pair. This is disabled by
// default because it breaks implementation agnosticism.
func AllowZapFields(allowed bool) Option {
	return func(zl *zapLogger) {
		zl.allowZapFields = allowed
	}
}

// DPanicOnBugs controls whether extra log messages are emitted for
// invalid log calls with zap's DPanic method. Depending on the
// configuration of the zap logger, the program then panics after
// emitting the log message which is useful in development because
// such invalid log calls are bugs in the program. The log messages
// explain why a call was invalid (for example, non-string
// key). Emitting them is enabled by default.
func DPanicOnBugs(enabled bool) Option {
	return func(zl *zapLogger) {
		zl.panicMessages = enabled
	}
}

var _ logr.LogSink = &zapLogger{}
var _ logr.CallDepthLogSink = &zapLogger{}
