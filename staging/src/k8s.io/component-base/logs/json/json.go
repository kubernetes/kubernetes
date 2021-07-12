/*
Copyright 2020 The Kubernetes Authors.

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

package logs

import (
	"os"
	"time"

	"github.com/go-logr/logr"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// Inspired from https://github.com/go-logr/zapr, some functions is copy from the repo.

var (
	// JSONLogger is global json log format logr
	JSONLogger logr.Logger

	// timeNow stubbed out for testing
	timeNow = time.Now
)

func init() {
	JSONLogger = NewJSONLogger(zapcore.Lock(os.Stdout))
}

// NewJSONLogger creates a new json logr.Logger using the given Zap Logger to log.
func NewJSONLogger(w zapcore.WriteSyncer) logr.Logger {
	encoder := zapcore.NewJSONEncoder(encoderConfig)
	core := zapcore.NewCore(encoder, zapcore.AddSync(w), zapcore.DebugLevel)
	l := zap.New(core, zap.WithCaller(true), zap.AddCallerSkip(1))
	return &zapLogger{l: l}
}

var encoderConfig = zapcore.EncoderConfig{
	MessageKey: "msg",

	CallerKey:      "caller",
	TimeKey:        "ts",
	EncodeTime:     epochMillisTimeEncoder,
	EncodeDuration: zapcore.StringDurationEncoder,
	EncodeCaller:   zapcore.ShortCallerEncoder,
}

// this has the same implementation as zapcore.EpochMillisTimeEncoder but
// uses timeNow() which is stubbed out for testing purposes.
func epochMillisTimeEncoder(_ time.Time, enc zapcore.PrimitiveArrayEncoder) {
	nanos := timeNow().UnixNano()
	millis := float64(nanos) / float64(time.Millisecond)
	enc.AppendFloat64(millis)
}

// zapLogger is a logr.Logger that uses Zap to record log.
type zapLogger struct {
	// NB: this looks very similar to zap.SugaredLogger, but
	// deals with our desire to have multiple verbosity levels.
	l   *zap.Logger
	lvl int
}

// zapLogger implement logr.Logger
var _ logr.Logger = &zapLogger{}

// Enabled should always return true
func (l *zapLogger) Enabled() bool {
	return true
}

// Info write message to error level log
func (l *zapLogger) Info(msg string, keysAndVals ...interface{}) {
	if checkedEntry := l.l.Check(zapcore.InfoLevel, msg); checkedEntry != nil {
		checkedEntry.Write(l.handleFields(keysAndVals)...)
	}
}

// Error write log message to error level
func (l *zapLogger) Error(err error, msg string, keysAndVals ...interface{}) {
	if checkedEntry := l.l.Check(zap.ErrorLevel, msg); checkedEntry != nil {
		checkedEntry.Write(l.handleFields(keysAndVals, handleError(err))...)
	}
}

// dPanic write message to DPanicLevel level log we need implement this because we need
// to have the "v" field as well.
func (l *zapLogger) dPanic(msg string) {
	if checkedEntry := l.l.Check(zapcore.DPanicLevel, msg); checkedEntry != nil {
		checkedEntry.Write(zap.Int("v", l.lvl))
	}
}

// handleFields converts a bunch of arbitrary key-value pairs into Zap fields.  It takes
// additional pre-converted Zap fields, for use with automatically attached fields, like
// `error`.
func (l *zapLogger) handleFields(args []interface{}, additional ...zap.Field) []zap.Field {
	// a slightly modified version of zap.SugaredLogger.sweetenFields
	if len(args) == 0 {
		// fast-return if we have no suggared fields.
		return append(additional, zap.Int("v", l.lvl))
	}

	// unlike Zap, we can be pretty sure users aren't passing structured
	// fields (since logr has no concept of that), so guess that we need a
	// little less space.
	fields := make([]zap.Field, 0, len(args)/2+len(additional)+1)
	fields = append(fields, zap.Int("v", l.lvl))
	for i := 0; i < len(args)-1; i += 2 {
		// check just in case for strongly-typed Zap fields, which is illegal (since
		// it breaks implementation agnosticism), so we can give a better error message.
		if _, ok := args[i].(zap.Field); ok {
			l.dPanic("strongly-typed Zap Field passed to logr")
			break
		}

		// process a key-value pair,
		// ensuring that the key is a string
		key, val := args[i], args[i+1]
		keyStr, isString := key.(string)
		if !isString {
			// if the key isn't a string, stop logging
			l.dPanic("non-string key argument passed to logging, ignoring all later arguments")
			break
		}

		fields = append(fields, zap.Any(keyStr, val))
	}

	return append(fields, additional...)
}

func handleError(err error) zap.Field {
	return zap.NamedError("err", err)
}

// V return info logr.Logger  with specified level
func (l *zapLogger) V(level int) logr.Logger {
	return &zapLogger{
		lvl: l.lvl + level,
		l:   l.l,
	}
}

// WithValues return logr.Logger with some keys And Values
func (l *zapLogger) WithValues(keysAndValues ...interface{}) logr.Logger {
	l.l = l.l.With(l.handleFields(keysAndValues)...)
	return l
}

// WithName return logger Named with specified name
func (l *zapLogger) WithName(name string) logr.Logger {
	l.l = l.l.Named(name)
	return l
}

func (l *zapLogger) WithCallDepth(depth int) logr.Logger {
	return l.newLoggerWithExtraSkip(depth)
}

// newLoggerWithExtraSkip allows creation of loggers with variable levels of callstack skipping
func (l *zapLogger) newLoggerWithExtraSkip(callerSkip int) logr.Logger {
	log := l.l.WithOptions(zap.AddCallerSkip(callerSkip))
	return &zapLogger{
		l:   log,
		lvl: l.lvl,
	}
}

var _ logr.CallDepthLogger = &zapLogger{}
