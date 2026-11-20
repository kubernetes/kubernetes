/*
   Copyright The containerd Authors.

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

// Package log provides types and functions related to logging, passing
// loggers through a context, and attaching context to the logger.
//
// # Transitional types
//
// This package contains various types that are aliases for types in [logrus].
// These aliases are intended for transitioning away from hard-coding logrus
// as logging implementation. Consumers of this package are encouraged to use
// the type-aliases from this package instead of directly using their logrus
// equivalent.
//
// The intent is to replace these aliases with locally defined types and
// interfaces once all consumers are no longer directly importing logrus
// types.
//
// IMPORTANT: due to the transitional purpose of this package, it is not
// guaranteed for the full logrus API to be provided in the future. As
// outlined, these aliases are provided as a step to transition away from
// a specific implementation which, as a result, exposes the full logrus API.
// While no decisions have been made on the ultimate design and interface
// provided by this package, we do not expect carrying "less common" features.
package log

import (
	"context"
	"fmt"
	"io"
	"log/slog"

	"github.com/sirupsen/logrus"
	lslog "github.com/sirupsen/logrus/hooks/slog"
)

// G is a shorthand for [GetLogger].
//
// We may want to define this locally to a package to get package tagged log
// messages.
var G = GetLogger

// L is an alias for the standard logger.
var L = &Entry{
	Logger: logrus.StandardLogger(),
	// Default is three fields plus a little extra room.
	Data: make(Fields, 6),
}

type loggerKey struct{}

// Fields type to pass to "WithFields".
type Fields = map[string]any

// Entry is a logging entry. It contains all the fields passed with
// [Entry.WithFields]. It's finally logged when Trace, Debug, Info, Warn,
// Error, Fatal or Panic is called on it. These objects can be reused and
// passed around as much as you wish to avoid field duplication.
//
// Entry is a transitional type, and currently an alias for [logrus.Entry].
type Entry = logrus.Entry

// RFC3339NanoFixed is [time.RFC3339Nano] with nanoseconds padded using
// zeros to ensure the formatted time is always the same number of
// characters.
const RFC3339NanoFixed = "2006-01-02T15:04:05.000000000Z07:00"

// Level is a logging level.
type Level = logrus.Level

// Supported log levels.
const (
	// TraceLevel level. Designates finer-grained informational events
	// than [DebugLevel].
	TraceLevel Level = logrus.TraceLevel

	// DebugLevel level. Usually only enabled when debugging. Very verbose
	// logging.
	DebugLevel Level = logrus.DebugLevel

	// InfoLevel level. General operational entries about what's going on
	// inside the application.
	InfoLevel Level = logrus.InfoLevel

	// WarnLevel level. Non-critical entries that deserve eyes.
	WarnLevel Level = logrus.WarnLevel

	// ErrorLevel level. Logs errors that should definitely be noted.
	// Commonly used for hooks to send errors to an error tracking service.
	ErrorLevel Level = logrus.ErrorLevel

	// FatalLevel level. Logs and then calls "logger.Exit(1)". It exits
	// even if the logging level is set to Panic.
	FatalLevel Level = logrus.FatalLevel

	// PanicLevel level. This is the highest level of severity. Logs and
	// then calls panic with the message passed to Debug, Info, ...
	PanicLevel Level = logrus.PanicLevel
)

// levelValue is a log level accepted by [SetLevel].
type levelValue interface {
	string | Level | slog.Level | lslog.Level | lslog.SlogLevel
}

// SetLevel sets the global log level.
//
// The level may be specified as a string, a [Level], a [slog.Level], an
// [lslog.Level], or an [lslog.SlogLevel].
//
// String levels are parsed using [logrus.ParseLevel] and may be one of:
//
//   - "trace" ([TraceLevel])
//   - "debug" ([DebugLevel])
//   - "info" ([InfoLevel])
//   - "warn" ([WarnLevel])
//   - "error" ([ErrorLevel])
//   - "fatal" ([FatalLevel])
//   - "panic" ([PanicLevel])
//
// slog levels are mapped to Logrus levels using the mapping defined by
// [lslog.SlogLevel]. [lslog.Level] values are accepted directly as their
// underlying Logrus level.
//
// SetLevel returns an error if a string level is not supported.
func SetLevel[T levelValue](level T) error {
	var lvl Level

	switch l := any(level).(type) {
	case string:
		var err error
		lvl, err = logrus.ParseLevel(l)
		if err != nil {
			return err
		}

	case Level:
		lvl = l

	case slog.Level:
		lvl = lslog.SlogLevel(l).Level()

	case lslog.Level:
		lvl = Level(l)

	case lslog.SlogLevel:
		lvl = l.Level()
	}

	L.Logger.SetLevel(lvl)
	return nil
}

// GetLevel returns the current log level.
func GetLevel() Level {
	return L.Logger.GetLevel()
}

// OutputFormat specifies a log output format.
type OutputFormat string

// Supported log output formats.
const (
	// TextFormat represents the text logging format.
	TextFormat OutputFormat = "text"

	// JSONFormat represents the JSON logging format.
	JSONFormat OutputFormat = "json"
)

// SetFormat sets the log output format ([TextFormat] or [JSONFormat]).
func SetFormat(format OutputFormat) error {
	if slogOut != nil {
		var handler slog.Handler
		switch format {
		case TextFormat:
			handler = slog.NewTextHandler(slogOut, &slog.HandlerOptions{Level: loggerLevel{}})
		case JSONFormat:
			handler = slog.NewJSONHandler(slogOut, &slog.HandlerOptions{Level: loggerLevel{}})
		default:
			return fmt.Errorf("unknown log format: %s", format)
		}
		slog.SetDefault(slog.New(handler))

		// Keep logrus formatting and output disabled, as both are handled by slog.
		L.Logger.SetFormatter(discardFormatter{})
		L.Logger.SetOutput(io.Discard)
		return nil
	}

	switch format {
	case TextFormat:
		L.Logger.SetFormatter(&logrus.TextFormatter{
			TimestampFormat: RFC3339NanoFixed,
			FullTimestamp:   true,
		})
	case JSONFormat:
		L.Logger.SetFormatter(&logrus.JSONFormatter{
			TimestampFormat: RFC3339NanoFixed,
		})
	default:
		return fmt.Errorf("unknown log format: %s", format)
	}

	return nil
}

// WithLogger returns a new context with the provided logger. Use in
// combination with logger.WithField(s) for great effect.
func WithLogger(ctx context.Context, logger *Entry) context.Context {
	return context.WithValue(ctx, loggerKey{}, logger.WithContext(ctx))
}

// GetLogger retrieves the current logger from the context. If no logger is
// available, the default logger is returned.
func GetLogger(ctx context.Context) *Entry {
	if logger := ctx.Value(loggerKey{}); logger != nil {
		return logger.(*Entry)
	}
	return L.WithContext(ctx)
}
