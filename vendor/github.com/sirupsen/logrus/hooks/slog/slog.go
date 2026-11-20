// Package slog provides adapters between Logrus and log/slog.
//
// [Handler] forwards slog records to a Logrus logger, while [Hook] forwards
// Logrus entries to a slog logger. They can be used independently to support
// incremental migration between the two logging APIs.
package slog

import (
	"context"
	"log/slog"
	"maps"

	"github.com/sirupsen/logrus"
)

// logrusLoggerContextKey is the context key used to track Logrus loggers a
// record has already traversed while being forwarded through slog adapters.
type logrusLoggerContextKey struct{}

// withLogrusLogger returns a context that records logger as having already
// handled the current log record.
func withLogrusLogger(ctx context.Context, logger *logrus.Logger) context.Context {
	seen, _ := ctx.Value(logrusLoggerContextKey{}).(map[*logrus.Logger]struct{})
	seen = maps.Clone(seen)
	if seen == nil {
		seen = make(map[*logrus.Logger]struct{}, 1)
	}
	seen[logger] = struct{}{}
	return context.WithValue(ctx, logrusLoggerContextKey{}, seen)
}

// hasLogrusLogger reports whether logger has already handled the current log
// record while it was forwarded through slog adapters.
func hasLogrusLogger(ctx context.Context, logger *logrus.Logger) bool {
	seen, _ := ctx.Value(logrusLoggerContextKey{}).(map[*logrus.Logger]struct{})
	_, ok := seen[logger]
	return ok
}

// HookOptions are options for a [Hook].
// A zero HookOptions consists entirely of default values.
type HookOptions struct {
	// LevelMapper maps Logrus levels to slog levels. If nil, the default
	// mapping is used. Set it to customize level mapping, for example to map
	// custom Logrus levels to specific slog levels.
	LevelMapper func(logrus.Level) slog.Level
}

// Hook sends Logrus entries to slog.
//
// It is intended for bridging libraries or application code that log via
// Logrus into an existing slog logger, for example during a gradual migration.
//
// By default, Logrus levels are mapped using [Level]. Set
// [HookOptions.LevelMapper] to customize the mapping.
type Hook struct {
	logger *slog.Logger
	opts   HookOptions
}

var _ logrus.Hook = (*Hook)(nil)

// NewHook creates a [logrus.Hook] that sends logs to the provided [slog.Logger].
//
// This hook is intended to be used during transition from Logrus to slog,
// or as a shim between different parts of your application or different
// libraries that depend on different loggers.
//
// The provided logger must not be nil. NewHook panics if logger is nil.
// If opts is nil, the default options are used.
func NewHook(logger *slog.Logger, opts *HookOptions) *Hook {
	if logger == nil {
		panic("cannot create hook from nil logger")
	}
	if opts == nil {
		opts = &HookOptions{}
	}
	return &Hook{
		logger: logger,
		opts:   *opts,
	}
}

// toSlogLevel maps a Logrus level using LevelMapper or the default mapping.
func (h *Hook) toSlogLevel(level logrus.Level) slog.Level {
	if h.opts.LevelMapper != nil {
		return h.opts.LevelMapper(level)
	}
	return Level(level).Level()
}

// Levels always returns all levels, since slog allows controlling level
// enabling based on context.
func (h *Hook) Levels() []logrus.Level {
	return logrus.AllLevels
}

// Fire forwards the provided logrus Entry to the underlying slog.Logger's
// Handler, mapping it to a slog.Record. Time and caller information are
// preserved when available, and Entry.Data is converted to attributes.
// If Entry.Context is nil, context.Background() is used.
func (h *Hook) Fire(entry *logrus.Entry) error {
	ctx := entry.Context
	if ctx == nil {
		ctx = context.Background()
	}
	// Track this logger to guard against recursive forwarding.
	ctx = withLogrusLogger(ctx, entry.Logger)

	level := h.toSlogLevel(entry.Level)
	handler := h.logger.Handler()
	if !handler.Enabled(ctx, level) {
		return nil
	}
	attrs := make([]slog.Attr, 0, len(entry.Data))
	for k, v := range entry.Data {
		attrs = append(attrs, slog.Any(k, v))
	}
	var pc uintptr
	if entry.Caller != nil {
		pc = entry.Caller.PC
	}
	r := slog.NewRecord(entry.Time, level, entry.Message, pc)
	r.AddAttrs(attrs...)
	return handler.Handle(ctx, r)
}
