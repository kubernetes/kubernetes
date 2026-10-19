package slog

import (
	"context"
	"log/slog"
	"maps"
	"runtime"
	"slices"
	"strings"

	"github.com/sirupsen/logrus"
)

// HandlerOptions are options for a [Handler].
// A zero HandlerOptions consists entirely of default values.
type HandlerOptions struct {
	// AddSource causes the handler to include the source code position
	// of the log statement in the Logrus entry.
	AddSource bool

	// LevelMapper maps slog levels to Logrus levels. If nil, the default
	// mapping is used. Set it to customize level mapping, for example to map
	// custom slog levels to specific Logrus levels.
	LevelMapper func(slog.Level) logrus.Level
}

// Handler is a [slog.Handler] that writes records to a [logrus.Logger].
//
// It is intended for bridging libraries or application code that log via slog
// into an existing Logrus logger, for example during a gradual migration.
//
// By default, slog levels are mapped using [SlogLevel].
// Set [HandlerOptions.LevelMapper] to customize the mapping.
//
// Mapping to [logrus.FatalLevel] or [logrus.PanicLevel] preserves the level
// only; handling a record does not exit or panic.
type Handler struct {
	logger *logrus.Logger
	opts   HandlerOptions

	// fields holds attributes from prior WithAttrs calls, already resolved
	// under the group prefix that applied when they were added.
	fields logrus.Fields

	// groups is the current group name stack for attributes added later.
	groups []string
}

var _ slog.Handler = (*Handler)(nil)

// NewHandler creates a [slog.Handler] that writes to the provided
// [logrus.Logger].
//
// The provided logger must not be nil. NewHandler panics if logger is nil.
// If opts is nil, the default options are used.
func NewHandler(logger *logrus.Logger, opts *HandlerOptions) *Handler {
	if logger == nil {
		panic("cannot create handler from nil logger")
	}
	if opts == nil {
		opts = &HandlerOptions{}
	}
	return &Handler{
		logger: logger,
		opts:   *opts,
	}
}

// toLogrusLevel maps a slog level using LevelMapper or the default mapping.
func (h *Handler) toLogrusLevel(level slog.Level) logrus.Level {
	if h.opts.LevelMapper != nil {
		return h.opts.LevelMapper(level)
	}
	return SlogLevel(level).Level()
}

// Enabled reports whether the handler handles records at the given level.
// It maps the slog level to a logrus level and consults the underlying logger.
func (h *Handler) Enabled(_ context.Context, level slog.Level) bool {
	return h.logger.IsLevelEnabled(h.toLogrusLevel(level))
}

// Handle converts the slog record into a logrus entry and logs it.
// Record time, context, message, and attributes (including those from
// [Handler.WithAttrs]/[Handler.WithGroup]) are preserved. Attributes are
// attached as logrus fields; group names are joined with "." as a key prefix
// (similar to [slog.TextHandler]).
func (h *Handler) Handle(ctx context.Context, record slog.Record) error {
	// Stop recursive forwarding before it can cycle back through this logger.
	if hasLogrusLogger(ctx, h.logger) {
		return nil
	}
	level := h.toLogrusLevel(record.Level)
	if !h.logger.IsLevelEnabled(level) {
		return nil
	}

	entry := &logrus.Entry{
		Logger:  h.logger,
		Data:    h.fields,
		Time:    record.Time,
		Context: ctx,
	}

	if h.opts.AddSource && record.PC != 0 {
		// Preserve the caller selected by slog instead of rediscovering it
		// through Logrus's caller stack filtering.
		//
		// Record.PC is a return PC; resolve it with CallersFrames.
		frame, _ := runtime.CallersFrames([]uintptr{record.PC}).Next()
		entry.Caller = &frame
	}

	if n := record.NumAttrs(); n > 0 {
		// Clone before mutating the handler's fields.
		entry.Data = maps.Clone(h.fields)
		if entry.Data == nil {
			entry.Data = make(logrus.Fields, n)
		}
		record.Attrs(func(a slog.Attr) bool {
			appendAttr(entry.Data, h.groups, a)
			return true
		})
	}

	entry.Log(level, record.Message)
	return nil
}

// WithAttrs returns a new Handler whose attributes consist of h's attributes
// followed by attrs.
func (h *Handler) WithAttrs(attrs []slog.Attr) slog.Handler {
	if len(attrs) == 0 {
		return h
	}
	h2 := h.clone()
	h2.fields = maps.Clone(h.fields)
	if h2.fields == nil {
		h2.fields = make(logrus.Fields, len(attrs))
	}
	for _, a := range attrs {
		appendAttr(h2.fields, h.groups, a)
	}
	return h2
}

// WithGroup returns a new Handler with a group appended to the receiver's
// existing groups. All attributes added to the returned handler (via WithAttrs
// or Handle) are nested under the combined group names.
// If name is empty, WithGroup returns h unchanged.
func (h *Handler) WithGroup(name string) slog.Handler {
	if name == "" {
		return h
	}
	h2 := h.clone()
	h2.groups = append(slices.Clip(h.groups), name)
	return h2
}

// clone returns a shallow copy of h. Callers must clone fields or groups
// before modifying them.
func (h *Handler) clone() *Handler {
	return &Handler{
		logger: h.logger,
		opts:   h.opts,
		fields: h.fields,
		groups: h.groups,
	}
}

// appendAttr adds attr to fields, resolving it and applying group prefixes.
// Group attributes are flattened with "."-separated keys.
func appendAttr(fields logrus.Fields, groups []string, attr slog.Attr) {
	attr.Value = attr.Value.Resolve()
	if attr.Equal(slog.Attr{}) {
		return
	}
	if attr.Value.Kind() == slog.KindGroup {
		gs := attr.Value.Group()
		if len(gs) == 0 {
			return
		}
		g := groups
		if attr.Key != "" {
			g = append(slices.Clip(groups), attr.Key)
		}
		for _, a := range gs {
			appendAttr(fields, g, a)
		}
		return
	}

	key := attr.Key
	if len(groups) > 0 {
		key = strings.Join(groups, ".")
		if attr.Key != "" {
			key += "." + attr.Key
		}
	}
	fields[key] = attr.Value.Any()
}
