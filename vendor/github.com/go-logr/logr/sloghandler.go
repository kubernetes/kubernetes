//go:build go1.21
// +build go1.21

/*
Copyright 2023 The logr Authors.

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

package logr

import (
	"context"
	"log/slog"
)

type slogHandler struct {
	// May be nil, in which case all logs get discarded.
	sink LogSink
	// Non-nil if sink is non-nil and implements SlogSink.
	slogSink SlogSink

	// groupPrefix collects values from WithGroup calls. It gets added as
	// prefix to value keys when handling a log record.
	groupPrefix string

	// levelBias can be set when constructing the handler to influence the
	// slog.Level of log records. A positive levelBias reduces the
	// slog.Level value. slog has no API to influence this value after the
	// handler got created, so it can only be set indirectly through
	// Logger.V.
	levelBias slog.Level
}

var _ slog.Handler = &slogHandler{}

// groupSeparator is used to concatenate WithGroup names and attribute keys.
const groupSeparator = "."

// GetLevel is used for black box unit testing.
func (l *slogHandler) GetLevel() slog.Level {
	return l.levelBias
}

func (l *slogHandler) Enabled(_ context.Context, level slog.Level) bool {
	return l.sink != nil && (level >= slog.LevelError || l.sink.Enabled(l.levelFromSlog(level)))
}

func (l *slogHandler) Handle(ctx context.Context, record slog.Record) error {
	if l.slogSink != nil {
		// Only adjust verbosity level of log entries < slog.LevelError.
		if record.Level < slog.LevelError {
			record.Level -= l.levelBias
		}
		return l.slogSink.Handle(ctx, record)
	}

	// No need to check for nil sink here because Handle will only be called
	// when Enabled returned true.

	kvList := make([]any, 0, 2*record.NumAttrs())
	record.Attrs(func(attr slog.Attr) bool {
		kvList = attrToKVs(attr, l.groupPrefix, kvList)
		return true
	})
	if record.Level >= slog.LevelError {
		l.sinkWithCallDepth().Error(nil, record.Message, kvList...)
	} else {
		level := l.levelFromSlog(record.Level)
		l.sinkWithCallDepth().Info(level, record.Message, kvList...)
	}
	return nil
}

// sinkWithCallDepth adjusts the stack unwinding so that when Error or Info
// are called by Handle, code in slog gets skipped.
//
// This offset currently (Go 1.21.0) works for calls through
// slog.New(ToSlogHandler(...)).  There's no guarantee that the call
// chain won't change. Wrapping the handler will also break unwinding. It's
// still better than not adjusting at all....
//
// This cannot be done when constructing the handler because FromSlogHandler needs
// access to the original sink without this adjustment. A second copy would
// work, but then WithAttrs would have to be called for both of them.
func (l *slogHandler) sinkWithCallDepth() LogSink {
	if sink, ok := l.sink.(CallDepthLogSink); ok {
		return sink.WithCallDepth(2)
	}
	return l.sink
}

func (l *slogHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	if l.sink == nil || len(attrs) == 0 {
		return l
	}

	clone := *l
	if l.slogSink != nil {
		clone.slogSink = l.slogSink.WithAttrs(attrs)
		clone.sink = clone.slogSink
	} else {
		kvList := make([]any, 0, 2*len(attrs))
		for _, attr := range attrs {
			kvList = attrToKVs(attr, l.groupPrefix, kvList)
		}
		clone.sink = l.sink.WithValues(kvList...)
	}
	return &clone
}

func (l *slogHandler) WithGroup(name string) slog.Handler {
	if l.sink == nil {
		return l
	}
	if name == "" {
		// slog says to inline empty groups
		return l
	}
	clone := *l
	if l.slogSink != nil {
		clone.slogSink = l.slogSink.WithGroup(name)
		clone.sink = clone.slogSink
	} else {
		clone.groupPrefix = addPrefix(clone.groupPrefix, name)
	}
	return &clone
}

// attrToKVs appends a slog.Attr to a logr-style kvList.  It handle slog Groups
// and other details of slog.
func attrToKVs(attr slog.Attr, groupPrefix string, kvList []any) []any {
	attrVal := attr.Value.Resolve()
	if attrVal.Kind() == slog.KindGroup {
		groupVal := attrVal.Group()
		grpKVs := make([]any, 0, 2*len(groupVal))
		prefix := groupPrefix
		if attr.Key != "" {
			prefix = addPrefix(groupPrefix, attr.Key)
		}
		for _, attr := range groupVal {
			grpKVs = attrToKVs(attr, prefix, grpKVs)
		}
		kvList = append(kvList, grpKVs...)
	} else if attr.Key != "" {
		kvList = append(kvList, addPrefix(groupPrefix, attr.Key), attrVal.Any())
	}

	return kvList
}

func addPrefix(prefix, name string) string {
	if prefix == "" {
		return name
	}
	if name == "" {
		return prefix
	}
	return prefix + groupSeparator + name
}

// levelFromSlog adjusts the level by the logger's verbosity and negates it.
// It ensures that the result is >= 0. This is necessary because the result is
// passed to a LogSink and that API did not historically document whether
// levels could be negative or what that meant.
//
// Some example usage:
//
//	logrV0 := getMyLogger()
//	logrV2 := logrV0.V(2)
//	slogV2 := slog.New(logr.ToSlogHandler(logrV2))
//	slogV2.Debug("msg") // =~ logrV2.V(4) =~ logrV0.V(6)
//	slogV2.Info("msg")  // =~  logrV2.V(0) =~ logrV0.V(2)
//	slogv2.Warn("msg")  // =~ logrV2.V(-4) =~ logrV0.V(0)
func (l *slogHandler) levelFromSlog(level slog.Level) int {
	result := -level
	result += l.levelBias // in case the original Logger had a V level
	if result < 0 {
		result = 0 // because LogSink doesn't expect negative V levels
	}
	return int(result)
}
