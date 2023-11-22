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

package slogr

import (
	"context"
	"log/slog"
	"runtime"
	"time"

	"github.com/go-logr/logr"
)

var (
	_ logr.LogSink          = &slogSink{}
	_ logr.CallDepthLogSink = &slogSink{}
	_ Underlier             = &slogSink{}
)

// Underlier is implemented by the LogSink returned by NewLogr.
type Underlier interface {
	// GetUnderlying returns the Handler used by the LogSink.
	GetUnderlying() slog.Handler
}

const (
	// nameKey is used to log the `WithName` values as an additional attribute.
	nameKey = "logger"

	// errKey is used to log the error parameter of Error as an additional attribute.
	errKey = "err"
)

type slogSink struct {
	callDepth int
	name      string
	handler   slog.Handler
}

func (l *slogSink) Init(info logr.RuntimeInfo) {
	l.callDepth = info.CallDepth
}

func (l *slogSink) GetUnderlying() slog.Handler {
	return l.handler
}

func (l *slogSink) WithCallDepth(depth int) logr.LogSink {
	newLogger := *l
	newLogger.callDepth += depth
	return &newLogger
}

func (l *slogSink) Enabled(level int) bool {
	return l.handler.Enabled(context.Background(), slog.Level(-level))
}

func (l *slogSink) Info(level int, msg string, kvList ...interface{}) {
	l.log(nil, msg, slog.Level(-level), kvList...)
}

func (l *slogSink) Error(err error, msg string, kvList ...interface{}) {
	l.log(err, msg, slog.LevelError, kvList...)
}

func (l *slogSink) log(err error, msg string, level slog.Level, kvList ...interface{}) {
	var pcs [1]uintptr
	// skip runtime.Callers, this function, Info/Error, and all helper functions above that.
	runtime.Callers(3+l.callDepth, pcs[:])

	record := slog.NewRecord(time.Now(), level, msg, pcs[0])
	if l.name != "" {
		record.AddAttrs(slog.String(nameKey, l.name))
	}
	if err != nil {
		record.AddAttrs(slog.Any(errKey, err))
	}
	record.Add(kvList...)
	l.handler.Handle(context.Background(), record)
}

func (l slogSink) WithName(name string) logr.LogSink {
	if l.name != "" {
		l.name = l.name + "/"
	}
	l.name += name
	return &l
}

func (l slogSink) WithValues(kvList ...interface{}) logr.LogSink {
	l.handler = l.handler.WithAttrs(kvListToAttrs(kvList...))
	return &l
}

func kvListToAttrs(kvList ...interface{}) []slog.Attr {
	// We don't need the record itself, only its Add method.
	record := slog.NewRecord(time.Time{}, 0, "", 0)
	record.Add(kvList...)
	attrs := make([]slog.Attr, 0, record.NumAttrs())
	record.Attrs(func(attr slog.Attr) bool {
		attrs = append(attrs, attr)
		return true
	})
	return attrs
}
