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

// FromSlogHandler returns a Logger which writes to the slog.Handler.
//
// The logr verbosity level is mapped to slog levels such that V(0) becomes
// slog.LevelInfo and V(4) becomes slog.LevelDebug.
func FromSlogHandler(handler slog.Handler) Logger {
	if handler, ok := handler.(*slogHandler); ok {
		if handler.sink == nil {
			return Discard()
		}
		return New(handler.sink).V(int(handler.levelBias))
	}
	return New(&slogSink{handler: handler})
}

// ToSlogHandler returns a slog.Handler which writes to the same sink as the Logger.
//
// The returned logger writes all records with level >= slog.LevelError as
// error log entries with LogSink.Error, regardless of the verbosity level of
// the Logger:
//
//	logger := <some Logger with 0 as verbosity level>
//	slog.New(ToSlogHandler(logger.V(10))).Error(...) -> logSink.Error(...)
//
// The level of all other records gets reduced by the verbosity
// level of the Logger and the result is negated. If it happens
// to be negative, then it gets replaced by zero because a LogSink
// is not expected to handled negative levels:
//
//	slog.New(ToSlogHandler(logger)).Debug(...) -> logger.GetSink().Info(level=4, ...)
//	slog.New(ToSlogHandler(logger)).Warning(...) -> logger.GetSink().Info(level=0, ...)
//	slog.New(ToSlogHandler(logger)).Info(...) -> logger.GetSink().Info(level=0, ...)
//	slog.New(ToSlogHandler(logger.V(4))).Info(...) -> logger.GetSink().Info(level=4, ...)
func ToSlogHandler(logger Logger) slog.Handler {
	if sink, ok := logger.GetSink().(*slogSink); ok && logger.GetV() == 0 {
		return sink.handler
	}

	handler := &slogHandler{sink: logger.GetSink(), levelBias: slog.Level(logger.GetV())}
	if slogSink, ok := handler.sink.(SlogSink); ok {
		handler.slogSink = slogSink
	}
	return handler
}

// SlogSink is an optional interface that a LogSink can implement to support
// logging through the slog.Logger or slog.Handler APIs better. It then should
// also support special slog values like slog.Group. When used as a
// slog.Handler, the advantages are:
//
//   - stack unwinding gets avoided in favor of logging the pre-recorded PC,
//     as intended by slog
//   - proper grouping of key/value pairs via WithGroup
//   - verbosity levels > slog.LevelInfo can be recorded
//   - less overhead
//
// Both APIs (Logger and slog.Logger/Handler) then are supported equally
// well. Developers can pick whatever API suits them better and/or mix
// packages which use either API in the same binary with a common logging
// implementation.
//
// This interface is necessary because the type implementing the LogSink
// interface cannot also implement the slog.Handler interface due to the
// different prototype of the common Enabled method.
//
// An implementation could support both interfaces in two different types, but then
// additional interfaces would be needed to convert between those types in FromSlogHandler
// and ToSlogHandler.
type SlogSink interface {
	LogSink

	Handle(ctx context.Context, record slog.Record) error
	WithAttrs(attrs []slog.Attr) SlogSink
	WithGroup(name string) SlogSink
}
