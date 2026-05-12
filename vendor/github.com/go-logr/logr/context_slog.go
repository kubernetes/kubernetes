//go:build go1.21
// +build go1.21

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

package logr

import (
	"context"
	"fmt"
	"log/slog"
)

// FromContext returns a Logger from ctx or an error if no Logger is found.
func FromContext(ctx context.Context) (Logger, error) {
	v := ctx.Value(contextKey{})
	if v == nil {
		return Logger{}, notFoundError{}
	}

	switch v := v.(type) {
	case Logger:
		return v, nil
	case *slog.Logger:
		return FromSlogHandler(v.Handler()), nil
	default:
		// Not reached.
		panic(fmt.Sprintf("unexpected value type for logr context key: %T", v))
	}
}

// FromContextAsSlogLogger returns a slog.Logger from ctx or nil if no such Logger is found.
func FromContextAsSlogLogger(ctx context.Context) *slog.Logger {
	v := ctx.Value(contextKey{})
	if v == nil {
		return nil
	}

	switch v := v.(type) {
	case Logger:
		return slog.New(ToSlogHandler(v))
	case *slog.Logger:
		return v
	default:
		// Not reached.
		panic(fmt.Sprintf("unexpected value type for logr context key: %T", v))
	}
}

// FromContextOrDiscard returns a Logger from ctx.  If no Logger is found, this
// returns a Logger that discards all log messages.
func FromContextOrDiscard(ctx context.Context) Logger {
	if logger, err := FromContext(ctx); err == nil {
		return logger
	}
	return Discard()
}

// NewContext returns a new Context, derived from ctx, which carries the
// provided Logger.
func NewContext(ctx context.Context, logger Logger) context.Context {
	return context.WithValue(ctx, contextKey{}, logger)
}

// NewContextWithSlogLogger returns a new Context, derived from ctx, which carries the
// provided slog.Logger.
func NewContextWithSlogLogger(ctx context.Context, logger *slog.Logger) context.Context {
	return context.WithValue(ctx, contextKey{}, logger)
}
