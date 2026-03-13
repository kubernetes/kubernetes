//go:build !go1.21
// +build !go1.21

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
)

// FromContext returns a Logger from ctx or an error if no Logger is found.
func FromContext(ctx context.Context) (Logger, error) {
	if v, ok := ctx.Value(contextKey{}).(Logger); ok {
		return v, nil
	}

	return Logger{}, notFoundError{}
}

// FromContextOrDiscard returns a Logger from ctx.  If no Logger is found, this
// returns a Logger that discards all log messages.
func FromContextOrDiscard(ctx context.Context) Logger {
	if v, ok := ctx.Value(contextKey{}).(Logger); ok {
		return v
	}

	return Discard()
}

// NewContext returns a new Context, derived from ctx, which carries the
// provided Logger.
func NewContext(ctx context.Context, logger Logger) context.Context {
	return context.WithValue(ctx, contextKey{}, logger)
}
