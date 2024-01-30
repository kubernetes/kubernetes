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

// Package slogr enables usage of a slog.Handler with logr.Logger as front-end
// API and of a logr.LogSink through the slog.Handler and thus slog.Logger
// APIs.
//
// See the README in the top-level [./logr] package for a discussion of
// interoperability.
//
// Deprecated: use the main logr package instead.
package slogr

import (
	"log/slog"

	"github.com/go-logr/logr"
)

// NewLogr returns a logr.Logger which writes to the slog.Handler.
//
// Deprecated: use [logr.FromSlogHandler] instead.
func NewLogr(handler slog.Handler) logr.Logger {
	return logr.FromSlogHandler(handler)
}

// NewSlogHandler returns a slog.Handler which writes to the same sink as the logr.Logger.
//
// Deprecated: use [logr.ToSlogHandler] instead.
func NewSlogHandler(logger logr.Logger) slog.Handler {
	return logr.ToSlogHandler(logger)
}

// ToSlogHandler returns a slog.Handler which writes to the same sink as the logr.Logger.
//
// Deprecated: use [logr.ToSlogHandler] instead.
func ToSlogHandler(logger logr.Logger) slog.Handler {
	return logr.ToSlogHandler(logger)
}

// SlogSink is an optional interface that a LogSink can implement to support
// logging through the slog.Logger or slog.Handler APIs better.
//
// Deprecated: use [logr.SlogSink] instead.
type SlogSink = logr.SlogSink
