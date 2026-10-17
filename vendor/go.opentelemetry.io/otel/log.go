// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otel

import (
	"go.opentelemetry.io/otel/internal/global"
	"go.opentelemetry.io/otel/log"
)

// Logger returns a [log.Logger] configured with the provided name and options
// from the globally configured [log.LoggerProvider].
//
// If this is called before a global LoggerProvider is configured, the returned
// Logger will be a No-Op implementation of a Logger. When a global
// LoggerProvider is registered for the first time, the returned Logger is
// updated in place to report to this new LoggerProvider. There is no need to
// call this function again for an updated instance.
//
// This is a convenience function. It is equivalent to:
//
//	GetLoggerProvider().Logger(name, options...)
func Logger(name string, options ...log.LoggerOption) log.Logger {
	return GetLoggerProvider().Logger(name, options...)
}

// GetLoggerProvider returns the globally configured [log.LoggerProvider].
//
// If a global LoggerProvider has not been configured with [SetLoggerProvider],
// the returned LoggerProvider will be a No-Op implementation. When
// a global LoggerProvider is registered for the first time, the returned
// LoggerProvider and all Loggers it has created are updated in place. There is
// no need to call this function again for an updated instance.
func GetLoggerProvider() log.LoggerProvider {
	return global.LoggerProvider()
}

// SetLoggerProvider configures provider as the global [log.LoggerProvider].
func SetLoggerProvider(provider log.LoggerProvider) {
	global.SetLoggerProvider(provider)
}
