// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package global // import "go.opentelemetry.io/otel/internal/global"

import (
	"log"
	"os"
	"sync/atomic"

	"github.com/go-logr/logr"
	"github.com/go-logr/stdr"
)

// globalLogger holds a reference to the [logr.Logger] used within
// go.opentelemetry.io/otel.
//
// The default logger uses stdr which is backed by the standard `log.Logger`
// interface. This logger will only show messages at the Error Level.
var globalLogger = func() *atomic.Pointer[logr.Logger] {
	l := stdr.New(log.New(os.Stderr, "", log.LstdFlags|log.Lshortfile))

	p := new(atomic.Pointer[logr.Logger])
	p.Store(&l)
	return p
}()

// SetLogger sets the global Logger to l.
//
// To see Warn messages use a logger with `l.V(1).Enabled() == true`
// To see Info messages use a logger with `l.V(4).Enabled() == true`
// To see Debug messages use a logger with `l.V(8).Enabled() == true`.
func SetLogger(l logr.Logger) {
	globalLogger.Store(&l)
}

// GetLogger returns the global logger.
func GetLogger() logr.Logger {
	return *globalLogger.Load()
}

// Info prints messages about the general state of the API or SDK.
// This should usually be less than 5 messages a minute.
func Info(msg string, keysAndValues ...any) {
	GetLogger().V(4).Info(msg, keysAndValues...)
}

// Error prints messages about exceptional states of the API or SDK.
func Error(err error, msg string, keysAndValues ...any) {
	GetLogger().Error(err, msg, keysAndValues...)
}

// Debug prints messages about all internal changes in the API or SDK.
func Debug(msg string, keysAndValues ...any) {
	GetLogger().V(8).Info(msg, keysAndValues...)
}

// Warn prints messages about warnings in the API or SDK.
// Not an error but is likely more important than an informational event.
func Warn(msg string, keysAndValues ...any) {
	GetLogger().V(1).Info(msg, keysAndValues...)
}
