// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package log

import "go.opentelemetry.io/otel/log/embedded"

// LoggerProvider provides access to [Logger].
//
// Warning: Methods may be added to this interface in minor releases. See
// package documentation on API implementation for information on how to set
// the default behavior for unimplemented methods.
type LoggerProvider interface {
	// Users of the interface can ignore this. This embedded type is only used
	// by implementations of this interface. See the "API Implementations"
	// section of the package documentation for more information.
	embedded.LoggerProvider

	// Logger returns a new [Logger] with the provided name and configuration.
	//
	// The name needs to uniquely identify the source of logged code. It is
	// recommended that name be the Go package name of the library using a log
	// bridge (note: this is not the name of the bridge package). Most
	// commonly, this means a bridge will need to accept this value from its
	// users.
	//
	// An empty name is invalid. Implementations should retain the empty value as the
	// instrumentation scope name, return a working Logger, and report the invalid value.
	//
	// The version of the package using a bridge can be critical information
	// to include when logging. The bridge should accept this version
	// information and use the [WithInstrumentationVersion] option to configure
	// the Logger appropriately.
	//
	// Implementations of this method need to be safe for a user to call
	// concurrently.
	Logger(name string, options ...LoggerOption) Logger
}
