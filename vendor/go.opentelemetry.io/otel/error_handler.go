// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otel // import "go.opentelemetry.io/otel"

// ErrorHandler handles irremediable events.
type ErrorHandler interface {
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Handle handles any error deemed irremediable by an OpenTelemetry
	// component.
	Handle(error)
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.
}

// ErrorHandlerFunc is a convenience adapter to allow the use of a function
// as an ErrorHandler.
type ErrorHandlerFunc func(error)

var _ ErrorHandler = ErrorHandlerFunc(nil)

// Handle handles the irremediable error by calling the ErrorHandlerFunc itself.
func (f ErrorHandlerFunc) Handle(err error) {
	f(err)
}
