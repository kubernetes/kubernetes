// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package semconv // import "go.opentelemetry.io/otel/semconv/v1.37.0"

import (
	"reflect"

	"go.opentelemetry.io/otel/attribute"
)

// ErrorType returns an [attribute.KeyValue] identifying the error type of err.
//
// If err is nil, the returned attribute has the default value
// [ErrorTypeOther].
//
// If err's type has the method
//
//	ErrorType() string
//
// then the returned attribute has the value of err.ErrorType(). Otherwise, the
// returned attribute has a value derived from the concrete type of err.
//
// The key of the returned attribute is [ErrorTypeKey].
func ErrorType(err error) attribute.KeyValue {
	if err == nil {
		return ErrorTypeOther
	}

	return ErrorTypeKey.String(errorType(err))
}

func errorType(err error) string {
	var s string
	if et, ok := err.(interface{ ErrorType() string }); ok {
		// Prioritize the ErrorType method if available.
		s = et.ErrorType()
	}
	if s == "" {
		// Fallback to reflection if the ErrorType method is not supported or
		// returns an empty value.

		t := reflect.TypeOf(err)
		pkg, name := t.PkgPath(), t.Name()
		if pkg != "" && name != "" {
			s = pkg + "." + name
		} else {
			// The type has no package path or name (predeclared, not-defined,
			// or alias for a not-defined type).
			//
			// This is not guaranteed to be unique, but is a best effort.
			s = t.String()
		}
	}
	return s
}
