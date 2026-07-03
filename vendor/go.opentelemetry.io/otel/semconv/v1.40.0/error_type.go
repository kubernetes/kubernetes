// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package semconv // import "go.opentelemetry.io/otel/semconv/v1.40.0"

import (
	"errors"
	"fmt"
	"reflect"

	"go.opentelemetry.io/otel/attribute"
)

// ErrorType returns an [attribute.KeyValue] identifying the error type of err.
//
// If err is nil, the returned attribute has the default value
// [ErrorTypeOther].
//
// If err or one of the errors in its chain has the method
//
//	ErrorType() string
//
// the returned attribute has that method's return value. If multiple errors in
// the chain implement this method, the value from the first match found by
// [errors.As] is used. Otherwise, the returned attribute has a value derived
// from the concrete type of err after unwrapping any wrappers created with
// [fmt.Errorf].
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
		// Fast path: check the top-level error first.
		s = et.ErrorType()
	} else {
		// Fallback: search the error chain for an ErrorType method.
		var et interface{ ErrorType() string }
		if errors.As(err, &et) {
			// Prioritize the ErrorType method if available.
			s = et.ErrorType()
		}
	}
	if s == "" {
		// Fallback to reflection if the ErrorType method is not supported or
		// returns an empty value.

		t := reflect.TypeOf(unwrapFmtWrapped(err))
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

var fmtWrapErrorType = reflect.TypeOf(fmt.Errorf("wrapped: %w", errors.New("err")))

func unwrapFmtWrapped(err error) error {
	for reflect.TypeOf(err) == fmtWrapErrorType {
		u := errors.Unwrap(err)
		if u == nil {
			return err // Should never happen, but avoid returning nil if unwrapping fails.
		}
		err = u
	}
	return err
}
