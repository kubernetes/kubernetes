// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package semconv // import "go.opentelemetry.io/otel/semconv/v1.37.0"

import (
	"fmt"
	"reflect"

	"go.opentelemetry.io/otel/attribute"
)

// ErrorType returns an [attribute.KeyValue] identifying the error type of err.
func ErrorType(err error) attribute.KeyValue {
	if err == nil {
		return ErrorTypeOther
	}
	t := reflect.TypeOf(err)
	var value string
	if t.PkgPath() == "" && t.Name() == "" {
		// Likely a builtin type.
		value = t.String()
	} else {
		value = fmt.Sprintf("%s.%s", t.PkgPath(), t.Name())
	}

	if value == "" {
		return ErrorTypeOther
	}
	return ErrorTypeKey.String(value)
}
