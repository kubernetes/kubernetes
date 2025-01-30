// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otel // import "go.opentelemetry.io/otel"

import (
	"github.com/go-logr/logr"

	"go.opentelemetry.io/otel/internal/global"
)

// SetLogger configures the logger used internally to opentelemetry.
func SetLogger(logger logr.Logger) {
	global.SetLogger(logger)
}
