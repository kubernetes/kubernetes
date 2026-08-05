// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package global provides the OpenTelemetry global API.
package global // import "go.opentelemetry.io/otel/internal/global"

import (
	"go.opentelemetry.io/otel/internal/errorhandler"
)

// ErrorHandler is an alias for errorhandler.ErrorHandler, kept for backward
// compatibility with existing callers of internal/global.
type ErrorHandler = errorhandler.ErrorHandler

// ErrDelegator is an alias for errorhandler.ErrDelegator, kept for backward
// compatibility with existing callers of internal/global.
type ErrDelegator = errorhandler.ErrDelegator
