// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package x documents experimental features for [go.opentelemetry.io/otel/sdk].
package x // import "go.opentelemetry.io/otel/sdk/internal/x"

import "strings"

// Resource is an experimental feature flag that defines if resource detectors
// should be included experimental semantic conventions.
//
// To enable this feature set the OTEL_GO_X_RESOURCE environment variable
// to the case-insensitive string value of "true" (i.e. "True" and "TRUE"
// will also enable this).
var Resource = newFeature(
	[]string{"RESOURCE"},
	func(v string) (string, bool) {
		if strings.EqualFold(v, "true") {
			return v, true
		}
		return "", false
	},
)

// Observability is an experimental feature flag that determines if SDK
// observability metrics are enabled.
//
// To enable this feature set the OTEL_GO_X_OBSERVABILITY environment variable
// to the case-insensitive string value of "true" (i.e. "True" and "TRUE"
// will also enable this).
var Observability = newFeature(
	[]string{"OBSERVABILITY", "SELF_OBSERVABILITY"},
	func(v string) (string, bool) {
		if strings.EqualFold(v, "true") {
			return v, true
		}
		return "", false
	},
)
