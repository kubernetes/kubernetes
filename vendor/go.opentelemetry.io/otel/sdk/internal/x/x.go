// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package x contains support for OTel SDK experimental features.
//
// This package should only be used for features defined in the specification.
// It should not be used for experiments or new project ideas.
package x // import "go.opentelemetry.io/otel/sdk/internal/x"

import (
	"os"
	"strings"
)

// Resource is an experimental feature flag that defines if resource detectors
// should be included experimental semantic conventions.
//
// To enable this feature set the OTEL_GO_X_RESOURCE environment variable
// to the case-insensitive string value of "true" (i.e. "True" and "TRUE"
// will also enable this).
var Resource = newFeature("RESOURCE", func(v string) (string, bool) {
	if strings.ToLower(v) == "true" {
		return v, true
	}
	return "", false
})

// Feature is an experimental feature control flag. It provides a uniform way
// to interact with these feature flags and parse their values.
type Feature[T any] struct {
	key   string
	parse func(v string) (T, bool)
}

func newFeature[T any](suffix string, parse func(string) (T, bool)) Feature[T] {
	const envKeyRoot = "OTEL_GO_X_"
	return Feature[T]{
		key:   envKeyRoot + suffix,
		parse: parse,
	}
}

// Key returns the environment variable key that needs to be set to enable the
// feature.
func (f Feature[T]) Key() string { return f.key }

// Lookup returns the user configured value for the feature and true if the
// user has enabled the feature. Otherwise, if the feature is not enabled, a
// zero-value and false are returned.
func (f Feature[T]) Lookup() (v T, ok bool) {
	// https://github.com/open-telemetry/opentelemetry-specification/blob/62effed618589a0bec416a87e559c0a9d96289bb/specification/configuration/sdk-environment-variables.md#parsing-empty-value
	//
	// > The SDK MUST interpret an empty value of an environment variable the
	// > same way as when the variable is unset.
	vRaw := os.Getenv(f.key)
	if vRaw == "" {
		return v, ok
	}
	return f.parse(vRaw)
}

// Enabled returns if the feature is enabled.
func (f Feature[T]) Enabled() bool {
	_, ok := f.Lookup()
	return ok
}
