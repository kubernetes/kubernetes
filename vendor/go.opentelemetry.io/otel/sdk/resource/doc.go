// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package resource provides detecting and representing resources.
//
// The fundamental struct is a Resource which holds identifying information
// about the entities for which telemetry is exported.
//
// To automatically construct Resources from an environment a Detector
// interface is defined. Implementations of this interface can be passed to
// the Detect function to generate a Resource from the merged information.
//
// To load a user defined Resource from the environment variable
// OTEL_RESOURCE_ATTRIBUTES the FromEnv Detector can be used. It will interpret
// the value as a list of comma delimited key/value pairs
// (e.g. `<key1>=<value1>,<key2>=<value2>,...`).
//
// While this package provides a stable API,
// the attributes added by resource detectors may change.
package resource // import "go.opentelemetry.io/otel/sdk/resource"
