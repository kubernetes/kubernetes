// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package attribute // import "go.opentelemetry.io/otel/attribute"

import (
	"math"
)

func boolToRaw(b bool) uint64 { // nolint:revive  // b is not a control flag.
	if b {
		return 1
	}
	return 0
}

func rawToBool(r uint64) bool {
	return r != 0
}

func int64ToRaw(i int64) uint64 {
	// Assumes original was a valid int64 (overflow not checked).
	return uint64(i) // nolint: gosec
}

func rawToInt64(r uint64) int64 {
	// Assumes original was a valid int64 (overflow not checked).
	return int64(r) // nolint: gosec
}

func float64ToRaw(f float64) uint64 {
	return math.Float64bits(f)
}

func rawToFloat64(r uint64) float64 {
	return math.Float64frombits(r)
}
