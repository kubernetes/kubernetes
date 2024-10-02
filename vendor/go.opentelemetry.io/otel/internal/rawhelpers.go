// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package internal // import "go.opentelemetry.io/otel/internal"

import (
	"math"
	"unsafe"
)

func BoolToRaw(b bool) uint64 { // nolint:revive  // b is not a control flag.
	if b {
		return 1
	}
	return 0
}

func RawToBool(r uint64) bool {
	return r != 0
}

func Int64ToRaw(i int64) uint64 {
	return uint64(i)
}

func RawToInt64(r uint64) int64 {
	return int64(r)
}

func Float64ToRaw(f float64) uint64 {
	return math.Float64bits(f)
}

func RawToFloat64(r uint64) float64 {
	return math.Float64frombits(r)
}

func RawPtrToFloat64Ptr(r *uint64) *float64 {
	return (*float64)(unsafe.Pointer(r))
}

func RawPtrToInt64Ptr(r *uint64) *int64 {
	return (*int64)(unsafe.Pointer(r))
}
