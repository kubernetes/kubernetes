// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package internal // import "go.opentelemetry.io/otel/sdk/internal"

import "time"

// MonotonicEndTime returns the end time at present
// but offset from start, monotonically.
//
// The monotonic clock is used in subtractions hence
// the duration since start added back to start gives
// end as a monotonic time.
// See https://golang.org/pkg/time/#hdr-Monotonic_Clocks
func MonotonicEndTime(start time.Time) time.Time {
	return start.Add(time.Since(start))
}
