/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package request

import (
	"fmt"
	"math"
	"time"
)

// SeatSeconds is a measure of work, in units of seat-seconds, using a fixed-point representation.
// `SeatSeconds(n)` represents `n/ssScale` seat-seconds.
// The `ssScale` constant is private to the implementation here,
// no other code should use it.
type SeatSeconds uint64

// MaxSeatsSeconds is the maximum representable value of SeatSeconds
const MaxSeatSeconds = SeatSeconds(math.MaxUint64)

// MinSeatSeconds is the lowest representable value of SeatSeconds
const MinSeatSeconds = SeatSeconds(0)

// SeatsTimeDuration produces the SeatSeconds value for the given factors.
// This is intended only to produce small values, increments in work
// rather than amount of work done since process start.
func SeatsTimesDuration(seats float64, duration time.Duration) SeatSeconds {
	return SeatSeconds(math.Round(seats * float64(duration/time.Nanosecond) / (1e9 / ssScale)))
}

// ToFloat converts to a floating-point representation.
// This conversion may lose precision.
func (ss SeatSeconds) ToFloat() float64 {
	return float64(ss) / ssScale
}

// DurationPerSeat returns duration per seat.
// This division may lose precision.
func (ss SeatSeconds) DurationPerSeat(seats float64) time.Duration {
	return time.Duration(float64(ss) / seats * (float64(time.Second) / ssScale))
}

// String converts to a string.
// This is suitable for large as well as small values.
func (ss SeatSeconds) String() string {
	const div = SeatSeconds(ssScale)
	quo := ss / div
	rem := ss - quo*div
	return fmt.Sprintf("%d.%08dss", quo, rem)
}

const ssScale = 1e8
