// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

import (
	"math"
	"time"
)

var (
	doubleTwoTo64 = math.Ldexp(1.0, 64)
)

// addInt64Checked performs addition with overflow detection of two int64 values.
//
// If the operation fails the error return value will be non-nil.
func addInt64Checked(x, y int64) (int64, error) {
	if (y > 0 && x > math.MaxInt64-y) || (y < 0 && x < math.MinInt64-y) {
		return 0, errIntOverflow
	}
	return x + y, nil
}

// subtractInt64Checked performs subtraction with overflow detection of two int64 values.
//
// If the operation fails the error return value will be non-nil.
func subtractInt64Checked(x, y int64) (int64, error) {
	if (y < 0 && x > math.MaxInt64+y) || (y > 0 && x < math.MinInt64+y) {
		return 0, errIntOverflow
	}
	return x - y, nil
}

// negateInt64Checked performs negation with overflow detection of an int64.
//
// If the operation fails the error return value will be non-nil.
func negateInt64Checked(x int64) (int64, error) {
	// In twos complement, negating MinInt64 would result in a valid of MaxInt64+1.
	if x == math.MinInt64 {
		return 0, errIntOverflow
	}
	return -x, nil
}

// multiplyInt64Checked performs multiplication with overflow detection of two int64 value.
//
// If the operation fails the error return value will be non-nil.
func multiplyInt64Checked(x, y int64) (int64, error) {
	// Detecting multiplication overflow is more complicated than the others. The first two detect
	// attempting to negate MinInt64, which would result in MaxInt64+1. The other four detect normal
	// overflow conditions.
	if (x == -1 && y == math.MinInt64) || (y == -1 && x == math.MinInt64) ||
		// x is positive, y is positive
		(x > 0 && y > 0 && x > math.MaxInt64/y) ||
		// x is positive, y is negative
		(x > 0 && y < 0 && y < math.MinInt64/x) ||
		// x is negative, y is positive
		(x < 0 && y > 0 && x < math.MinInt64/y) ||
		// x is negative, y is negative
		(x < 0 && y < 0 && y < math.MaxInt64/x) {
		return 0, errIntOverflow
	}
	return x * y, nil
}

// divideInt64Checked performs division with overflow detection of two int64 values,
// as well as a division by zero check.
//
// If the operation fails the error return value will be non-nil.
func divideInt64Checked(x, y int64) (int64, error) {
	// Division by zero.
	if y == 0 {
		return 0, errDivideByZero
	}
	// In twos complement, negating MinInt64 would result in a valid of MaxInt64+1.
	if x == math.MinInt64 && y == -1 {
		return 0, errIntOverflow
	}
	return x / y, nil
}

// moduloInt64Checked performs modulo with overflow detection of two int64 values
// as well as a modulus by zero check.
//
// If the operation fails the error return value will be non-nil.
func moduloInt64Checked(x, y int64) (int64, error) {
	// Modulus by zero.
	if y == 0 {
		return 0, errModulusByZero
	}
	// In twos complement, negating MinInt64 would result in a valid of MaxInt64+1.
	if x == math.MinInt64 && y == -1 {
		return 0, errIntOverflow
	}
	return x % y, nil
}

// addUint64Checked performs addition with overflow detection of two uint64 values.
//
// If the operation fails due to overflow the error return value will be non-nil.
func addUint64Checked(x, y uint64) (uint64, error) {
	if y > 0 && x > math.MaxUint64-y {
		return 0, errUintOverflow
	}
	return x + y, nil
}

// subtractUint64Checked performs subtraction with overflow detection of two uint64 values.
//
// If the operation fails due to overflow the error return value will be non-nil.
func subtractUint64Checked(x, y uint64) (uint64, error) {
	if y > x {
		return 0, errUintOverflow
	}
	return x - y, nil
}

// multiplyUint64Checked performs multiplication with overflow detection of two uint64 values.
//
// If the operation fails due to overflow the error return value will be non-nil.
func multiplyUint64Checked(x, y uint64) (uint64, error) {
	if y != 0 && x > math.MaxUint64/y {
		return 0, errUintOverflow
	}
	return x * y, nil
}

// divideUint64Checked performs division with a test for division by zero.
//
// If the operation fails the error return value will be non-nil.
func divideUint64Checked(x, y uint64) (uint64, error) {
	if y == 0 {
		return 0, errDivideByZero
	}
	return x / y, nil
}

// moduloUint64Checked performs modulo with a test for modulus by zero.
//
// If the operation fails the error return value will be non-nil.
func moduloUint64Checked(x, y uint64) (uint64, error) {
	if y == 0 {
		return 0, errModulusByZero
	}
	return x % y, nil
}

// addDurationChecked performs addition with overflow detection of two time.Durations.
//
// If the operation fails due to overflow the error return value will be non-nil.
func addDurationChecked(x, y time.Duration) (time.Duration, error) {
	val, err := addInt64Checked(int64(x), int64(y))
	if err != nil {
		return time.Duration(0), err
	}
	return time.Duration(val), nil
}

// subtractDurationChecked performs subtraction with overflow detection of two time.Durations.
//
// If the operation fails due to overflow the error return value will be non-nil.
func subtractDurationChecked(x, y time.Duration) (time.Duration, error) {
	val, err := subtractInt64Checked(int64(x), int64(y))
	if err != nil {
		return time.Duration(0), err
	}
	return time.Duration(val), nil
}

// negateDurationChecked performs negation with overflow detection of a time.Duration.
//
// If the operation fails due to overflow the error return value will be non-nil.
func negateDurationChecked(x time.Duration) (time.Duration, error) {
	val, err := negateInt64Checked(int64(x))
	if err != nil {
		return time.Duration(0), err
	}
	return time.Duration(val), nil
}

// addDurationChecked performs addition with overflow detection of a time.Time and time.Duration.
//
// If the operation fails due to overflow the error return value will be non-nil.
func addTimeDurationChecked(x time.Time, y time.Duration) (time.Time, error) {
	// This is tricky. A time is represented as (int64, int32) where the first is seconds and second
	// is nanoseconds. A duration is int64 representing nanoseconds. We cannot normalize time to int64
	// as it could potentially overflow. The only way to proceed is to break time and duration into
	// second and nanosecond components.

	// First we break time into its components by truncating and subtracting.
	sec1 := x.Truncate(time.Second).Unix()                // Truncate to seconds.
	nsec1 := x.Sub(x.Truncate(time.Second)).Nanoseconds() // Get nanoseconds by truncating and subtracting.

	// Second we break duration into its components by dividing and modulo.
	sec2 := int64(y) / int64(time.Second)  // Truncate to seconds.
	nsec2 := int64(y) % int64(time.Second) // Get remainder.

	// Add seconds first, detecting any overflow.
	sec, err := addInt64Checked(sec1, sec2)
	if err != nil {
		return time.Time{}, err
	}
	// Nanoseconds cannot overflow as time.Time normalizes them to [0, 999999999].
	nsec := nsec1 + nsec2

	// We need to normalize nanoseconds to be positive and carry extra nanoseconds to seconds.
	// Adapted from time.Unix(int64, int64).
	if nsec < 0 || nsec >= int64(time.Second) {
		// Add seconds.
		sec, err = addInt64Checked(sec, nsec/int64(time.Second))
		if err != nil {
			return time.Time{}, err
		}

		nsec -= (nsec / int64(time.Second)) * int64(time.Second)
		if nsec < 0 {
			// Subtract an extra second
			sec, err = addInt64Checked(sec, -1)
			if err != nil {
				return time.Time{}, err
			}
			nsec += int64(time.Second)
		}
	}

	// Check if the the number of seconds from Unix epoch is within our acceptable range.
	if sec < minUnixTime || sec > maxUnixTime {
		return time.Time{}, errTimestampOverflow
	}

	// Return resulting time and propagate time zone.
	return time.Unix(sec, nsec).In(x.Location()), nil
}

// subtractTimeChecked performs subtraction with overflow detection of two time.Time.
//
// If the operation fails due to overflow the error return value will be non-nil.
func subtractTimeChecked(x, y time.Time) (time.Duration, error) {
	// Similar to addTimeDurationOverflow() above.

	// First we break time into its components by truncating and subtracting.
	sec1 := x.Truncate(time.Second).Unix()                // Truncate to seconds.
	nsec1 := x.Sub(x.Truncate(time.Second)).Nanoseconds() // Get nanoseconds by truncating and subtracting.

	// Second we break duration into its components by truncating and subtracting.
	sec2 := y.Truncate(time.Second).Unix()                // Truncate to seconds.
	nsec2 := y.Sub(y.Truncate(time.Second)).Nanoseconds() // Get nanoseconds by truncating and subtracting.

	// Subtract seconds first, detecting any overflow.
	sec, err := subtractInt64Checked(sec1, sec2)
	if err != nil {
		return time.Duration(0), err
	}

	// Nanoseconds cannot overflow as time.Time normalizes them to [0, 999999999].
	nsec := nsec1 - nsec2

	// Scale seconds to nanoseconds detecting overflow.
	tsec, err := multiplyInt64Checked(sec, int64(time.Second))
	if err != nil {
		return time.Duration(0), err
	}

	// Lastly we need to add the two nanoseconds together.
	val, err := addInt64Checked(tsec, nsec)
	if err != nil {
		return time.Duration(0), err
	}

	return time.Duration(val), nil
}

// subtractTimeDurationChecked performs subtraction with overflow detection of a time.Time and
// time.Duration.
//
// If the operation fails due to overflow the error return value will be non-nil.
func subtractTimeDurationChecked(x time.Time, y time.Duration) (time.Time, error) {
	// The easiest way to implement this is to negate y and add them.
	// x - y = x + -y
	val, err := negateDurationChecked(y)
	if err != nil {
		return time.Time{}, err
	}
	return addTimeDurationChecked(x, val)
}

// doubleToInt64Checked converts a double to an int64 value.
//
// If the conversion fails due to overflow the error return value will be non-nil.
func doubleToInt64Checked(v float64) (int64, error) {
	if math.IsInf(v, 0) || math.IsNaN(v) || v <= float64(math.MinInt64) || v >= float64(math.MaxInt64) {
		return 0, errIntOverflow
	}
	return int64(v), nil
}

// doubleToInt64Checked converts a double to a uint64 value.
//
// If the conversion fails due to overflow the error return value will be non-nil.
func doubleToUint64Checked(v float64) (uint64, error) {
	if math.IsInf(v, 0) || math.IsNaN(v) || v < 0 || v >= doubleTwoTo64 {
		return 0, errUintOverflow
	}
	return uint64(v), nil
}

// int64toUint64Checked converts an int64 to a uint64 value.
//
// If the conversion fails due to overflow the error return value will be non-nil.
func int64ToUint64Checked(v int64) (uint64, error) {
	if v < 0 {
		return 0, errUintOverflow
	}
	return uint64(v), nil
}

// int64toInt32Checked converts an int64 to an int32 value.
//
// If the conversion fails due to overflow the error return value will be non-nil.
func int64ToInt32Checked(v int64) (int32, error) {
	if v < math.MinInt32 || v > math.MaxInt32 {
		return 0, errIntOverflow
	}
	return int32(v), nil
}

// uint64toUint32Checked converts a uint64 to a uint32 value.
//
// If the conversion fails due to overflow the error return value will be non-nil.
func uint64ToUint32Checked(v uint64) (uint32, error) {
	if v > math.MaxUint32 {
		return 0, errUintOverflow
	}
	return uint32(v), nil
}

// uint64toInt64Checked converts a uint64 to an int64 value.
//
// If the conversion fails due to overflow the error return value will be non-nil.
func uint64ToInt64Checked(v uint64) (int64, error) {
	if v > math.MaxInt64 {
		return 0, errIntOverflow
	}
	return int64(v), nil
}
