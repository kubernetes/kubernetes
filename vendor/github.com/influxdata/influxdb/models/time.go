package models

// Helper time methods since parsing time can easily overflow and we only support a
// specific time range.

import (
	"fmt"
	"math"
	"time"
)

var (
	// MaxNanoTime is the maximum time that can be represented via int64 nanoseconds since the epoch.
	MaxNanoTime = time.Unix(0, math.MaxInt64).UTC()
	// MinNanoTime is the minumum time that can be represented via int64 nanoseconds since the epoch.
	MinNanoTime = time.Unix(0, math.MinInt64).UTC()

	// ErrTimeOutOfRange gets returned when time is out of the representable range using int64 nanoseconds since the epoch.
	ErrTimeOutOfRange = fmt.Errorf("time outside range %s - %s", MinNanoTime, MaxNanoTime)
)

// SafeCalcTime safely calculates the time given. Will return error if the time is outside the
// supported range.
func SafeCalcTime(timestamp int64, precision string) (time.Time, error) {
	mult := GetPrecisionMultiplier(precision)
	if t, ok := safeSignedMult(timestamp, mult); ok {
		return time.Unix(0, t).UTC(), nil
	}

	return time.Time{}, ErrTimeOutOfRange
}

// CheckTime checks that a time is within the safe range.
func CheckTime(t time.Time) error {
	if t.Before(MinNanoTime) || t.After(MaxNanoTime) {
		return ErrTimeOutOfRange
	}
	return nil
}

// Perform the multiplication and check to make sure it didn't overflow.
func safeSignedMult(a, b int64) (int64, bool) {
	if a == 0 || b == 0 || a == 1 || b == 1 {
		return a * b, true
	}
	if a == math.MinInt64 || b == math.MaxInt64 {
		return 0, false
	}
	c := a * b
	return c, c/b == a
}
