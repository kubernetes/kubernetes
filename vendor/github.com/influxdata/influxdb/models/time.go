package models

// Helper time methods since parsing time can easily overflow and we only support a
// specific time range.

import (
	"fmt"
	"math"
	"time"
)

const (
	// MinNanoTime is the minumum time that can be represented.
	//
	// 1677-09-21 00:12:43.145224194 +0000 UTC
	//
	// The two lowest minimum integers are used as sentinel values.  The
	// minimum value needs to be used as a value lower than any other value for
	// comparisons and another separate value is needed to act as a sentinel
	// default value that is unusable by the user, but usable internally.
	// Because these two values need to be used for a special purpose, we do
	// not allow users to write points at these two times.
	MinNanoTime = int64(math.MinInt64) + 2

	// MaxNanoTime is the maximum time that can be represented.
	//
	// 2262-04-11 23:47:16.854775806 +0000 UTC
	//
	// The highest time represented by a nanosecond needs to be used for an
	// exclusive range in the shard group, so the maximum time needs to be one
	// less than the possible maximum number of nanoseconds representable by an
	// int64 so that we don't lose a point at that one time.
	MaxNanoTime = int64(math.MaxInt64) - 1
)

var (
	minNanoTime = time.Unix(0, MinNanoTime).UTC()
	maxNanoTime = time.Unix(0, MaxNanoTime).UTC()

	// ErrTimeOutOfRange gets returned when time is out of the representable range using int64 nanoseconds since the epoch.
	ErrTimeOutOfRange = fmt.Errorf("time outside range %d - %d", MinNanoTime, MaxNanoTime)
)

// SafeCalcTime safely calculates the time given. Will return error if the time is outside the
// supported range.
func SafeCalcTime(timestamp int64, precision string) (time.Time, error) {
	mult := GetPrecisionMultiplier(precision)
	if t, ok := safeSignedMult(timestamp, mult); ok {
		tme := time.Unix(0, t).UTC()
		return tme, CheckTime(tme)
	}

	return time.Time{}, ErrTimeOutOfRange
}

// CheckTime checks that a time is within the safe range.
func CheckTime(t time.Time) error {
	if t.Before(minNanoTime) || t.After(maxNanoTime) {
		return ErrTimeOutOfRange
	}
	return nil
}

// Perform the multiplication and check to make sure it didn't overflow.
func safeSignedMult(a, b int64) (int64, bool) {
	if a == 0 || b == 0 || a == 1 || b == 1 {
		return a * b, true
	}
	if a == MinNanoTime || b == MaxNanoTime {
		return 0, false
	}
	c := a * b
	return c, c/b == a
}
