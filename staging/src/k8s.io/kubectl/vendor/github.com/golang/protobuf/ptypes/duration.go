// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ptypes

import (
	"errors"
	"fmt"
	"time"

	durationpb "github.com/golang/protobuf/ptypes/duration"
)

// Range of google.protobuf.Duration as specified in duration.proto.
// This is about 10,000 years in seconds.
const (
	maxSeconds = int64(10000 * 365.25 * 24 * 60 * 60)
	minSeconds = -maxSeconds
)

// Duration converts a durationpb.Duration to a time.Duration.
// Duration returns an error if dur is invalid or overflows a time.Duration.
//
// Deprecated: Call the dur.AsDuration and dur.CheckValid methods instead.
func Duration(dur *durationpb.Duration) (time.Duration, error) {
	if err := validateDuration(dur); err != nil {
		return 0, err
	}
	d := time.Duration(dur.Seconds) * time.Second
	if int64(d/time.Second) != dur.Seconds {
		return 0, fmt.Errorf("duration: %v is out of range for time.Duration", dur)
	}
	if dur.Nanos != 0 {
		d += time.Duration(dur.Nanos) * time.Nanosecond
		if (d < 0) != (dur.Nanos < 0) {
			return 0, fmt.Errorf("duration: %v is out of range for time.Duration", dur)
		}
	}
	return d, nil
}

// DurationProto converts a time.Duration to a durationpb.Duration.
//
// Deprecated: Call the durationpb.New function instead.
func DurationProto(d time.Duration) *durationpb.Duration {
	nanos := d.Nanoseconds()
	secs := nanos / 1e9
	nanos -= secs * 1e9
	return &durationpb.Duration{
		Seconds: int64(secs),
		Nanos:   int32(nanos),
	}
}

// validateDuration determines whether the durationpb.Duration is valid
// according to the definition in google/protobuf/duration.proto.
// A valid durpb.Duration may still be too large to fit into a time.Duration
// Note that the range of durationpb.Duration is about 10,000 years,
// while the range of time.Duration is about 290 years.
func validateDuration(dur *durationpb.Duration) error {
	if dur == nil {
		return errors.New("duration: nil Duration")
	}
	if dur.Seconds < minSeconds || dur.Seconds > maxSeconds {
		return fmt.Errorf("duration: %v: seconds out of range", dur)
	}
	if dur.Nanos <= -1e9 || dur.Nanos >= 1e9 {
		return fmt.Errorf("duration: %v: nanos out of range", dur)
	}
	// Seconds and Nanos must have the same sign, unless d.Nanos is zero.
	if (dur.Seconds < 0 && dur.Nanos > 0) || (dur.Seconds > 0 && dur.Nanos < 0) {
		return fmt.Errorf("duration: %v: seconds and nanos have different signs", dur)
	}
	return nil
}
