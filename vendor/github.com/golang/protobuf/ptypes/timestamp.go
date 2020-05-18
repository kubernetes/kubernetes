// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ptypes

import (
	"errors"
	"fmt"
	"time"

	timestamppb "github.com/golang/protobuf/ptypes/timestamp"
)

// Range of google.protobuf.Duration as specified in timestamp.proto.
const (
	// Seconds field of the earliest valid Timestamp.
	// This is time.Date(1, 1, 1, 0, 0, 0, 0, time.UTC).Unix().
	minValidSeconds = -62135596800
	// Seconds field just after the latest valid Timestamp.
	// This is time.Date(10000, 1, 1, 0, 0, 0, 0, time.UTC).Unix().
	maxValidSeconds = 253402300800
)

// Timestamp converts a timestamppb.Timestamp to a time.Time.
// It returns an error if the argument is invalid.
//
// Unlike most Go functions, if Timestamp returns an error, the first return
// value is not the zero time.Time. Instead, it is the value obtained from the
// time.Unix function when passed the contents of the Timestamp, in the UTC
// locale. This may or may not be a meaningful time; many invalid Timestamps
// do map to valid time.Times.
//
// A nil Timestamp returns an error. The first return value in that case is
// undefined.
func Timestamp(ts *timestamppb.Timestamp) (time.Time, error) {
	// Don't return the zero value on error, because corresponds to a valid
	// timestamp. Instead return whatever time.Unix gives us.
	var t time.Time
	if ts == nil {
		t = time.Unix(0, 0).UTC() // treat nil like the empty Timestamp
	} else {
		t = time.Unix(ts.Seconds, int64(ts.Nanos)).UTC()
	}
	return t, validateTimestamp(ts)
}

// TimestampNow returns a google.protobuf.Timestamp for the current time.
func TimestampNow() *timestamppb.Timestamp {
	ts, err := TimestampProto(time.Now())
	if err != nil {
		panic("ptypes: time.Now() out of Timestamp range")
	}
	return ts
}

// TimestampProto converts the time.Time to a google.protobuf.Timestamp proto.
// It returns an error if the resulting Timestamp is invalid.
func TimestampProto(t time.Time) (*timestamppb.Timestamp, error) {
	ts := &timestamppb.Timestamp{
		Seconds: t.Unix(),
		Nanos:   int32(t.Nanosecond()),
	}
	if err := validateTimestamp(ts); err != nil {
		return nil, err
	}
	return ts, nil
}

// TimestampString returns the RFC 3339 string for valid Timestamps.
// For invalid Timestamps, it returns an error message in parentheses.
func TimestampString(ts *timestamppb.Timestamp) string {
	t, err := Timestamp(ts)
	if err != nil {
		return fmt.Sprintf("(%v)", err)
	}
	return t.Format(time.RFC3339Nano)
}

// validateTimestamp determines whether a Timestamp is valid.
// A valid timestamp represents a time in the range [0001-01-01, 10000-01-01)
// and has a Nanos field in the range [0, 1e9).
//
// If the Timestamp is valid, validateTimestamp returns nil.
// Otherwise, it returns an error that describes the problem.
//
// Every valid Timestamp can be represented by a time.Time,
// but the converse is not true.
func validateTimestamp(ts *timestamppb.Timestamp) error {
	if ts == nil {
		return errors.New("timestamp: nil Timestamp")
	}
	if ts.Seconds < minValidSeconds {
		return fmt.Errorf("timestamp: %v before 0001-01-01", ts)
	}
	if ts.Seconds >= maxValidSeconds {
		return fmt.Errorf("timestamp: %v after 10000-01-01", ts)
	}
	if ts.Nanos < 0 || ts.Nanos >= 1e9 {
		return fmt.Errorf("timestamp: %v: nanos not in range [0, 1e9)", ts)
	}
	return nil
}
