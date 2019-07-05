// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2016 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package types

// This file implements conversions between google.protobuf.Duration
// and time.Duration.

import (
	"errors"
	"fmt"
	"time"
)

const (
	// Range of a Duration in seconds, as specified in
	// google/protobuf/duration.proto. This is about 10,000 years in seconds.
	maxSeconds = int64(10000 * 365.25 * 24 * 60 * 60)
	minSeconds = -maxSeconds
)

// validateDuration determines whether the Duration is valid according to the
// definition in google/protobuf/duration.proto. A valid Duration
// may still be too large to fit into a time.Duration (the range of Duration
// is about 10,000 years, and the range of time.Duration is about 290).
func validateDuration(d *Duration) error {
	if d == nil {
		return errors.New("duration: nil Duration")
	}
	if d.Seconds < minSeconds || d.Seconds > maxSeconds {
		return fmt.Errorf("duration: %#v: seconds out of range", d)
	}
	if d.Nanos <= -1e9 || d.Nanos >= 1e9 {
		return fmt.Errorf("duration: %#v: nanos out of range", d)
	}
	// Seconds and Nanos must have the same sign, unless d.Nanos is zero.
	if (d.Seconds < 0 && d.Nanos > 0) || (d.Seconds > 0 && d.Nanos < 0) {
		return fmt.Errorf("duration: %#v: seconds and nanos have different signs", d)
	}
	return nil
}

// DurationFromProto converts a Duration to a time.Duration. DurationFromProto
// returns an error if the Duration is invalid or is too large to be
// represented in a time.Duration.
func DurationFromProto(p *Duration) (time.Duration, error) {
	if err := validateDuration(p); err != nil {
		return 0, err
	}
	d := time.Duration(p.Seconds) * time.Second
	if int64(d/time.Second) != p.Seconds {
		return 0, fmt.Errorf("duration: %#v is out of range for time.Duration", p)
	}
	if p.Nanos != 0 {
		d += time.Duration(p.Nanos)
		if (d < 0) != (p.Nanos < 0) {
			return 0, fmt.Errorf("duration: %#v is out of range for time.Duration", p)
		}
	}
	return d, nil
}

// DurationProto converts a time.Duration to a Duration.
func DurationProto(d time.Duration) *Duration {
	nanos := d.Nanoseconds()
	secs := nanos / 1e9
	nanos -= secs * 1e9
	return &Duration{
		Seconds: secs,
		Nanos:   int32(nanos),
	}
}
