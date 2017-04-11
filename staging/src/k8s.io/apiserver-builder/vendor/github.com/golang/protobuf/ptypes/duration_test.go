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

package ptypes

import (
	"math"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	durpb "github.com/golang/protobuf/ptypes/duration"
)

const (
	minGoSeconds = math.MinInt64 / int64(1e9)
	maxGoSeconds = math.MaxInt64 / int64(1e9)
)

var durationTests = []struct {
	proto   *durpb.Duration
	isValid bool
	inRange bool
	dur     time.Duration
}{
	// The zero duration.
	{&durpb.Duration{0, 0}, true, true, 0},
	// Some ordinary non-zero durations.
	{&durpb.Duration{100, 0}, true, true, 100 * time.Second},
	{&durpb.Duration{-100, 0}, true, true, -100 * time.Second},
	{&durpb.Duration{100, 987}, true, true, 100*time.Second + 987},
	{&durpb.Duration{-100, -987}, true, true, -(100*time.Second + 987)},
	// The largest duration representable in Go.
	{&durpb.Duration{maxGoSeconds, int32(math.MaxInt64 - 1e9*maxGoSeconds)}, true, true, math.MaxInt64},
	// The smallest duration representable in Go.
	{&durpb.Duration{minGoSeconds, int32(math.MinInt64 - 1e9*minGoSeconds)}, true, true, math.MinInt64},
	{nil, false, false, 0},
	{&durpb.Duration{-100, 987}, false, false, 0},
	{&durpb.Duration{100, -987}, false, false, 0},
	{&durpb.Duration{math.MinInt64, 0}, false, false, 0},
	{&durpb.Duration{math.MaxInt64, 0}, false, false, 0},
	// The largest valid duration.
	{&durpb.Duration{maxSeconds, 1e9 - 1}, true, false, 0},
	// The smallest valid duration.
	{&durpb.Duration{minSeconds, -(1e9 - 1)}, true, false, 0},
	// The smallest invalid duration above the valid range.
	{&durpb.Duration{maxSeconds + 1, 0}, false, false, 0},
	// The largest invalid duration below the valid range.
	{&durpb.Duration{minSeconds - 1, -(1e9 - 1)}, false, false, 0},
	// One nanosecond past the largest duration representable in Go.
	{&durpb.Duration{maxGoSeconds, int32(math.MaxInt64-1e9*maxGoSeconds) + 1}, true, false, 0},
	// One nanosecond past the smallest duration representable in Go.
	{&durpb.Duration{minGoSeconds, int32(math.MinInt64-1e9*minGoSeconds) - 1}, true, false, 0},
	// One second past the largest duration representable in Go.
	{&durpb.Duration{maxGoSeconds + 1, int32(math.MaxInt64 - 1e9*maxGoSeconds)}, true, false, 0},
	// One second past the smallest duration representable in Go.
	{&durpb.Duration{minGoSeconds - 1, int32(math.MinInt64 - 1e9*minGoSeconds)}, true, false, 0},
}

func TestValidateDuration(t *testing.T) {
	for _, test := range durationTests {
		err := validateDuration(test.proto)
		gotValid := (err == nil)
		if gotValid != test.isValid {
			t.Errorf("validateDuration(%v) = %t, want %t", test.proto, gotValid, test.isValid)
		}
	}
}

func TestDuration(t *testing.T) {
	for _, test := range durationTests {
		got, err := Duration(test.proto)
		gotOK := (err == nil)
		wantOK := test.isValid && test.inRange
		if gotOK != wantOK {
			t.Errorf("Duration(%v) ok = %t, want %t", test.proto, gotOK, wantOK)
		}
		if err == nil && got != test.dur {
			t.Errorf("Duration(%v) = %v, want %v", test.proto, got, test.dur)
		}
	}
}

func TestDurationProto(t *testing.T) {
	for _, test := range durationTests {
		if test.isValid && test.inRange {
			got := DurationProto(test.dur)
			if !proto.Equal(got, test.proto) {
				t.Errorf("DurationProto(%v) = %v, want %v", test.dur, got, test.proto)
			}
		}
	}
}
