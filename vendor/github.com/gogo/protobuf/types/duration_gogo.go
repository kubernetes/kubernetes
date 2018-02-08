// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2016, The GoGo Authors. All rights reserved.
// http://github.com/gogo/protobuf
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

import (
	"fmt"
	"time"
)

func NewPopulatedDuration(r interface {
	Int63() int64
}, easy bool) *Duration {
	this := &Duration{}
	maxSecs := time.Hour.Nanoseconds() / 1e9
	max := 2 * maxSecs
	s := int64(r.Int63()) % max
	s -= maxSecs
	neg := int64(1)
	if s < 0 {
		neg = -1
	}
	this.Seconds = s
	this.Nanos = int32(neg * (r.Int63() % 1e9))
	return this
}

func (d *Duration) String() string {
	td, err := DurationFromProto(d)
	if err != nil {
		return fmt.Sprintf("(%v)", err)
	}
	return td.String()
}

func NewPopulatedStdDuration(r interface {
	Int63() int64
}, easy bool) *time.Duration {
	dur := NewPopulatedDuration(r, easy)
	d, err := DurationFromProto(dur)
	if err != nil {
		return nil
	}
	return &d
}

func SizeOfStdDuration(d time.Duration) int {
	dur := DurationProto(d)
	return dur.Size()
}

func StdDurationMarshal(d time.Duration) ([]byte, error) {
	size := SizeOfStdDuration(d)
	buf := make([]byte, size)
	_, err := StdDurationMarshalTo(d, buf)
	return buf, err
}

func StdDurationMarshalTo(d time.Duration, data []byte) (int, error) {
	dur := DurationProto(d)
	return dur.MarshalTo(data)
}

func StdDurationUnmarshal(d *time.Duration, data []byte) error {
	dur := &Duration{}
	if err := dur.Unmarshal(data); err != nil {
		return err
	}
	dd, err := DurationFromProto(dur)
	if err != nil {
		return err
	}
	*d = dd
	return nil
}
