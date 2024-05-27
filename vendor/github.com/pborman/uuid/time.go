// Copyright 2014 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import (
	"encoding/binary"

	guuid "github.com/google/uuid"
)

// A Time represents a time as the number of 100's of nanoseconds since 15 Oct
// 1582.
type Time = guuid.Time

// GetTime returns the current Time (100s of nanoseconds since 15 Oct 1582) and
// clock sequence as well as adjusting the clock sequence as needed.  An error
// is returned if the current time cannot be determined.
func GetTime() (Time, uint16, error) { return guuid.GetTime() }

// ClockSequence returns the current clock sequence, generating one if not
// already set.  The clock sequence is only used for Version 1 UUIDs.
//
// The uuid package does not use global static storage for the clock sequence or
// the last time a UUID was generated.  Unless SetClockSequence a new random
// clock sequence is generated the first time a clock sequence is requested by
// ClockSequence, GetTime, or NewUUID.  (section 4.2.1.1) sequence is generated
// for
func ClockSequence() int { return guuid.ClockSequence() }

// SetClockSequence sets the clock sequence to the lower 14 bits of seq.  Setting to
// -1 causes a new sequence to be generated.
func SetClockSequence(seq int) { guuid.SetClockSequence(seq) }

// Time returns the time in 100s of nanoseconds since 15 Oct 1582 encoded in
// uuid.  It returns false if uuid is not valid.  The time is only well defined
// for version 1 and 2 UUIDs.
func (uuid UUID) Time() (Time, bool) {
	if len(uuid) != 16 {
		return 0, false
	}
	time := int64(binary.BigEndian.Uint32(uuid[0:4]))
	time |= int64(binary.BigEndian.Uint16(uuid[4:6])) << 32
	time |= int64(binary.BigEndian.Uint16(uuid[6:8])&0xfff) << 48
	return Time(time), true
}

// ClockSequence returns the clock sequence encoded in uuid.  It returns false
// if uuid is not valid.  The clock sequence is only well defined for version 1
// and 2 UUIDs.
func (uuid UUID) ClockSequence() (int, bool) {
	if len(uuid) != 16 {
		return 0, false
	}
	return int(binary.BigEndian.Uint16(uuid[8:10])) & 0x3fff, true
}
