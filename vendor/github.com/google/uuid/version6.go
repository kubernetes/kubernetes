// Copyright 2023 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import "encoding/binary"

// UUID version 6 is a field-compatible version of UUIDv1, reordered for improved DB locality.
// It is expected that UUIDv6 will primarily be used in contexts where there are existing v1 UUIDs.
// Systems that do not involve legacy UUIDv1 SHOULD consider using UUIDv7 instead.
//
// see https://datatracker.ietf.org/doc/html/draft-peabody-dispatch-new-uuid-format-03#uuidv6
//
// NewV6 returns a Version 6 UUID based on the current NodeID and clock
// sequence, and the current time. If the NodeID has not been set by SetNodeID
// or SetNodeInterface then it will be set automatically. If the NodeID cannot
// be set NewV6 set NodeID is random bits automatically . If clock sequence has not been set by
// SetClockSequence then it will be set automatically. If GetTime fails to
// return the current NewV6 returns Nil and an error.
func NewV6() (UUID, error) {
	var uuid UUID
	now, seq, err := GetTime()
	if err != nil {
		return uuid, err
	}

	/*
	    0                   1                   2                   3
	    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
	   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	   |                           time_high                           |
	   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	   |           time_mid            |      time_low_and_version     |
	   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	   |clk_seq_hi_res |  clk_seq_low  |         node (0-1)            |
	   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	   |                         node (2-5)                            |
	   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	*/

	binary.BigEndian.PutUint64(uuid[0:], uint64(now))
	binary.BigEndian.PutUint16(uuid[8:], seq)

	uuid[6] = 0x60 | (uuid[6] & 0x0F)
	uuid[8] = 0x80 | (uuid[8] & 0x3F)

	nodeMu.Lock()
	if nodeID == zeroID {
		setNodeInterface("")
	}
	copy(uuid[10:], nodeID[:])
	nodeMu.Unlock()

	return uuid, nil
}
