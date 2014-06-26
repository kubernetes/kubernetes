// Copyright 2011 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import (
	"encoding/binary"
)

// NewUUID returns a Version 1 UUID based on the current NodeID and clock
// sequence, and the current time.  If the NodeID has not been set by SetNodeID
// or SetNodeInterface then it will be set automatically.  If the NodeID cannot
// be set NewUUID returns nil.  If clock sequence has not been set by
// SetClockSequence then it will be set automatically.  If GetTime fails to
// return the current NewUUID returns nil.
func NewUUID() UUID {
	if nodeID == nil {
		SetNodeInterface("")
	}

	now, err := GetTime()
	if err != nil {
		return nil
	}

	uuid := make([]byte, 16)

	time_low := uint32(now & 0xffffffff)
	time_mid := uint16((now >> 32) & 0xffff)
	time_hi := uint16((now >> 48) & 0x0fff)
	time_hi |= 0x1000 // Version 1

	binary.BigEndian.PutUint32(uuid[0:], time_low)
	binary.BigEndian.PutUint16(uuid[4:], time_mid)
	binary.BigEndian.PutUint16(uuid[6:], time_hi)
	binary.BigEndian.PutUint16(uuid[8:], clock_seq)
	copy(uuid[10:], nodeID)

	return uuid
}
