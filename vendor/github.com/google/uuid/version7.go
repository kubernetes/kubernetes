// Copyright 2023 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import (
	"io"
)

// UUID version 7 features a time-ordered value field derived from the widely
// implemented and well known Unix Epoch timestamp source,
// the number of milliseconds seconds since midnight 1 Jan 1970 UTC, leap seconds excluded.
// As well as improved entropy characteristics over versions 1 or 6.
//
// see https://datatracker.ietf.org/doc/html/draft-peabody-dispatch-new-uuid-format-03#name-uuid-version-7
//
// Implementations SHOULD utilize UUID version 7 over UUID version 1 and 6 if possible.
//
// NewV7 returns a Version 7 UUID based on the current time(Unix Epoch).
// Uses the randomness pool if it was enabled with EnableRandPool.
// On error, NewV7 returns Nil and an error
func NewV7() (UUID, error) {
	uuid, err := NewRandom()
	if err != nil {
		return uuid, err
	}
	makeV7(uuid[:])
	return uuid, nil
}

// NewV7FromReader returns a Version 7 UUID based on the current time(Unix Epoch).
// it use NewRandomFromReader fill random bits.
// On error, NewV7FromReader returns Nil and an error.
func NewV7FromReader(r io.Reader) (UUID, error) {
	uuid, err := NewRandomFromReader(r)
	if err != nil {
		return uuid, err
	}

	makeV7(uuid[:])
	return uuid, nil
}

// makeV7 fill 48 bits time (uuid[0] - uuid[5]), set version b0111 (uuid[6])
// uuid[8] already has the right version number (Variant is 10)
// see function NewV7 and NewV7FromReader
func makeV7(uuid []byte) {
	/*
		 0                   1                   2                   3
		 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
		+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
		|                           unix_ts_ms                          |
		+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
		|          unix_ts_ms           |  ver  |  rand_a (12 bit seq)  |
		+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
		|var|                        rand_b                             |
		+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
		|                            rand_b                             |
		+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	*/
	_ = uuid[15] // bounds check

	t, s := getV7Time()

	uuid[0] = byte(t >> 40)
	uuid[1] = byte(t >> 32)
	uuid[2] = byte(t >> 24)
	uuid[3] = byte(t >> 16)
	uuid[4] = byte(t >> 8)
	uuid[5] = byte(t)

	uuid[6] = 0x70 | (0x0F & byte(s>>8))
	uuid[7] = byte(s)
}

// lastV7time is the last time we returned stored as:
//
//	52 bits of time in milliseconds since epoch
//	12 bits of (fractional nanoseconds) >> 8
var lastV7time int64

const nanoPerMilli = 1000000

// getV7Time returns the time in milliseconds and nanoseconds / 256.
// The returned (milli << 12 + seq) is guarenteed to be greater than
// (milli << 12 + seq) returned by any previous call to getV7Time.
func getV7Time() (milli, seq int64) {
	timeMu.Lock()
	defer timeMu.Unlock()

	nano := timeNow().UnixNano()
	milli = nano / nanoPerMilli
	// Sequence number is between 0 and 3906 (nanoPerMilli>>8)
	seq = (nano - milli*nanoPerMilli) >> 8
	now := milli<<12 + seq
	if now <= lastV7time {
		now = lastV7time + 1
		milli = now >> 12
		seq = now & 0xfff
	}
	lastV7time = now
	return milli, seq
}
