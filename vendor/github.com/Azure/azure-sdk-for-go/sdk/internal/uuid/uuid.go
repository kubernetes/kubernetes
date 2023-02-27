//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package uuid

import (
	"crypto/rand"
	"errors"
	"fmt"
	"strconv"
)

// The UUID reserved variants.
const (
	reservedRFC4122 byte = 0x40
)

// A UUID representation compliant with specification in RFC4122 document.
type UUID [16]byte

// New returns a new UUID using the RFC4122 algorithm.
func New() (UUID, error) {
	u := UUID{}
	// Set all bits to pseudo-random values.
	// NOTE: this takes a process-wide lock
	_, err := rand.Read(u[:])
	if err != nil {
		return u, err
	}
	u[8] = (u[8] | reservedRFC4122) & 0x7F // u.setVariant(ReservedRFC4122)

	var version byte = 4
	u[6] = (u[6] & 0xF) | (version << 4) // u.setVersion(4)
	return u, nil
}

// String returns the UUID in "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" format.
func (u UUID) String() string {
	return fmt.Sprintf("%x-%x-%x-%x-%x", u[0:4], u[4:6], u[6:8], u[8:10], u[10:])
}

// Parse parses a string formatted as "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
// or "{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}" into a UUID.
func Parse(s string) (UUID, error) {
	var uuid UUID
	// ensure format
	switch len(s) {
	case 36:
		// xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
	case 38:
		// {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}
		s = s[1:37]
	default:
		return uuid, errors.New("invalid UUID format")
	}
	if s[8] != '-' || s[13] != '-' || s[18] != '-' || s[23] != '-' {
		return uuid, errors.New("invalid UUID format")
	}
	// parse chunks
	for i, x := range [16]int{
		0, 2, 4, 6,
		9, 11,
		14, 16,
		19, 21,
		24, 26, 28, 30, 32, 34} {
		b, err := strconv.ParseUint(s[x:x+2], 16, 8)
		if err != nil {
			return uuid, fmt.Errorf("invalid UUID format: %s", err)
		}
		uuid[i] = byte(b)
	}
	return uuid, nil
}
