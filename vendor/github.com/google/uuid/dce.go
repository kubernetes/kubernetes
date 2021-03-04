// Copyright 2016 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import (
	"encoding/binary"
	"fmt"
	"os"
)

// A Domain represents a Version 2 domain
type Domain byte

// Domain constants for DCE Security (Version 2) UUIDs.
const (
	Person = Domain(0)
	Group  = Domain(1)
	Org    = Domain(2)
)

// NewDCESecurity returns a DCE Security (Version 2) UUID.
//
// The domain should be one of Person, Group or Org.
// On a POSIX system the id should be the users UID for the Person
// domain and the users GID for the Group.  The meaning of id for
// the domain Org or on non-POSIX systems is site defined.
//
// For a given domain/id pair the same token may be returned for up to
// 7 minutes and 10 seconds.
func NewDCESecurity(domain Domain, id uint32) (UUID, error) {
	uuid, err := NewUUID()
	if err == nil {
		uuid[6] = (uuid[6] & 0x0f) | 0x20 // Version 2
		uuid[9] = byte(domain)
		binary.BigEndian.PutUint32(uuid[0:], id)
	}
	return uuid, err
}

// NewDCEPerson returns a DCE Security (Version 2) UUID in the person
// domain with the id returned by os.Getuid.
//
//  NewDCESecurity(Person, uint32(os.Getuid()))
func NewDCEPerson() (UUID, error) {
	return NewDCESecurity(Person, uint32(os.Getuid()))
}

// NewDCEGroup returns a DCE Security (Version 2) UUID in the group
// domain with the id returned by os.Getgid.
//
//  NewDCESecurity(Group, uint32(os.Getgid()))
func NewDCEGroup() (UUID, error) {
	return NewDCESecurity(Group, uint32(os.Getgid()))
}

// Domain returns the domain for a Version 2 UUID.  Domains are only defined
// for Version 2 UUIDs.
func (uuid UUID) Domain() Domain {
	return Domain(uuid[9])
}

// ID returns the id for a Version 2 UUID. IDs are only defined for Version 2
// UUIDs.
func (uuid UUID) ID() uint32 {
	return binary.BigEndian.Uint32(uuid[0:4])
}

func (d Domain) String() string {
	switch d {
	case Person:
		return "Person"
	case Group:
		return "Group"
	case Org:
		return "Org"
	}
	return fmt.Sprintf("Domain%d", int(d))
}
