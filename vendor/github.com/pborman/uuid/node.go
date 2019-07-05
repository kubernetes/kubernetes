// Copyright 2011 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import (
	guuid "github.com/google/uuid"
)

// NodeInterface returns the name of the interface from which the NodeID was
// derived.  The interface "user" is returned if the NodeID was set by
// SetNodeID.
func NodeInterface() string {
	return guuid.NodeInterface()
}

// SetNodeInterface selects the hardware address to be used for Version 1 UUIDs.
// If name is "" then the first usable interface found will be used or a random
// Node ID will be generated.  If a named interface cannot be found then false
// is returned.
//
// SetNodeInterface never fails when name is "".
func SetNodeInterface(name string) bool {
	return guuid.SetNodeInterface(name)
}

// NodeID returns a slice of a copy of the current Node ID, setting the Node ID
// if not already set.
func NodeID() []byte {
	return guuid.NodeID()
}

// SetNodeID sets the Node ID to be used for Version 1 UUIDs.  The first 6 bytes
// of id are used.  If id is less than 6 bytes then false is returned and the
// Node ID is not set.
func SetNodeID(id []byte) bool {
	return guuid.SetNodeID(id)
}

// NodeID returns the 6 byte node id encoded in uuid.  It returns nil if uuid is
// not valid.  The NodeID is only well defined for version 1 and 2 UUIDs.
func (uuid UUID) NodeID() []byte {
	if len(uuid) != 16 {
		return nil
	}
	node := make([]byte, 6)
	copy(node, uuid[10:])
	return node
}
