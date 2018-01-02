//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package glusterfs

import (
	"github.com/heketi/heketi/executors"
)

type NoneDurability struct {
	VolumeReplicaDurability
}

func NewNoneDurability() *NoneDurability {
	n := &NoneDurability{}
	n.Replica = 1

	return n
}

func (n *NoneDurability) SetDurability() {
	n.Replica = 1
}

func (n *NoneDurability) BricksInSet() int {
	return 1
}

func (n *NoneDurability) SetExecutorVolumeRequest(v *executors.VolumeRequest) {
	v.Type = executors.DurabilityNone
	v.Replica = n.Replica
}
