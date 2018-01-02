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

type VolumeDurability interface {
	BrickSizeGenerator(size uint64) func() (int, uint64, error)
	MinVolumeSize() uint64
	BricksInSet() int
	SetDurability()
	SetExecutorVolumeRequest(v *executors.VolumeRequest)
}
