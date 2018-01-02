//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package glusterfs

var (
	// Default limits
	BrickMinSize = uint64(1 * GB)
	BrickMaxSize = uint64(4 * TB)
	BrickMaxNum  = 32
)
