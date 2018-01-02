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
	"github.com/heketi/heketi/pkg/glusterfs/api"
)

type VolumeDisperseDurability struct {
	api.DisperseDurability
}

func NewVolumeDisperseDurability(d *api.DisperseDurability) *VolumeDisperseDurability {
	v := &VolumeDisperseDurability{}
	v.Data = d.Data
	v.Redundancy = d.Redundancy

	return v
}

func (d *VolumeDisperseDurability) SetDurability() {
	if d.Data == 0 {
		d.Data = DEFAULT_EC_DATA
	}
	if d.Redundancy == 0 {
		d.Redundancy = DEFAULT_EC_REDUNDANCY
	}
}

func (d *VolumeDisperseDurability) BrickSizeGenerator(size uint64) func() (int, uint64, error) {

	sets := 1
	return func() (int, uint64, error) {

		var brick_size uint64
		var num_sets int

		for {
			num_sets = sets
			sets *= 2
			brick_size = size / uint64(num_sets)

			// Divide what would be the brick size for replica by the
			// number of data drives in the disperse request
			brick_size /= uint64(d.Data)

			if brick_size < BrickMinSize {
				return 0, 0, ErrMinimumBrickSize
			} else if brick_size <= BrickMaxSize {
				break
			}
		}

		return num_sets, brick_size, nil
	}
}

func (d *VolumeDisperseDurability) MinVolumeSize() uint64 {
	return BrickMinSize * uint64(d.Data)
}

func (d *VolumeDisperseDurability) BricksInSet() int {
	return d.Data + d.Redundancy
}

func (d *VolumeDisperseDurability) SetExecutorVolumeRequest(v *executors.VolumeRequest) {
	v.Type = executors.DurabilityDispersion
	v.Data = d.Data
	v.Redundancy = d.Redundancy
}
