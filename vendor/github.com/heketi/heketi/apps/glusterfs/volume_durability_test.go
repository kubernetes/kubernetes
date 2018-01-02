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
	"testing"

	"github.com/heketi/heketi/executors"
	"github.com/heketi/tests"
)

func TestNoneDurabilityDefaults(t *testing.T) {
	r := &NoneDurability{}
	tests.Assert(t, r.Replica == 0)

	r.SetDurability()
	tests.Assert(t, r.Replica == 1)
}

func TestDisperseDurabilityDefaults(t *testing.T) {
	r := &VolumeDisperseDurability{}
	tests.Assert(t, r.Data == 0)
	tests.Assert(t, r.Redundancy == 0)

	r.SetDurability()
	tests.Assert(t, r.Data == DEFAULT_EC_DATA)
	tests.Assert(t, r.Redundancy == DEFAULT_EC_REDUNDANCY)
}

func TestReplicaDurabilityDefaults(t *testing.T) {
	r := &VolumeReplicaDurability{}
	tests.Assert(t, r.Replica == 0)

	r.SetDurability()
	tests.Assert(t, r.Replica == DEFAULT_REPLICA)
}

func TestNoneDurabilitySetExecutorRequest(t *testing.T) {
	r := &NoneDurability{}
	r.SetDurability()

	v := &executors.VolumeRequest{}
	r.SetExecutorVolumeRequest(v)
	tests.Assert(t, v.Replica == 1)
	tests.Assert(t, v.Type == executors.DurabilityNone)
}

func TestDisperseDurabilitySetExecutorRequest(t *testing.T) {
	r := &VolumeDisperseDurability{}
	r.SetDurability()

	v := &executors.VolumeRequest{}
	r.SetExecutorVolumeRequest(v)
	tests.Assert(t, v.Data == r.Data)
	tests.Assert(t, v.Redundancy == r.Redundancy)
	tests.Assert(t, v.Type == executors.DurabilityDispersion)
}

func TestReplicaDurabilitySetExecutorRequest(t *testing.T) {
	r := &VolumeReplicaDurability{}
	r.SetDurability()

	v := &executors.VolumeRequest{}
	r.SetExecutorVolumeRequest(v)
	tests.Assert(t, v.Replica == r.Replica)
	tests.Assert(t, v.Type == executors.DurabilityReplica)
}

func TestNoneDurability(t *testing.T) {
	r := &NoneDurability{}
	r.SetDurability()

	gen := r.BrickSizeGenerator(100 * GB)

	// Gen 1
	sets, brick_size, err := gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 1)
	tests.Assert(t, brick_size == 100*GB)
	tests.Assert(t, 1 == r.BricksInSet())

	// Gen 2
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 2)
	tests.Assert(t, brick_size == 50*GB)
	tests.Assert(t, 1 == r.BricksInSet())

	// Gen 3
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 4)
	tests.Assert(t, brick_size == 25*GB)
	tests.Assert(t, 1 == r.BricksInSet())

	// Gen 4
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 8)
	tests.Assert(t, brick_size == 12800*1024)
	tests.Assert(t, 1 == r.BricksInSet())

	// Gen 5
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 16)
	tests.Assert(t, brick_size == 6400*1024)
	tests.Assert(t, 1 == r.BricksInSet())

	// Gen 6
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil, err)
	tests.Assert(t, sets == 32)
	tests.Assert(t, brick_size == 3200*1024)
	tests.Assert(t, 1 == r.BricksInSet())

	// Gen 7
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil, err)
	tests.Assert(t, sets == 64)
	tests.Assert(t, brick_size == 1600*1024)
	tests.Assert(t, 1 == r.BricksInSet())

	// Gen 8
	sets, brick_size, err = gen()
	tests.Assert(t, err == ErrMinimumBrickSize)
	tests.Assert(t, sets == 0)
	tests.Assert(t, brick_size == 0)
	tests.Assert(t, 1 == r.BricksInSet())
}

func TestDisperseDurability(t *testing.T) {

	r := &VolumeDisperseDurability{}
	r.Data = 8
	r.Redundancy = 3

	gen := r.BrickSizeGenerator(200 * GB)

	// Gen 1
	sets, brick_size, err := gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 1)
	tests.Assert(t, brick_size == uint64(200*GB/8))
	tests.Assert(t, 8+3 == r.BricksInSet())

	// Gen 2
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 2)
	tests.Assert(t, brick_size == uint64(100*GB/8))
	tests.Assert(t, 8+3 == r.BricksInSet())

	// Gen 3
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 4)
	tests.Assert(t, brick_size == uint64(50*GB/8))
	tests.Assert(t, 8+3 == r.BricksInSet())

	// Gen 4
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil, err)
	tests.Assert(t, sets == 8)
	tests.Assert(t, brick_size == uint64(25*GB/8))
	tests.Assert(t, 8+3 == r.BricksInSet())

	// Gen 5
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil, err)
	tests.Assert(t, sets == 16)
	tests.Assert(t, brick_size == uint64(12800*1024/8))
	tests.Assert(t, 8+3 == r.BricksInSet())

	// Gen 6
	sets, brick_size, err = gen()
	tests.Assert(t, err == ErrMinimumBrickSize)
	tests.Assert(t, 8+3 == r.BricksInSet())
}

func TestDisperseDurabilityLargeBrickGenerator(t *testing.T) {
	r := &VolumeDisperseDurability{}
	r.Data = 8
	r.Redundancy = 3

	gen := r.BrickSizeGenerator(800 * TB)

	// Gen 1
	sets, brick_size, err := gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 32)
	tests.Assert(t, brick_size == 3200*GB)
	tests.Assert(t, 8+3 == r.BricksInSet())
}

func TestReplicaDurabilityGenerator(t *testing.T) {
	r := &VolumeReplicaDurability{}
	r.Replica = 2

	gen := r.BrickSizeGenerator(100 * GB)

	// Gen 1
	sets, brick_size, err := gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 1)
	tests.Assert(t, brick_size == 100*GB)
	tests.Assert(t, 2 == r.BricksInSet())

	// Gen 2
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 2, "sets we got:", sets)
	tests.Assert(t, brick_size == 50*GB)
	tests.Assert(t, 2 == r.BricksInSet())

	// Gen 3
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 4)
	tests.Assert(t, brick_size == 25*GB)
	tests.Assert(t, 2 == r.BricksInSet())

	// Gen 4
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 8)
	tests.Assert(t, brick_size == 12800*1024)
	tests.Assert(t, 2 == r.BricksInSet())

	// Gen 5
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 16)
	tests.Assert(t, brick_size == 6400*1024)
	tests.Assert(t, 2 == r.BricksInSet())

	// Gen 6
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil, err)
	tests.Assert(t, sets == 32, sets)
	tests.Assert(t, brick_size == 3200*1024)
	tests.Assert(t, 2 == r.BricksInSet())

	// Gen 7
	sets, brick_size, err = gen()
	tests.Assert(t, err == nil, err)
	tests.Assert(t, sets == 64, sets)
	tests.Assert(t, brick_size == 1600*1024)
	tests.Assert(t, 2 == r.BricksInSet())

	// Gen 8
	sets, brick_size, err = gen()
	tests.Assert(t, err == ErrMinimumBrickSize)
	tests.Assert(t, sets == 0)
	tests.Assert(t, brick_size == 0)
	tests.Assert(t, 2 == r.BricksInSet())
}

func TestReplicaDurabilityLargeBrickGenerator(t *testing.T) {
	r := &VolumeReplicaDurability{}
	r.Replica = 2

	gen := r.BrickSizeGenerator(100 * TB)

	// Gen 1
	sets, brick_size, err := gen()
	tests.Assert(t, err == nil)
	tests.Assert(t, sets == 32)
	tests.Assert(t, brick_size == 3200*GB)
	tests.Assert(t, 2 == r.BricksInSet())
}

func TestNoneDurabilityMinVolumeSize(t *testing.T) {
	r := &NoneDurability{}
	r.SetDurability()

	minvolsize := r.MinVolumeSize()

	tests.Assert(t, minvolsize == BrickMinSize)
}

func TestReplicaDurabilityMinVolumeSize(t *testing.T) {
	r := &VolumeReplicaDurability{}
	r.Replica = 3

	minvolsize := r.MinVolumeSize()

	tests.Assert(t, minvolsize == BrickMinSize)
}

func TestDisperseDurabilityMinVolumeSize(t *testing.T) {
	r := &VolumeDisperseDurability{}
	r.Data = 8
	r.Redundancy = 3

	minvolsize := r.MinVolumeSize()

	tests.Assert(t, minvolsize == BrickMinSize*8)
}
