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
	"bytes"
	"encoding/gob"

	"github.com/boltdb/bolt"
	"github.com/heketi/heketi/executors"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/lpabon/godbc"
)

type BrickEntry struct {
	Info             api.BrickInfo
	TpSize           uint64
	PoolMetadataSize uint64
	gidRequested     int64
}

func BrickList(tx *bolt.Tx) ([]string, error) {

	list := EntryKeys(tx, BOLTDB_BUCKET_BRICK)
	if list == nil {
		return nil, ErrAccessList
	}
	return list, nil
}

func NewBrickEntry(size, tpsize, poolMetadataSize uint64,
	deviceid, nodeid string, gid int64, volumeid string) *BrickEntry {

	godbc.Require(size > 0)
	godbc.Require(tpsize > 0)
	godbc.Require(deviceid != "")
	godbc.Require(nodeid != "")

	entry := &BrickEntry{}
	entry.gidRequested = gid
	entry.TpSize = tpsize
	entry.PoolMetadataSize = poolMetadataSize
	entry.Info.Id = utils.GenUUID()
	entry.Info.Size = size
	entry.Info.NodeId = nodeid
	entry.Info.DeviceId = deviceid
	entry.Info.VolumeId = volumeid

	godbc.Ensure(entry.Info.Id != "")
	godbc.Ensure(entry.TpSize == tpsize)
	godbc.Ensure(entry.Info.Size == size)
	godbc.Ensure(entry.Info.NodeId == nodeid)
	godbc.Ensure(entry.Info.DeviceId == deviceid)

	return entry
}

func NewBrickEntryFromId(tx *bolt.Tx, id string) (*BrickEntry, error) {
	godbc.Require(tx != nil)

	entry := &BrickEntry{}
	err := EntryLoad(tx, entry, id)
	if err != nil {
		return nil, err
	}

	return entry, nil
}

func (b *BrickEntry) BucketName() string {
	return BOLTDB_BUCKET_BRICK
}

func (b *BrickEntry) SetId(id string) {
	b.Info.Id = id
}

func (b *BrickEntry) Id() string {
	return b.Info.Id
}

func (b *BrickEntry) Save(tx *bolt.Tx) error {
	godbc.Require(tx != nil)
	godbc.Require(len(b.Info.Id) > 0)

	return EntrySave(tx, b, b.Info.Id)
}

func (b *BrickEntry) Delete(tx *bolt.Tx) error {
	return EntryDelete(tx, b, b.Info.Id)
}

func (b *BrickEntry) NewInfoResponse(tx *bolt.Tx) (*api.BrickInfo, error) {
	info := &api.BrickInfo{}
	*info = b.Info

	return info, nil
}

func (b *BrickEntry) Marshal() ([]byte, error) {
	var buffer bytes.Buffer
	enc := gob.NewEncoder(&buffer)
	err := enc.Encode(*b)

	return buffer.Bytes(), err
}

func (b *BrickEntry) Unmarshal(buffer []byte) error {
	dec := gob.NewDecoder(bytes.NewReader(buffer))
	err := dec.Decode(b)
	if err != nil {
		return err
	}

	return nil
}

func (b *BrickEntry) Create(db *bolt.DB, executor executors.Executor) error {
	godbc.Require(db != nil)
	godbc.Require(b.TpSize > 0)
	godbc.Require(b.Info.Size > 0)

	// Get node hostname
	var host string
	err := db.View(func(tx *bolt.Tx) error {
		node, err := NewNodeEntryFromId(tx, b.Info.NodeId)
		if err != nil {
			return err
		}

		host = node.ManageHostName()
		godbc.Check(host != "")
		return nil
	})
	if err != nil {
		return err
	}

	// Create request
	req := &executors.BrickRequest{}
	req.Gid = b.gidRequested
	req.Name = b.Info.Id
	req.Size = b.Info.Size
	req.TpSize = b.TpSize
	req.VgId = b.Info.DeviceId
	req.PoolMetadataSize = b.PoolMetadataSize

	// Create brick on node
	logger.Info("Creating brick %v", b.Info.Id)
	info, err := executor.BrickCreate(host, req)
	if err != nil {
		return err
	}
	b.Info.Path = info.Path

	godbc.Ensure(b.Info.Path != "")

	return nil
}

func (b *BrickEntry) Destroy(db *bolt.DB, executor executors.Executor) error {

	godbc.Require(db != nil)
	godbc.Require(b.TpSize > 0)
	godbc.Require(b.Info.Size > 0)

	// Get node hostname
	var host string
	err := db.View(func(tx *bolt.Tx) error {
		node, err := NewNodeEntryFromId(tx, b.Info.NodeId)
		if err != nil {
			return err
		}

		host = node.ManageHostName()
		godbc.Check(host != "")
		return nil
	})
	if err != nil {
		return err
	}

	// Create request
	req := &executors.BrickRequest{}
	req.Name = b.Info.Id
	req.Size = b.Info.Size
	req.TpSize = b.TpSize
	req.VgId = b.Info.DeviceId

	// Delete brick on node
	logger.Info("Deleting brick %v", b.Info.Id)
	err = executor.BrickDestroy(host, req)
	if err != nil {
		return err
	}

	return nil
}

func (b *BrickEntry) DestroyCheck(db *bolt.DB, executor executors.Executor) error {
	godbc.Require(db != nil)
	godbc.Require(b.TpSize > 0)
	godbc.Require(b.Info.Size > 0)

	// Get node hostname
	var host string
	err := db.View(func(tx *bolt.Tx) error {
		node, err := NewNodeEntryFromId(tx, b.Info.NodeId)
		if err != nil {
			return err
		}

		host = node.ManageHostName()
		godbc.Check(host != "")
		return nil
	})
	if err != nil {
		return err
	}

	// Create request
	req := &executors.BrickRequest{}
	req.Name = b.Info.Id
	req.Size = b.Info.Size
	req.TpSize = b.TpSize
	req.VgId = b.Info.DeviceId

	// Check brick on node
	return executor.BrickDestroyCheck(host, req)
}

// Size consumed on device
func (b *BrickEntry) TotalSize() uint64 {
	return b.TpSize + b.PoolMetadataSize
}

func BrickEntryUpgrade(tx *bolt.Tx) error {
	err := addVolumeIdInBrickEntry(tx)
	if err != nil {
		return err
	}
	return nil
}

func addVolumeIdInBrickEntry(tx *bolt.Tx) error {
	clusters, err := ClusterList(tx)
	if err != nil {
		return err
	}
	for _, cluster := range clusters {
		clusterEntry, err := NewClusterEntryFromId(tx, cluster)
		if err != nil {
			return err
		}
		for _, volume := range clusterEntry.Info.Volumes {
			volumeEntry, err := NewVolumeEntryFromId(tx, volume)
			if err != nil {
				return err
			}
			for _, brick := range volumeEntry.Bricks {
				brickEntry, err := NewBrickEntryFromId(tx, brick)
				if err != nil {
					return err
				}
				if brickEntry.Info.VolumeId == "" {
					brickEntry.Info.VolumeId = volume
					err = brickEntry.Save(tx)
					if err != nil {
						return err
					}
				} else {
					break
				}
			}
		}
	}
	return nil
}
