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
	"fmt"
	"sort"

	"github.com/boltdb/bolt"
	"github.com/heketi/heketi/executors"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/lpabon/godbc"
)

const (

	// Byte values in KB
	KB = 1
	MB = KB * 1024
	GB = MB * 1024
	TB = GB * 1024

	// Default values
	DEFAULT_REPLICA               = 2
	DEFAULT_EC_DATA               = 4
	DEFAULT_EC_REDUNDANCY         = 2
	DEFAULT_THINP_SNAPSHOT_FACTOR = 1.5
)

type VolumeEntry struct {
	Info                 api.VolumeInfo
	Bricks               sort.StringSlice
	Durability           VolumeDurability
	GlusterVolumeOptions []string
}

func VolumeList(tx *bolt.Tx) ([]string, error) {

	list := EntryKeys(tx, BOLTDB_BUCKET_VOLUME)
	if list == nil {
		return nil, ErrAccessList
	}
	return list, nil
}

func NewVolumeEntry() *VolumeEntry {
	entry := &VolumeEntry{}
	entry.Bricks = make(sort.StringSlice, 0)

	gob.Register(&NoneDurability{})
	gob.Register(&VolumeReplicaDurability{})
	gob.Register(&VolumeDisperseDurability{})

	return entry
}

func NewVolumeEntryFromRequest(req *api.VolumeCreateRequest) *VolumeEntry {
	godbc.Require(req != nil)

	vol := NewVolumeEntry()
	vol.Info.Gid = req.Gid
	vol.Info.Id = utils.GenUUID()
	vol.Info.Durability = req.Durability
	vol.Info.Snapshot = req.Snapshot
	vol.Info.Size = req.Size

	// Set default durability values
	durability := vol.Info.Durability.Type
	switch {

	case durability == api.DurabilityReplicate:
		logger.Debug("[%v] Replica %v",
			vol.Info.Id,
			vol.Info.Durability.Replicate.Replica)
		vol.Durability = NewVolumeReplicaDurability(&vol.Info.Durability.Replicate)

	case durability == api.DurabilityEC:
		logger.Debug("[%v] EC %v + %v ",
			vol.Info.Id,
			vol.Info.Durability.Disperse.Data,
			vol.Info.Durability.Disperse.Redundancy)
		vol.Durability = NewVolumeDisperseDurability(&vol.Info.Durability.Disperse)

	case durability == api.DurabilityDistributeOnly || durability == "":
		logger.Debug("[%v] Distributed", vol.Info.Id)
		vol.Durability = NewNoneDurability()

	default:
		panic(fmt.Sprintf("BUG: Unknown type: %v\n", vol.Info.Durability))
	}

	// Set the default values accordingly
	vol.Durability.SetDurability()

	// Set default name
	if req.Name == "" {
		vol.Info.Name = "vol_" + vol.Info.Id
	} else {
		vol.Info.Name = req.Name
	}

	// Set default thinp factor
	if vol.Info.Snapshot.Enable && vol.Info.Snapshot.Factor == 0 {
		vol.Info.Snapshot.Factor = DEFAULT_THINP_SNAPSHOT_FACTOR
	} else if !vol.Info.Snapshot.Enable {
		vol.Info.Snapshot.Factor = 1
	}

	// If it is zero, then no volume options are set.
	vol.GlusterVolumeOptions = req.GlusterVolumeOptions

	// If it is zero, then it will be assigned during volume creation
	vol.Info.Clusters = req.Clusters

	return vol
}

func NewVolumeEntryFromId(tx *bolt.Tx, id string) (*VolumeEntry, error) {
	godbc.Require(tx != nil)

	entry := NewVolumeEntry()
	err := EntryLoad(tx, entry, id)
	if err != nil {
		return nil, err
	}

	return entry, nil
}

func (v *VolumeEntry) BucketName() string {
	return BOLTDB_BUCKET_VOLUME
}

func (v *VolumeEntry) Save(tx *bolt.Tx) error {
	godbc.Require(tx != nil)
	godbc.Require(len(v.Info.Id) > 0)

	return EntrySave(tx, v, v.Info.Id)
}

func (v *VolumeEntry) Delete(tx *bolt.Tx) error {
	return EntryDelete(tx, v, v.Info.Id)
}

func (v *VolumeEntry) NewInfoResponse(tx *bolt.Tx) (*api.VolumeInfoResponse, error) {
	godbc.Require(tx != nil)

	info := api.NewVolumeInfoResponse()
	info.Id = v.Info.Id
	info.Cluster = v.Info.Cluster
	info.Mount = v.Info.Mount
	info.Snapshot = v.Info.Snapshot
	info.Size = v.Info.Size
	info.Durability = v.Info.Durability
	info.Name = v.Info.Name
	info.GlusterVolumeOptions = v.GlusterVolumeOptions

	for _, brickid := range v.BricksIds() {
		brick, err := NewBrickEntryFromId(tx, brickid)
		if err != nil {
			return nil, err
		}
		brickinfo, err := brick.NewInfoResponse(tx)
		if err != nil {
			return nil, err
		}

		info.Bricks = append(info.Bricks, *brickinfo)
	}

	return info, nil
}

func (v *VolumeEntry) Marshal() ([]byte, error) {
	var buffer bytes.Buffer
	enc := gob.NewEncoder(&buffer)
	err := enc.Encode(*v)

	return buffer.Bytes(), err
}

func (v *VolumeEntry) Unmarshal(buffer []byte) error {
	dec := gob.NewDecoder(bytes.NewReader(buffer))
	err := dec.Decode(v)
	if err != nil {
		return err
	}

	// Make sure to setup arrays if nil
	if v.Bricks == nil {
		v.Bricks = make(sort.StringSlice, 0)
	}

	return nil
}

func (v *VolumeEntry) BrickAdd(id string) {
	godbc.Require(!utils.SortedStringHas(v.Bricks, id))

	v.Bricks = append(v.Bricks, id)
	v.Bricks.Sort()
}

func (v *VolumeEntry) BrickDelete(id string) {
	v.Bricks = utils.SortedStringsDelete(v.Bricks, id)
}

func (v *VolumeEntry) Create(db *bolt.DB,
	executor executors.Executor,
	allocator Allocator) (e error) {

	// On any error, remove the volume
	defer func() {
		if e != nil {
			db.Update(func(tx *bolt.Tx) error {
				v.Delete(tx)

				return nil
			})
		}
	}()

	// Get list of clusters
	var possibleClusters []string
	if len(v.Info.Clusters) == 0 {
		err := db.View(func(tx *bolt.Tx) error {
			var err error
			possibleClusters, err = ClusterList(tx)
			return err
		})
		if err != nil {
			return err
		}
	} else {
		possibleClusters = v.Info.Clusters
	}

	// Check we have clusters
	if len(possibleClusters) == 0 {
		logger.LogError("Volume being ask to be created, but there are no clusters configured")
		return ErrNoSpace
	}
	logger.Debug("Using the following clusters: %+v", possibleClusters)

	// Check for volume name conflict on any cluster
	var clusters []string
	for _, cluster := range possibleClusters {
		var err error

		// Check this cluster does not have a volume with the name
		err = db.View(func(tx *bolt.Tx) error {
			ce, err := NewClusterEntryFromId(tx, cluster)
			if err != nil {
				return err
			}

			for _, volumeId := range ce.Info.Volumes {
				volume, err := NewVolumeEntryFromId(tx, volumeId)
				if err != nil {
					return err
				}
				if v.Info.Name == volume.Info.Name {
					return fmt.Errorf("Name %v already in use in cluster %v",
						v.Info.Name, cluster)
				}
			}

			return nil

		})
		if err != nil {
			logger.Warning("%v", err.Error())
		} else {
			clusters = append(clusters, cluster)
		}
	}
	if len(clusters) == 0 {
		return fmt.Errorf("Name %v is already in use in all available clusters", v.Info.Name)
	}

	// For each cluster look for storage space for this volume
	var brick_entries []*BrickEntry
	var err error
	for _, cluster := range clusters {

		// Check this cluster for space
		brick_entries, err = v.allocBricksInCluster(db, allocator, cluster, v.Info.Size)

		if err == nil {
			v.Info.Cluster = cluster
			logger.Debug("Volume to be created on cluster %v", cluster)
			break
		} else if err == ErrNoSpace ||
			err == ErrMaxBricks ||
			err == ErrMinimumBrickSize {
			logger.Debug("Cluster %v can not accommodate volume "+
				"(%v), trying next cluster", cluster, err)
			continue
		} else {
			// A genuine error occurred - bail out
			return logger.LogError("Error calling v.allocBricksInCluster: %v", err)
		}
	}

	if err != nil || brick_entries == nil {
		// Map all 'valid' errors to NoSpace here:
		// Only the last such error could get propagated down,
		// so it does not make sense to hand the granularity on.
		// But for other callers (Expand), we keep it.
		return ErrNoSpace
	}

	// Make sure to clean up bricks on error
	defer func() {
		if e != nil {
			db.Update(func(tx *bolt.Tx) error {
				for _, brick := range brick_entries {
					v.removeBrickFromDb(tx, brick)
				}
				return nil
			})
		}
	}()

	// Create the bricks on the nodes
	err = CreateBricks(db, executor, brick_entries)
	if err != nil {
		return err
	}

	// Clean up created bricks on failure
	defer func() {
		if e != nil {
			DestroyBricks(db, executor, brick_entries)
		}
	}()

	// Create GlusterFS volume
	err = v.createVolume(db, executor, brick_entries)
	if err != nil {
		return err
	}

	// Destroy volume on failure
	defer func() {
		if e != nil {
			v.Destroy(db, executor)
		}
	}()

	// Save information on db
	err = db.Update(func(tx *bolt.Tx) error {

		// Save brick entries
		for _, brick := range brick_entries {
			err := brick.Save(tx)
			if err != nil {
				return err
			}
		}

		// Save volume information
		err = v.Save(tx)
		if err != nil {
			return err
		}

		// Save cluster
		cluster, err := NewClusterEntryFromId(tx, v.Info.Cluster)
		if err != nil {
			return err
		}
		cluster.VolumeAdd(v.Info.Id)
		return cluster.Save(tx)
	})
	if err != nil {
		return err
	}

	return nil

}

func (v *VolumeEntry) Destroy(db *bolt.DB, executor executors.Executor) error {
	logger.Info("Destroying volume %v", v.Info.Id)

	// Get the entries from the database
	brick_entries := make([]*BrickEntry, len(v.Bricks))
	var sshhost string
	db.View(func(tx *bolt.Tx) error {
		for index, id := range v.BricksIds() {
			brick, err := NewBrickEntryFromId(tx, id)
			if err != nil {
				logger.LogError("Brick %v not found in db: %v", id, err)
				continue
			}
			brick_entries[index] = brick

			// Set ssh host to send volume commands
			if sshhost == "" {
				node, err := NewNodeEntryFromId(tx, brick.Info.NodeId)
				if err != nil {
					logger.LogError("Unable to determine brick node: %v", err)
					return err
				}
				sshhost = node.ManageHostName()
			}
		}
		return nil
	})

	// Determine if we can destroy the volume
	err := executor.VolumeDestroyCheck(sshhost, v.Info.Name)
	if err != nil {
		logger.Err(err)
		return err
	}

	// Determine if the bricks can be destroyed
	err = v.checkBricksCanBeDestroyed(db, executor, brick_entries)
	if err != nil {
		logger.Err(err)
		return err
	}

	// :TODO: What if the host is no longer available, we may need to try others
	// Stop volume
	err = executor.VolumeDestroy(sshhost, v.Info.Name)
	if err != nil {
		logger.LogError("Unable to delete volume: %v", err)
		return err
	}

	// Destroy bricks
	err = DestroyBricks(db, executor, brick_entries)
	if err != nil {
		logger.LogError("Unable to delete bricks: %v", err)
		return err
	}

	// Remove from entries from the db
	err = db.Update(func(tx *bolt.Tx) error {
		for _, brick := range brick_entries {
			err = v.removeBrickFromDb(tx, brick)
			if err != nil {
				logger.Err(err)
				// Everything is destroyed anyways, just keep deleting the others
				// Do not return here
			}
		}

		// Remove volume from cluster
		cluster, err := NewClusterEntryFromId(tx, v.Info.Cluster)
		if err != nil {
			logger.Err(err)
			// Do not return here.. keep going
		}
		cluster.VolumeDelete(v.Info.Id)

		err = cluster.Save(tx)
		if err != nil {
			logger.Err(err)
			// Do not return here.. keep going
		}

		// Delete volume
		v.Delete(tx)

		return nil
	})

	return err
}

func (v *VolumeEntry) Expand(db *bolt.DB,
	executor executors.Executor,
	allocator Allocator,
	sizeGB int) (e error) {

	// Allocate new bricks in the cluster
	brick_entries, err := v.allocBricksInCluster(db, allocator, v.Info.Cluster, sizeGB)
	if err != nil {
		return err
	}

	// Setup cleanup function
	defer func() {
		if e != nil {
			logger.Debug("Error detected, cleaning up")

			// Remove from db
			db.Update(func(tx *bolt.Tx) error {
				for _, brick := range brick_entries {
					v.removeBrickFromDb(tx, brick)
				}
				err := v.Save(tx)
				godbc.Check(err == nil)

				return nil
			})
		}
	}()

	// Create bricks
	err = CreateBricks(db, executor, brick_entries)
	if err != nil {
		return err
	}

	// Setup cleanup function
	defer func() {
		if e != nil {
			logger.Debug("Error detected, cleaning up")
			DestroyBricks(db, executor, brick_entries)
		}
	}()

	// Create a volume request to send to executor
	// so that it can add the new bricks
	vr, host, err := v.createVolumeRequest(db, brick_entries)
	if err != nil {
		return err
	}

	// Expand the volume
	_, err = executor.VolumeExpand(host, vr)
	if err != nil {
		return err
	}

	// Increase the recorded volume size
	v.Info.Size += sizeGB

	// Save volume entry
	err = db.Update(func(tx *bolt.Tx) error {

		// Save brick entries
		for _, brick := range brick_entries {
			err := brick.Save(tx)
			if err != nil {
				return err
			}
		}

		return v.Save(tx)
	})

	return err

}

func (v *VolumeEntry) BricksIds() sort.StringSlice {
	ids := make(sort.StringSlice, len(v.Bricks))
	copy(ids, v.Bricks)
	return ids
}

func (v *VolumeEntry) checkBricksCanBeDestroyed(db *bolt.DB,
	executor executors.Executor,
	brick_entries []*BrickEntry) error {

	sg := utils.NewStatusGroup()

	// Create a goroutine for each brick
	for _, brick := range brick_entries {
		sg.Add(1)
		go func(b *BrickEntry) {
			defer sg.Done()
			sg.Err(b.DestroyCheck(db, executor))
		}(brick)
	}

	// Wait here until all goroutines have returned.  If
	// any of errored, it would be cought here
	err := sg.Result()
	if err != nil {
		logger.Err(err)
	}
	return err
}

func VolumeEntryUpgrade(tx *bolt.Tx) error {
	return nil
}
