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
	"errors"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/boltdb/bolt"
	"github.com/heketi/heketi/executors"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
)

func createSampleVolumeEntry(size int) *VolumeEntry {
	req := &api.VolumeCreateRequest{}
	req.Size = size
	req.Durability.Type = api.DurabilityReplicate
	req.Durability.Replicate.Replica = 2

	v := NewVolumeEntryFromRequest(req)

	return v
}

func setupSampleDbWithTopology(app *App,
	clusters, nodes_per_cluster, devices_per_node int,
	disksize uint64) error {

	var clusterlist []string
	err := app.db.Update(func(tx *bolt.Tx) error {
		for c := 0; c < clusters; c++ {
			cluster := createSampleClusterEntry()

			for n := 0; n < nodes_per_cluster; n++ {
				node := createSampleNodeEntry()
				node.Info.ClusterId = cluster.Info.Id
				node.Info.Zone = n % 2

				cluster.NodeAdd(node.Info.Id)

				for d := 0; d < devices_per_node; d++ {
					device := createSampleDeviceEntry(node.Info.Id, disksize)
					node.DeviceAdd(device.Id())

					// Update allocator
					err := app.allocator.AddDevice(cluster, node, device)
					if err != nil {
						return nil
					}

					err = device.Save(tx)
					if err != nil {
						return err
					}
				}
				err := node.Save(tx)
				if err != nil {
					return err
				}
			}
			err := cluster.Save(tx)
			if err != nil {
				return err
			}
		}

		var err error
		clusterlist, err = ClusterList(tx)
		if err != nil {
			return err
		}

		return nil
	})
	if err != nil {
		return nil
	}

	return nil
}

func TestNewVolumeEntry(t *testing.T) {
	v := NewVolumeEntry()

	tests.Assert(t, v.Bricks != nil)
	tests.Assert(t, len(v.Info.Id) == 0)
	tests.Assert(t, len(v.Info.Cluster) == 0)
	tests.Assert(t, len(v.Info.Clusters) == 0)
}

func TestNewVolumeEntryFromRequestOnlySize(t *testing.T) {

	req := &api.VolumeCreateRequest{}
	req.Size = 1024

	v := NewVolumeEntryFromRequest(req)
	tests.Assert(t, v.Info.Name == "vol_"+v.Info.Id)
	tests.Assert(t, len(v.Info.Clusters) == 0)
	tests.Assert(t, v.Info.Snapshot.Enable == false)
	tests.Assert(t, v.Info.Snapshot.Factor == 1)
	tests.Assert(t, v.Info.Size == 1024)
	tests.Assert(t, v.Info.Cluster == "")
	tests.Assert(t, len(v.Info.Id) != 0)
	tests.Assert(t, len(v.Bricks) == 0)
	tests.Assert(t, v.Info.Durability.Type == "")
}

func TestNewVolumeEntryFromRequestReplicaDefault(t *testing.T) {
	req := &api.VolumeCreateRequest{}
	req.Size = 1024
	req.Durability.Type = api.DurabilityReplicate

	v := NewVolumeEntryFromRequest(req)
	tests.Assert(t, v.Info.Name == "vol_"+v.Info.Id)
	tests.Assert(t, len(v.Info.Clusters) == 0)
	tests.Assert(t, v.Info.Snapshot.Enable == false)
	tests.Assert(t, v.Info.Snapshot.Factor == 1)
	tests.Assert(t, v.Info.Size == 1024)
	tests.Assert(t, v.Info.Cluster == "")
	tests.Assert(t, len(v.Info.Id) != 0)
	tests.Assert(t, len(v.Bricks) == 0)
	tests.Assert(t, v.Info.Durability.Type == api.DurabilityReplicate)
	tests.Assert(t, v.Info.Durability.Replicate.Replica == 0)
}

func TestNewVolumeEntryFromRequestReplica5(t *testing.T) {
	req := &api.VolumeCreateRequest{}
	req.Size = 1024
	req.Durability.Type = api.DurabilityReplicate
	req.Durability.Replicate.Replica = 5

	v := NewVolumeEntryFromRequest(req)
	tests.Assert(t, v.Info.Name == "vol_"+v.Info.Id)
	tests.Assert(t, len(v.Info.Clusters) == 0)
	tests.Assert(t, v.Info.Snapshot.Enable == false)
	tests.Assert(t, v.Info.Snapshot.Factor == 1)
	tests.Assert(t, v.Info.Size == 1024)
	tests.Assert(t, v.Info.Cluster == "")
	tests.Assert(t, len(v.Info.Id) != 0)
	tests.Assert(t, len(v.Bricks) == 0)
	tests.Assert(t, v.Info.Durability.Type == api.DurabilityReplicate)
	tests.Assert(t, v.Info.Durability.Replicate.Replica == 5)
}

func TestNewVolumeEntryFromRequestDistribute(t *testing.T) {
	req := &api.VolumeCreateRequest{}
	req.Size = 1024
	req.Durability.Type = api.DurabilityDistributeOnly

	v := NewVolumeEntryFromRequest(req)
	tests.Assert(t, v.Info.Name == "vol_"+v.Info.Id)
	tests.Assert(t, len(v.Info.Clusters) == 0)
	tests.Assert(t, v.Info.Snapshot.Enable == false)
	tests.Assert(t, v.Info.Snapshot.Factor == 1)
	tests.Assert(t, v.Info.Size == 1024)
	tests.Assert(t, v.Info.Cluster == "")
	tests.Assert(t, len(v.Info.Id) != 0)
	tests.Assert(t, len(v.Bricks) == 0)
	tests.Assert(t, v.Info.Durability.Type == api.DurabilityDistributeOnly)
}

func TestNewVolumeEntryFromRequestDisperseDefault(t *testing.T) {
	req := &api.VolumeCreateRequest{}
	req.Size = 1024
	req.Durability.Type = api.DurabilityEC

	v := NewVolumeEntryFromRequest(req)
	tests.Assert(t, v.Info.Name == "vol_"+v.Info.Id)
	tests.Assert(t, len(v.Info.Clusters) == 0)
	tests.Assert(t, v.Info.Snapshot.Enable == false)
	tests.Assert(t, v.Info.Snapshot.Factor == 1)
	tests.Assert(t, v.Info.Size == 1024)
	tests.Assert(t, v.Info.Cluster == "")
	tests.Assert(t, len(v.Info.Id) != 0)
	tests.Assert(t, len(v.Bricks) == 0)
	tests.Assert(t, v.Info.Durability.Type == api.DurabilityEC)
	tests.Assert(t, v.Info.Durability.Disperse.Data == 0)
	tests.Assert(t, v.Info.Durability.Disperse.Redundancy == 0)
}

func TestNewVolumeEntryFromRequestDisperseDefault48(t *testing.T) {
	req := &api.VolumeCreateRequest{}
	req.Size = 1024
	req.Durability.Type = api.DurabilityEC
	req.Durability.Disperse.Data = 8
	req.Durability.Disperse.Redundancy = 4

	v := NewVolumeEntryFromRequest(req)
	tests.Assert(t, v.Info.Name == "vol_"+v.Info.Id)
	tests.Assert(t, len(v.Info.Clusters) == 0)
	tests.Assert(t, v.Info.Snapshot.Enable == false)
	tests.Assert(t, v.Info.Snapshot.Factor == 1)
	tests.Assert(t, v.Info.Size == 1024)
	tests.Assert(t, v.Info.Cluster == "")
	tests.Assert(t, len(v.Info.Id) != 0)
	tests.Assert(t, len(v.Bricks) == 0)
	tests.Assert(t, v.Info.Durability.Type == api.DurabilityEC)
	tests.Assert(t, v.Info.Durability.Disperse.Data == 8)
	tests.Assert(t, v.Info.Durability.Disperse.Redundancy == 4)
}

func TestNewVolumeEntryFromRequestClusters(t *testing.T) {

	req := &api.VolumeCreateRequest{}
	req.Size = 1024
	req.Clusters = []string{"abc", "def"}

	v := NewVolumeEntryFromRequest(req)
	tests.Assert(t, v.Info.Name == "vol_"+v.Info.Id)
	tests.Assert(t, v.Info.Snapshot.Enable == false)
	tests.Assert(t, v.Info.Snapshot.Factor == 1)
	tests.Assert(t, v.Info.Size == 1024)
	tests.Assert(t, reflect.DeepEqual(req.Clusters, v.Info.Clusters))
	tests.Assert(t, len(v.Info.Id) != 0)
	tests.Assert(t, len(v.Bricks) == 0)

}

func TestNewVolumeEntryFromRequestSnapshotEnabledDefaultFactor(t *testing.T) {

	req := &api.VolumeCreateRequest{}
	req.Size = 1024
	req.Clusters = []string{"abc", "def"}
	req.Snapshot.Enable = true

	v := NewVolumeEntryFromRequest(req)
	tests.Assert(t, v.Info.Name == "vol_"+v.Info.Id)
	tests.Assert(t, v.Info.Snapshot.Enable == true)
	tests.Assert(t, v.Info.Snapshot.Factor == DEFAULT_THINP_SNAPSHOT_FACTOR)
	tests.Assert(t, v.Info.Size == 1024)
	tests.Assert(t, reflect.DeepEqual(req.Clusters, v.Info.Clusters))
	tests.Assert(t, len(v.Info.Id) != 0)
	tests.Assert(t, len(v.Bricks) == 0)

}

func TestNewVolumeEntryFromRequestSnapshotFactor(t *testing.T) {

	req := &api.VolumeCreateRequest{}
	req.Size = 1024
	req.Clusters = []string{"abc", "def"}
	req.Snapshot.Enable = true
	req.Snapshot.Factor = 1.3

	v := NewVolumeEntryFromRequest(req)
	tests.Assert(t, v.Info.Name == "vol_"+v.Info.Id)
	tests.Assert(t, v.Info.Snapshot.Enable == true)
	tests.Assert(t, v.Info.Snapshot.Factor == 1.3)
	tests.Assert(t, v.Info.Size == 1024)
	tests.Assert(t, reflect.DeepEqual(req.Clusters, v.Info.Clusters))
	tests.Assert(t, len(v.Info.Id) != 0)
	tests.Assert(t, len(v.Bricks) == 0)

}

func TestNewVolumeEntryFromRequestName(t *testing.T) {

	req := &api.VolumeCreateRequest{}
	req.Size = 1024
	req.Clusters = []string{"abc", "def"}
	req.Snapshot.Enable = true
	req.Snapshot.Factor = 1.3
	req.Name = "myvol"

	v := NewVolumeEntryFromRequest(req)
	tests.Assert(t, v.Info.Name == "myvol")
	tests.Assert(t, v.Info.Snapshot.Enable == true)
	tests.Assert(t, v.Info.Snapshot.Factor == 1.3)
	tests.Assert(t, v.Info.Size == 1024)
	tests.Assert(t, reflect.DeepEqual(req.Clusters, v.Info.Clusters))
	tests.Assert(t, len(v.Info.Id) != 0)
	tests.Assert(t, len(v.Bricks) == 0)

}

func TestNewVolumeEntryMarshal(t *testing.T) {

	req := &api.VolumeCreateRequest{}
	req.Size = 1024
	req.Clusters = []string{"abc", "def"}
	req.Snapshot.Enable = true
	req.Snapshot.Factor = 1.3
	req.Name = "myvol"

	v := NewVolumeEntryFromRequest(req)
	v.BrickAdd("abc")
	v.BrickAdd("def")

	buffer, err := v.Marshal()
	tests.Assert(t, err == nil)
	tests.Assert(t, buffer != nil)
	tests.Assert(t, len(buffer) > 0)

	um := &VolumeEntry{}
	err = um.Unmarshal(buffer)
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(v, um))

}

func TestVolumeEntryAddDeleteDevices(t *testing.T) {

	v := NewVolumeEntry()
	tests.Assert(t, len(v.Bricks) == 0)

	v.BrickAdd("123")
	tests.Assert(t, utils.SortedStringHas(v.Bricks, "123"))
	tests.Assert(t, len(v.Bricks) == 1)
	v.BrickAdd("abc")
	tests.Assert(t, utils.SortedStringHas(v.Bricks, "123"))
	tests.Assert(t, utils.SortedStringHas(v.Bricks, "abc"))
	tests.Assert(t, len(v.Bricks) == 2)

	v.BrickDelete("123")
	tests.Assert(t, !utils.SortedStringHas(v.Bricks, "123"))
	tests.Assert(t, utils.SortedStringHas(v.Bricks, "abc"))
	tests.Assert(t, len(v.Bricks) == 1)

	v.BrickDelete("ccc")
	tests.Assert(t, !utils.SortedStringHas(v.Bricks, "123"))
	tests.Assert(t, utils.SortedStringHas(v.Bricks, "abc"))
	tests.Assert(t, len(v.Bricks) == 1)
}

func TestVolumeEntryFromIdNotFound(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Test for ID not found
	err := app.db.View(func(tx *bolt.Tx) error {
		_, err := NewVolumeEntryFromId(tx, "123")
		return err
	})
	tests.Assert(t, err == ErrNotFound)

}

func TestVolumeEntryFromId(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a volume entry
	v := createSampleVolumeEntry(1024)

	// Save in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return v.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Load from database
	var entry *VolumeEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		entry, err = NewVolumeEntryFromId(tx, v.Info.Id)
		return err
	})
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(entry, v))

}

func TestVolumeEntrySaveDelete(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a volume entry
	v := createSampleVolumeEntry(1024)

	// Save in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return v.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Delete entry which has devices
	var entry *VolumeEntry
	err = app.db.Update(func(tx *bolt.Tx) error {
		var err error
		entry, err = NewVolumeEntryFromId(tx, v.Info.Id)
		if err != nil {
			return err
		}

		err = entry.Delete(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)

	// Check volume has been deleted and is not in db
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		entry, err = NewVolumeEntryFromId(tx, v.Info.Id)
		if err != nil {
			return err
		}
		return nil

	})
	tests.Assert(t, err == ErrNotFound)
}

func TestNewVolumeEntryNewInfoResponse(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a volume entry
	v := createSampleVolumeEntry(1024)

	// Save in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return v.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Retrieve info response
	var info *api.VolumeInfoResponse
	err = app.db.View(func(tx *bolt.Tx) error {
		volume, err := NewVolumeEntryFromId(tx, v.Info.Id)
		if err != nil {
			return err
		}

		info, err = volume.NewInfoResponse(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil, err)

	tests.Assert(t, info.Cluster == v.Info.Cluster)
	tests.Assert(t, reflect.DeepEqual(info.Snapshot, v.Info.Snapshot))
	tests.Assert(t, info.Name == v.Info.Name)
	tests.Assert(t, info.Id == v.Info.Id)
	tests.Assert(t, info.Size == v.Info.Size)
	tests.Assert(t, len(info.Bricks) == 0)
}

func TestVolumeEntryCreateMissingCluster(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a volume entry
	v := createSampleVolumeEntry(1024)
	v.Info.Clusters = []string{}

	// Save in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return v.Save(tx)
	})
	tests.Assert(t, err == nil)

	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == ErrNoSpace)

}

func TestVolumeEntryCreateRunOutOfSpaceMinBrickSizeLimit(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Total 80GB
	err := setupSampleDbWithTopology(app,
		1,     // clusters
		2,     // nodes_per_cluster
		4,     // devices_per_node,
		10*GB, // disksize, 10G)
	)
	tests.Assert(t, err == nil)

	// Create a 100 GB volume
	// Shouldn't be able to break it down enough to allocate volume
	v := createSampleVolumeEntry(100)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == ErrNoSpace)
	tests.Assert(t, v.Info.Cluster == "")

	// Check database volume does not exist
	err = app.db.View(func(tx *bolt.Tx) error {
		_, err := NewVolumeEntryFromId(tx, v.Info.Id)
		return err
	})
	tests.Assert(t, err == ErrNotFound)

	// Check no bricks or volumes exist
	var bricks []string
	var volumes []string
	err = app.db.View(func(tx *bolt.Tx) error {
		bricks = EntryKeys(tx, BOLTDB_BUCKET_BRICK)
		volumes = EntryKeys(tx, BOLTDB_BUCKET_VOLUME)

		return nil
	})
	tests.Assert(t, err == nil)
	tests.Assert(t, len(bricks) == 0, bricks)
	tests.Assert(t, len(volumes) == 0)

}

func TestVolumeEntryCreateRunOutOfSpaceMaxBrickLimit(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Lots of nodes with little drives
	err := setupSampleDbWithTopology(app,
		1,  // clusters
		20, // nodes_per_cluster
		40, // devices_per_node,

		// Must be larger than the brick min size
		BrickMinSize*2, // disksize
	)
	tests.Assert(t, err == nil)

	// Create a volume who will be broken down to
	// Shouldn't be able to break it down enough to allocate volume
	v := createSampleVolumeEntry(BrickMaxNum * 2 * int(BrickMinSize/GB))
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == ErrNoSpace)

	// Check database volume does not exist
	err = app.db.View(func(tx *bolt.Tx) error {
		_, err := NewVolumeEntryFromId(tx, v.Info.Id)
		return err
	})
	tests.Assert(t, err == ErrNotFound)

	// Check no bricks or volumes exist
	var bricks []string
	var volumes []string
	err = app.db.View(func(tx *bolt.Tx) error {
		bricks = EntryKeys(tx, BOLTDB_BUCKET_BRICK)

		volumes = EntryKeys(tx, BOLTDB_BUCKET_VOLUME)
		return nil
	})
	tests.Assert(t, len(bricks) == 0)
	tests.Assert(t, len(volumes) == 0)

}

func TestVolumeEntryCreateTwoBricks(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a cluster in the database
	err := setupSampleDbWithTopology(app,
		1,      // clusters
		4,      // nodes_per_cluster
		4,      // devices_per_node,
		500*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Mock Brick creation and check it was called
	brickCreateCount := 0
	gid := int64(1000)
	var mutex sync.Mutex
	app.xo.MockBrickCreate = func(host string,
		brick *executors.BrickRequest) (*executors.BrickInfo, error) {

		mutex.Lock()
		brickCreateCount++
		mutex.Unlock()

		bInfo := &executors.BrickInfo{
			Path: "/mockpath",
		}

		tests.Assert(t, brick.Gid == gid)
		return bInfo, nil
	}

	// Create a volume who will be broken down to
	v := createSampleVolumeEntry(250)

	// Set a GID
	v.Info.Gid = gid

	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil, err)
	tests.Assert(t, brickCreateCount == 2)

	// Check database
	var info *api.VolumeInfoResponse
	var nodelist sort.StringSlice
	err = app.db.View(func(tx *bolt.Tx) error {
		entry, err := NewVolumeEntryFromId(tx, v.Info.Id)
		if err != nil {
			return err
		}

		info, err = entry.NewInfoResponse(tx)
		if err != nil {
			return err
		}

		cluster, err := NewClusterEntryFromId(tx, v.Info.Cluster)
		if err != nil {
			return err
		}
		nodelist = make(sort.StringSlice, len(cluster.Info.Nodes))

		for i, id := range cluster.Info.Nodes {
			node, err := NewNodeEntryFromId(tx, id)
			if err != nil {
				return err
			}
			nodelist[i] = node.StorageHostName()
		}
		nodelist.Sort()

		return nil

	})
	tests.Assert(t, err == nil)

	// Check that it used only two bricks each with only two replicas
	tests.Assert(t, len(info.Bricks) == 2)
	tests.Assert(t, info.Bricks[0].Size == info.Bricks[1].Size)
	tests.Assert(t, info.Cluster == v.Info.Cluster)

	// Check information on the bricks
	for _, brick := range info.Bricks {
		tests.Assert(t, brick.DeviceId != "")
		tests.Assert(t, brick.NodeId != "")
		tests.Assert(t, brick.Path != "")
	}

	// Check mount information
	host := strings.Split(info.Mount.GlusterFS.MountPoint, ":")[0]
	tests.Assert(t, utils.SortedStringHas(nodelist, host), host, nodelist)
	volfileServers := strings.Split(info.Mount.GlusterFS.Options["backup-volfile-servers"], ",")
	for index, node := range volfileServers {
		tests.Assert(t, node != host, index, node, host)
	}

	// Should have at least the number nodes as replicas
	tests.Assert(t, len(info.Mount.GlusterFS.Hosts) >= info.Durability.Replicate.Replica,
		info.Mount.GlusterFS.Hosts,
		info)

	// Check all hosts are in the list
	err = app.db.View(func(tx *bolt.Tx) error {
		for _, brick := range info.Bricks {
			found := false

			node, err := NewNodeEntryFromId(tx, brick.NodeId)
			tests.Assert(t, err == nil)

			for _, host := range info.Mount.GlusterFS.Hosts {
				if host == node.StorageHostName() {
					found = true
					break
				}
			}
			tests.Assert(t, found, node.StorageHostName(),
				info.Mount.GlusterFS.Hosts)
		}

		return nil
	})

}

func TestVolumeEntryCreateBrickDivision(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create 50TB of storage
	err := setupSampleDbWithTopology(app,
		1,      // clusters
		10,     // nodes_per_cluster
		10,     // devices_per_node,
		500*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create a volume which is so big that it does
	// not fit into a single replica set
	v := createSampleVolumeEntry(2000)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	var info *api.VolumeInfoResponse
	var nodelist sort.StringSlice
	err = app.db.View(func(tx *bolt.Tx) error {
		entry, err := NewVolumeEntryFromId(tx, v.Info.Id)
		if err != nil {
			return err
		}

		info, err = entry.NewInfoResponse(tx)
		if err != nil {
			return err
		}

		cluster, err := NewClusterEntryFromId(tx, v.Info.Cluster)
		if err != nil {
			return err
		}
		nodelist = make(sort.StringSlice, len(cluster.Info.Nodes))

		for i, id := range cluster.Info.Nodes {
			node, err := NewNodeEntryFromId(tx, id)
			if err != nil {
				return err
			}
			nodelist[i] = node.StorageHostName()
		}
		nodelist.Sort()

		return nil

	})
	tests.Assert(t, err == nil)

	// Will need 3 splits for a total of 8 bricks + replicas
	//
	// NOTE: Why 8 bricksets of 250GB instead of 4 bricksets
	// of 500GB each? Because the disk needed for hosting
	// a brick if size X is a little larger than X, because
	// we additionally allocate at least some space for metadata.
	// Hence we will end up using 250GB bricks, and no two bricks
	// will be on the same disk. Requesting slightly less than
	// 2000GB, e.g. 1940GB yields a four 485GB bricksets.
	//
	tests.Assert(t, len(info.Bricks) == 16)
	for b := 1; b < 16; b++ {
		tests.Assert(t, 250*GB == info.Bricks[b].Size, b)
	}
	tests.Assert(t, info.Cluster == v.Info.Cluster)

	// Check mount information
	host := strings.Split(info.Mount.GlusterFS.MountPoint, ":")[0]
	tests.Assert(t, utils.SortedStringHas(nodelist, host), host, nodelist)
	volfileServers := strings.Split(info.Mount.GlusterFS.Options["backup-volfile-servers"], ",")
	for index, node := range volfileServers {
		tests.Assert(t, node != host, index, node, host)
	}

}

func TestVolumeEntryCreateMaxBrickSize(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create 500TB of storage
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		10,   // nodes_per_cluster
		10,   // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create a volume whose bricks must be at most BrickMaxSize
	v := createSampleVolumeEntry(int(BrickMaxSize / GB * 4))
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Get volume information
	var info *api.VolumeInfoResponse
	err = app.db.View(func(tx *bolt.Tx) error {
		entry, err := NewVolumeEntryFromId(tx, v.Info.Id)
		if err != nil {
			return err
		}

		info, err = entry.NewInfoResponse(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)

	// Check the size of the bricks are not bigger than BrickMaxSize
	tests.Assert(t, len(info.Bricks) == 8)
	for b := 1; b < len(info.Bricks); b++ {
		tests.Assert(t, info.Bricks[b].Size <= BrickMaxSize)
	}
	tests.Assert(t, info.Cluster == v.Info.Cluster)

}

func TestVolumeEntryCreateOnClustersRequested(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create 50TB of storage
	err := setupSampleDbWithTopology(app,
		10,   // clusters
		10,   // nodes_per_cluster
		10,   // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Get a cluster list
	var clusters sort.StringSlice
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		clusters, err = ClusterList(tx)
		return err
	})
	tests.Assert(t, err == nil)
	clusters.Sort()

	// Create a 1TB volume
	v := createSampleVolumeEntry(1024)

	// Set the clusters to the first two cluster ids
	v.Info.Clusters = []string{clusters[0]}

	// Create volume
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Check database volume does not exist
	var info *api.VolumeInfoResponse
	err = app.db.View(func(tx *bolt.Tx) error {
		entry, err := NewVolumeEntryFromId(tx, v.Info.Id)
		if err != nil {
			return err
		}

		info, err = entry.NewInfoResponse(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)
	tests.Assert(t, info.Cluster == clusters[0])

	// Create a new volume on either of three clusters
	clusterset := clusters[2:5]
	v = createSampleVolumeEntry(1024)
	v.Info.Clusters = clusterset
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Check database volume exists
	err = app.db.View(func(tx *bolt.Tx) error {
		entry, err := NewVolumeEntryFromId(tx, v.Info.Id)
		if err != nil {
			return err
		}

		info, err = entry.NewInfoResponse(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)
	tests.Assert(t, utils.SortedStringHas(clusterset, info.Cluster))

}

func TestVolumeEntryCreateCheckingClustersForSpace(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create 100 small clusters
	err := setupSampleDbWithTopology(app,
		10,    // clusters
		1,     // nodes_per_cluster
		1,     // devices_per_node,
		10*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create one large cluster
	cluster := createSampleClusterEntry()
	err = app.db.Update(func(tx *bolt.Tx) error {
		for n := 0; n < 100; n++ {
			node := createSampleNodeEntry()
			node.Info.ClusterId = cluster.Info.Id
			node.Info.Zone = n % 2

			cluster.NodeAdd(node.Info.Id)

			for d := 0; d < 10; d++ {
				device := createSampleDeviceEntry(node.Info.Id, 4*TB)
				node.DeviceAdd(device.Id())

				// update allocator
				err := app.allocator.AddDevice(cluster, node, device)
				if err != nil {
					return nil
				}

				// Save
				err = device.Save(tx)
				if err != nil {
					return err
				}
			}
			err := node.Save(tx)
			if err != nil {
				return err
			}
		}
		err := cluster.Save(tx)
		if err != nil {
			return err
		}

		return nil
	})

	// Create a 1TB volume
	v := createSampleVolumeEntry(1024)

	// Create volume
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Check database volume exists
	var info *api.VolumeInfoResponse
	err = app.db.View(func(tx *bolt.Tx) error {
		entry, err := NewVolumeEntryFromId(tx, v.Info.Id)
		if err != nil {
			return err
		}

		info, err = entry.NewInfoResponse(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)
	tests.Assert(t, info.Cluster == cluster.Info.Id)
}

func TestVolumeEntryCreateWithSnapshot(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Lots of nodes with little drives
	err := setupSampleDbWithTopology(app,
		1,      // clusters
		4,      // nodes_per_cluster
		4,      // devices_per_node,
		500*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create a volume with a snapshot factor of 1.5
	// For a 200G vol, it would get a brick size of 100G, with a thin pool
	// size of 100G * 1.5 = 150GB.
	v := createSampleVolumeEntry(200)
	v.Info.Snapshot.Enable = true
	v.Info.Snapshot.Factor = 1.5

	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Check database volume exists
	var info *api.VolumeInfoResponse
	err = app.db.View(func(tx *bolt.Tx) error {
		entry, err := NewVolumeEntryFromId(tx, v.Info.Id)
		if err != nil {
			return err
		}

		info, err = entry.NewInfoResponse(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)

	// Check that it used only two bricks each with only two replicas
	tests.Assert(t, len(info.Bricks) == 2)
	err = app.db.View(func(tx *bolt.Tx) error {
		for _, b := range info.Bricks {
			device, err := NewDeviceEntryFromId(tx, b.DeviceId)
			if err != nil {
				return err
			}

			tests.Assert(t, device.Info.Storage.Used >= uint64(1.5*float32(b.Size)))
		}

		return nil
	})
	tests.Assert(t, err == nil)
}

func TestVolumeEntryCreateBrickCreationFailure(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Lots of nodes with little drives
	err := setupSampleDbWithTopology(app,
		1,      // clusters
		4,      // nodes_per_cluster
		4,      // devices_per_node,
		500*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Cause a brick creation failure
	mockerror := errors.New("MOCK")
	app.xo.MockBrickCreate = func(host string, brick *executors.BrickRequest) (*executors.BrickInfo, error) {
		return nil, mockerror
	}

	// Create a volume with a snapshot factor of 1.5
	// For a 200G vol, it would get a brick size of 100G, with a thin pool
	// size of 100G * 1.5 = 150GB.
	v := createSampleVolumeEntry(200)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == mockerror, err, mockerror)

	// Check database is still clean. No bricks and No volumes
	err = app.db.View(func(tx *bolt.Tx) error {
		volumes, err := VolumeList(tx)
		tests.Assert(t, err == nil)
		tests.Assert(t, len(volumes) == 0)

		bricks, err := BrickList(tx)
		tests.Assert(t, err == nil)
		tests.Assert(t, len(bricks) == 0)

		clusters, err := ClusterList(tx)
		tests.Assert(t, err == nil)
		tests.Assert(t, len(clusters) == 1)

		cluster, err := NewClusterEntryFromId(tx, clusters[0])
		tests.Assert(t, err == nil)
		tests.Assert(t, len(cluster.Info.Volumes) == 0)

		return nil

	})
	tests.Assert(t, err == nil)
}

func TestVolumeEntryCreateVolumeCreationFailure(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Lots of nodes with little drives
	err := setupSampleDbWithTopology(app,
		1,      // clusters
		4,      // nodes_per_cluster
		4,      // devices_per_node,
		500*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Cause a brick creation failure
	mockerror := errors.New("MOCK")
	app.xo.MockVolumeCreate = func(host string, volume *executors.VolumeRequest) (*executors.Volume, error) {
		return nil, mockerror
	}

	// Create a volume with a snapshot factor of 1.5
	// For a 200G vol, it would get a brick size of 100G, with a thin pool
	// size of 100G * 1.5 = 150GB.
	v := createSampleVolumeEntry(200)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == mockerror)

	// Check database is still clean. No bricks and No volumes
	err = app.db.View(func(tx *bolt.Tx) error {
		volumes, err := VolumeList(tx)
		tests.Assert(t, err == nil)
		tests.Assert(t, len(volumes) == 0)

		bricks, err := BrickList(tx)
		tests.Assert(t, err == nil)
		tests.Assert(t, len(bricks) == 0)

		clusters, err := ClusterList(tx)
		tests.Assert(t, err == nil)
		tests.Assert(t, len(clusters) == 1)

		cluster, err := NewClusterEntryFromId(tx, clusters[0])
		tests.Assert(t, err == nil)
		tests.Assert(t, len(cluster.Info.Volumes) == 0)

		return nil

	})
	tests.Assert(t, err == nil)
}

func TestVolumeEntryDestroy(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Lots of nodes with little drives
	err := setupSampleDbWithTopology(app,
		1,      // clusters
		4,      // nodes_per_cluster
		4,      // devices_per_node,
		500*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create a volume with a snapshot factor of 1.5
	// For a 200G vol, it would get a brick size of 100G, with a thin pool
	// size of 100G * 1.5 = 150GB.
	v := createSampleVolumeEntry(200)
	v.Info.Snapshot.Enable = true
	v.Info.Snapshot.Factor = 1.5

	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Destroy the volume
	err = v.Destroy(app.db, app.executor)
	tests.Assert(t, err == nil)

	// Check database volume does not exist
	err = app.db.View(func(tx *bolt.Tx) error {

		// Check that all devices have no used data
		devices, err := DeviceList(tx)
		tests.Assert(t, err == nil)
		for _, id := range devices {
			device, err := NewDeviceEntryFromId(tx, id)
			tests.Assert(t, err == nil)
			tests.Assert(t, device.Info.Storage.Used == 0)
			tests.Assert(t, device.Info.Storage.Total == device.Info.Storage.Free)
		}

		// Check there are no bricks
		bricks, err := BrickList(tx)
		tests.Assert(t, len(bricks) == 0)

		return nil

	})
	tests.Assert(t, err == nil)

	// Check that the devices have no bricks
	err = app.db.View(func(tx *bolt.Tx) error {
		devices, err := DeviceList(tx)
		if err != nil {
			return err
		}

		for _, id := range devices {
			device, err := NewDeviceEntryFromId(tx, id)
			if err != nil {
				return err
			}
			tests.Assert(t, len(device.Bricks) == 0, id, device)
		}

		return err
	})
	tests.Assert(t, err == nil)

	// Check that the cluster has no volumes
	err = app.db.View(func(tx *bolt.Tx) error {
		clusters, err := ClusterList(tx)
		if err != nil {
			return err
		}

		tests.Assert(t, len(clusters) == 1)
		cluster, err := NewClusterEntryFromId(tx, clusters[0])
		tests.Assert(t, err == nil)
		tests.Assert(t, len(cluster.Info.Volumes) == 0)

		return nil
	})
	tests.Assert(t, err == nil)

}

func TestVolumeEntryExpandNoSpace(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create cluster
	err := setupSampleDbWithTopology(app,
		10,     // clusters
		2,      // nodes_per_cluster
		2,      // devices_per_node,
		600*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create large volume
	v := createSampleVolumeEntry(1190)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Save a copy of the volume before expansion
	vcopy := &VolumeEntry{}
	*vcopy = *v

	// Asking for a large amount will require too many little bricks
	err = v.Expand(app.db, app.executor, app.allocator, 5000)
	tests.Assert(t, err == ErrMaxBricks, err)

	// Asking for a small amount will set the bricks too small
	err = v.Expand(app.db, app.executor, app.allocator, 10)
	tests.Assert(t, err == ErrMinimumBrickSize, err)

	// Check db is the same as before expansion
	var entry *VolumeEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		entry, err = NewVolumeEntryFromId(tx, v.Info.Id)

		return err
	})
	tests.Assert(t, err == nil, err)
	tests.Assert(t, reflect.DeepEqual(vcopy, entry))
}

func TestVolumeEntryExpandMaxBrickLimit(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a large cluster
	err := setupSampleDbWithTopology(app,
		10,     // clusters
		4,      // nodes_per_cluster
		24,     // devices_per_node,
		600*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create large volume
	v := createSampleVolumeEntry(100)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Add a bunch of bricks until the limit
	fakebricks := make(sort.StringSlice, BrickMaxNum-len(v.Bricks))
	v.Bricks = append(v.Bricks, fakebricks...)

	// Try to expand the volume, but it will return that the max number
	// of bricks has been reached
	err = v.Expand(app.db, app.executor, app.allocator, 100)
	tests.Assert(t, err == ErrMaxBricks, err)
}

func TestVolumeEntryExpandCreateBricksFailure(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create large cluster
	err := setupSampleDbWithTopology(app,
		10,     // clusters
		10,     // nodes_per_cluster
		20,     // devices_per_node,
		600*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create volume
	v := createSampleVolumeEntry(100)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Save a copy of the volume before expansion
	vcopy := &VolumeEntry{}
	*vcopy = *v

	// Mock create bricks to fail
	ErrMock := errors.New("MOCK")
	app.xo.MockBrickCreate = func(host string, brick *executors.BrickRequest) (*executors.BrickInfo, error) {
		return nil, ErrMock
	}

	// Expand volume
	err = v.Expand(app.db, app.executor, app.allocator, 500)
	tests.Assert(t, err == ErrMock)

	// Check db is the same as before expansion
	var entry *VolumeEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		entry, err = NewVolumeEntryFromId(tx, v.Info.Id)

		return err
	})
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(vcopy, entry))
}

func TestVolumeEntryExpand(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create large cluster
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		10,   // nodes_per_cluster
		20,   // devices_per_node,
		6*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create volume
	v := createSampleVolumeEntry(1024)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)
	tests.Assert(t, v.Info.Size == 1024)
	tests.Assert(t, len(v.Bricks) == 2)

	// Expand volume
	err = v.Expand(app.db, app.executor, app.allocator, 1234)
	tests.Assert(t, err == nil)
	tests.Assert(t, v.Info.Size == 1024+1234)
	tests.Assert(t, len(v.Bricks) == 4)

	// Check db
	var entry *VolumeEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		entry, err = NewVolumeEntryFromId(tx, v.Info.Id)

		return err
	})
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(entry, v))
}

func TestVolumeEntryDoNotAllowDeviceOnSameNode(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create cluster with plenty of space, but
	// it will not have enough nodes
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		1,    // nodes_per_cluster
		200,  // devices_per_node,
		6*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create volume
	v := createSampleVolumeEntry(100)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err != nil, err)
	tests.Assert(t, err == ErrNoSpace)

	v = createSampleVolumeEntry(10000)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err != nil, err)
	tests.Assert(t, err == ErrNoSpace)
}

func TestVolumeEntryDestroyCheck(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Lots of nodes with little drives
	err := setupSampleDbWithTopology(app,
		1,      // clusters
		4,      // nodes_per_cluster
		4,      // devices_per_node,
		500*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create a volume with a snapshot factor of 1.5
	// For a 200G vol, it would get a brick size of 100G, with a thin pool
	// size of 100G * 1.5 = 150GB.
	v := createSampleVolumeEntry(200)
	v.Info.Snapshot.Enable = true
	v.Info.Snapshot.Factor = 1.5

	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Test that a volume that is sharing space in a thin pool
	// with either a clone or a snapshot cannot be deleted
	app.xo.MockBrickDestroyCheck = func(host string, brick *executors.BrickRequest) error {
		return fmt.Errorf("BRICKMOCK")
	}
	err = v.Destroy(app.db, app.executor)
	tests.Assert(t, err != nil)
	tests.Assert(t, err.Error() == "BRICKMOCK")
	app.xo.MockBrickDestroyCheck = func(host string, brick *executors.BrickRequest) error {
		return nil
	}

	// Check that a volume with snapshots cannot be deleted
	app.xo.MockVolumeDestroyCheck = func(host, volume string) error {
		return fmt.Errorf("VOLMOCK")
	}
	err = v.Destroy(app.db, app.executor)
	tests.Assert(t, err != nil)
	tests.Assert(t, err.Error() == "VOLMOCK")
	app.xo.MockVolumeDestroyCheck = func(host, volume string) error {
		return nil
	}

	// Now it should be able to be deleted
	err = v.Destroy(app.db, app.executor)
	tests.Assert(t, err == nil)

}

func TestVolumeEntryNameConflictSingleCluster(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	err := setupSampleDbWithTopology(app,
		1,    // clusters
		3,    // nodes_per_cluster
		6,    // devices_per_node,
		6*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create volume
	v := createSampleVolumeEntry(1024)
	v.Info.Name = "myvol"
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Create another volume same name
	v = createSampleVolumeEntry(10000)
	v.Info.Name = "myvol"
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err != nil, err)
}

func TestVolumeEntryNameConflictMultiCluster(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create 10 clusters
	err := setupSampleDbWithTopology(app,
		10,   // clusters
		3,    // nodes_per_cluster
		6,    // devices_per_node,
		6*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create 10 volumes
	for i := 0; i < 10; i++ {
		v := createSampleVolumeEntry(1024)
		v.Info.Name = "myvol"
		err = v.Create(app.db, app.executor, app.allocator)
		logger.Info("%v", v.Info.Cluster)
		tests.Assert(t, err == nil, err)
	}

	// Create another volume same name
	v := createSampleVolumeEntry(10000)
	v.Info.Name = "myvol"
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err != nil, err)
}

func TestReplaceBrickInVolume(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a cluster in the database
	err := setupSampleDbWithTopology(app,
		1,      // clusters
		3,      // nodes_per_cluster
		1,      // devices_per_node,
		500*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	v := createSampleVolumeEntry(100)

	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil, err)
	var brickNames []string
	var be *BrickEntry
	err = app.db.View(func(tx *bolt.Tx) error {

		for _, brick := range v.Bricks {
			be, err = NewBrickEntryFromId(tx, brick)
			if err != nil {
				return err
			}
			ne, err := NewNodeEntryFromId(tx, be.Info.NodeId)
			if err != nil {
				return err
			}
			brickName := fmt.Sprintf("%v:%v", ne.Info.Hostnames.Storage[0], be.Info.Path)
			brickNames = append(brickNames, brickName)
		}
		return nil
	})
	app.xo.MockVolumeInfo = func(host string, volume string) (*executors.Volume, error) {
		var bricks []executors.Brick
		brick := executors.Brick{Name: brickNames[0]}
		bricks = append(bricks, brick)
		brick = executors.Brick{Name: brickNames[1]}
		bricks = append(bricks, brick)
		Bricks := executors.Bricks{
			BrickList: bricks,
		}
		b := &executors.Volume{
			Bricks: Bricks,
		}
		return b, nil
	}
	brickId := be.Id()
	err = v.replaceBrickInVolume(app.db, app.executor, app.allocator, brickId)
	tests.Assert(t, err == nil)

	oldNode := be.Info.NodeId
	brickOnOldNode := false
	oldBrickIdExists := false

	err = app.db.View(func(tx *bolt.Tx) error {

		for _, brick := range v.Bricks {
			be, err = NewBrickEntryFromId(tx, brick)
			if err != nil {
				return err
			}
			ne, err := NewNodeEntryFromId(tx, be.Info.NodeId)
			if err != nil {
				return err
			}
			if ne.Info.Id == oldNode {
				brickOnOldNode = true
			}
			if be.Info.Id == brickId {
				oldBrickIdExists = true
			}
		}
		return nil
	})

	tests.Assert(t, !brickOnOldNode, "brick found on oldNode")
	tests.Assert(t, !oldBrickIdExists, "old Brick not deleted")
}

func TestNewVolumeEntryWithVolumeOptions(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a cluster in the database
	err := setupSampleDbWithTopology(app,
		1,      // clusters
		3,      // nodes_per_cluster
		1,      // devices_per_node,
		500*GB, // disksize)
	)
	tests.Assert(t, err == nil)
	req := &api.VolumeCreateRequest{}
	req.Size = 1024
	req.GlusterVolumeOptions = []string{"test-option"}

	v := NewVolumeEntryFromRequest(req)
	tests.Assert(t, v.Info.Name == "vol_"+v.Info.Id)
	tests.Assert(t, v.Info.Snapshot.Enable == false)
	tests.Assert(t, v.Info.Snapshot.Factor == 1)
	tests.Assert(t, v.Info.Size == 1024)
	tests.Assert(t, len(v.Info.Id) != 0)
	tests.Assert(t, len(v.Bricks) == 0)
	tests.Assert(t, v.GlusterVolumeOptions[0] == "test-option")

	err = v.Create(app.db, app.executor, app.allocator)
	logger.Info("%v", v.Info.Cluster)
	tests.Assert(t, err == nil, err)

	// Check that the data on the database is recorded correctly
	var entry VolumeEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		return entry.Unmarshal(
			tx.Bucket([]byte(BOLTDB_BUCKET_VOLUME)).
				Get([]byte(v.Info.Id)))
	})
	tests.Assert(t, err == nil)
	tests.Assert(t, entry.GlusterVolumeOptions[0] == "test-option")

}
