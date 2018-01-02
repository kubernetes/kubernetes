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
	"os"
	"reflect"
	"testing"

	"github.com/boltdb/bolt"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
)

func createSampleClusterEntry() *ClusterEntry {
	return NewClusterEntryFromRequest()
}

func TestNewClusterEntry(t *testing.T) {
	c := NewClusterEntry()
	tests.Assert(t, c.Info.Id == "")
	tests.Assert(t, c.Info.Volumes != nil)
	tests.Assert(t, c.Info.Nodes != nil)
	tests.Assert(t, len(c.Info.Volumes) == 0)
	tests.Assert(t, len(c.Info.Nodes) == 0)
}

func TestNewClusterEntryFromRequest(t *testing.T) {

	c := NewClusterEntryFromRequest()
	tests.Assert(t, c != nil)
	tests.Assert(t, len(c.Info.Id) > 0)
	tests.Assert(t, c.Info.Id != "")
	tests.Assert(t, c.Info.Volumes != nil)
	tests.Assert(t, c.Info.Nodes != nil)
	tests.Assert(t, len(c.Info.Volumes) == 0)
	tests.Assert(t, len(c.Info.Nodes) == 0)

}

func TestClusterEntryMarshal(t *testing.T) {
	m := NewClusterEntry()
	m.Info.Id = "123"
	m.Info.Nodes = []string{"1", "2"}
	m.Info.Volumes = []string{"3", "4", "5"}

	buffer, err := m.Marshal()
	tests.Assert(t, err == nil)
	tests.Assert(t, buffer != nil)
	tests.Assert(t, len(buffer) > 0)

	um := NewClusterEntry()
	err = um.Unmarshal(buffer)
	tests.Assert(t, err == nil)

	tests.Assert(t, m.Info.Id == um.Info.Id)
	tests.Assert(t, len(um.Info.Volumes) == 3)
	tests.Assert(t, len(um.Info.Nodes) == 2)
	tests.Assert(t, um.Info.Nodes[0] == "1")
	tests.Assert(t, um.Info.Nodes[1] == "2")
	tests.Assert(t, um.Info.Volumes[0] == "3")
	tests.Assert(t, um.Info.Volumes[1] == "4")
	tests.Assert(t, um.Info.Volumes[2] == "5")
}

func TestClusterEntryAddDeleteElements(t *testing.T) {
	c := NewClusterEntry()

	c.NodeAdd("123")
	tests.Assert(t, len(c.Info.Nodes) == 1)
	tests.Assert(t, len(c.Info.Volumes) == 0)
	tests.Assert(t, utils.SortedStringHas(c.Info.Nodes, "123"))

	c.NodeAdd("456")
	tests.Assert(t, len(c.Info.Nodes) == 2)
	tests.Assert(t, len(c.Info.Volumes) == 0)
	tests.Assert(t, utils.SortedStringHas(c.Info.Nodes, "123"))
	tests.Assert(t, utils.SortedStringHas(c.Info.Nodes, "456"))

	c.VolumeAdd("aabb")
	tests.Assert(t, len(c.Info.Nodes) == 2)
	tests.Assert(t, len(c.Info.Volumes) == 1)
	tests.Assert(t, utils.SortedStringHas(c.Info.Nodes, "123"))
	tests.Assert(t, utils.SortedStringHas(c.Info.Nodes, "456"))
	tests.Assert(t, utils.SortedStringHas(c.Info.Volumes, "aabb"))

	c.NodeDelete("aabb")
	tests.Assert(t, len(c.Info.Nodes) == 2)
	tests.Assert(t, len(c.Info.Volumes) == 1)
	tests.Assert(t, utils.SortedStringHas(c.Info.Nodes, "123"))
	tests.Assert(t, utils.SortedStringHas(c.Info.Nodes, "456"))
	tests.Assert(t, utils.SortedStringHas(c.Info.Volumes, "aabb"))

	c.NodeDelete("456")
	tests.Assert(t, len(c.Info.Nodes) == 1)
	tests.Assert(t, len(c.Info.Volumes) == 1)
	tests.Assert(t, utils.SortedStringHas(c.Info.Nodes, "123"))
	tests.Assert(t, !utils.SortedStringHas(c.Info.Nodes, "456"))
	tests.Assert(t, utils.SortedStringHas(c.Info.Volumes, "aabb"))

	c.NodeDelete("123")
	tests.Assert(t, len(c.Info.Nodes) == 0)
	tests.Assert(t, len(c.Info.Volumes) == 1)
	tests.Assert(t, !utils.SortedStringHas(c.Info.Nodes, "123"))
	tests.Assert(t, !utils.SortedStringHas(c.Info.Nodes, "456"))
	tests.Assert(t, utils.SortedStringHas(c.Info.Volumes, "aabb"))

	c.VolumeDelete("123")
	tests.Assert(t, len(c.Info.Nodes) == 0)
	tests.Assert(t, len(c.Info.Volumes) == 1)
	tests.Assert(t, !utils.SortedStringHas(c.Info.Nodes, "123"))
	tests.Assert(t, !utils.SortedStringHas(c.Info.Nodes, "456"))
	tests.Assert(t, utils.SortedStringHas(c.Info.Volumes, "aabb"))

	c.VolumeDelete("aabb")
	tests.Assert(t, len(c.Info.Nodes) == 0)
	tests.Assert(t, len(c.Info.Volumes) == 0)
	tests.Assert(t, !utils.SortedStringHas(c.Info.Nodes, "123"))
	tests.Assert(t, !utils.SortedStringHas(c.Info.Nodes, "456"))
	tests.Assert(t, !utils.SortedStringHas(c.Info.Volumes, "aabb"))
}

func TestNewClusterEntryFromIdNotFound(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Test for ID not found
	err := app.db.View(func(tx *bolt.Tx) error {
		_, err := NewClusterEntryFromId(tx, "123")
		return err
	})
	tests.Assert(t, err == ErrNotFound)

}

func TestNewClusterEntryFromId(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a cluster
	c := NewClusterEntryFromRequest()
	c.NodeAdd("node_abc")
	c.NodeAdd("node_def")
	c.VolumeAdd("vol_abc")

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return c.Save(tx)
	})
	tests.Assert(t, err == nil)

	var cluster *ClusterEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		cluster, err = NewClusterEntryFromId(tx, c.Info.Id)
		if err != nil {
			return err
		}
		return nil

	})
	tests.Assert(t, err == nil)

	tests.Assert(t, cluster.Info.Id == c.Info.Id)
	tests.Assert(t, len(c.Info.Nodes) == 2)
	tests.Assert(t, len(c.Info.Volumes) == 1)
	tests.Assert(t, utils.SortedStringHas(c.Info.Nodes, "node_abc"))
	tests.Assert(t, utils.SortedStringHas(c.Info.Nodes, "node_def"))
	tests.Assert(t, utils.SortedStringHas(c.Info.Volumes, "vol_abc"))

}

func TestNewClusterEntrySaveDelete(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a cluster
	c := NewClusterEntryFromRequest()
	c.NodeAdd("node_abc")
	c.NodeAdd("node_def")
	c.VolumeAdd("vol_abc")

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return c.Save(tx)
	})
	tests.Assert(t, err == nil)

	var cluster *ClusterEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		cluster, err = NewClusterEntryFromId(tx, c.Info.Id)
		if err != nil {
			return err
		}
		return nil

	})
	tests.Assert(t, err == nil)

	tests.Assert(t, cluster.Info.Id == c.Info.Id)
	tests.Assert(t, len(c.Info.Nodes) == 2)
	tests.Assert(t, len(c.Info.Volumes) == 1)
	tests.Assert(t, utils.SortedStringHas(c.Info.Nodes, "node_abc"))
	tests.Assert(t, utils.SortedStringHas(c.Info.Nodes, "node_def"))
	tests.Assert(t, utils.SortedStringHas(c.Info.Volumes, "vol_abc"))

	// Delete entry which has devices
	err = app.db.Update(func(tx *bolt.Tx) error {
		var err error
		cluster, err = NewClusterEntryFromId(tx, c.Info.Id)
		if err != nil {
			return err
		}

		err = cluster.Delete(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == ErrConflict)

	// Delete devices in cluster
	cluster.VolumeDelete("vol_abc")
	tests.Assert(t, len(cluster.Info.Volumes) == 0)
	tests.Assert(t, len(cluster.Info.Nodes) == 2)

	// Save cluster
	err = app.db.Update(func(tx *bolt.Tx) error {
		return cluster.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Try do delete a cluster which still has nodes
	err = app.db.Update(func(tx *bolt.Tx) error {
		var err error
		cluster, err = NewClusterEntryFromId(tx, c.Info.Id)
		if err != nil {
			return err
		}

		err = cluster.Delete(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == ErrConflict)

	// Delete cluster
	cluster.NodeDelete("node_abc")
	cluster.NodeDelete("node_def")
	tests.Assert(t, len(cluster.Info.Nodes) == 0)

	// Save cluster
	err = app.db.Update(func(tx *bolt.Tx) error {
		return cluster.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Now try to delete the cluster with no elements
	err = app.db.Update(func(tx *bolt.Tx) error {
		var err error
		cluster, err = NewClusterEntryFromId(tx, c.Info.Id)
		if err != nil {
			return err
		}

		err = cluster.Delete(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)

	// Check cluster has been deleted and is not in db
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		cluster, err = NewClusterEntryFromId(tx, c.Info.Id)
		if err != nil {
			return err
		}
		return nil

	})
	tests.Assert(t, err == ErrNotFound)
}

func TestNewClusterEntryNewInfoResponse(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a cluster
	c := NewClusterEntryFromRequest()
	c.NodeAdd("node_abc")
	c.NodeAdd("node_def")
	c.VolumeAdd("vol_abc")

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return c.Save(tx)
	})
	tests.Assert(t, err == nil)

	var info *api.ClusterInfoResponse
	err = app.db.View(func(tx *bolt.Tx) error {
		cluster, err := NewClusterEntryFromId(tx, c.Info.Id)
		if err != nil {
			return err
		}

		info, err = cluster.NewClusterInfoResponse(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)

	tests.Assert(t, info.Id == c.Info.Id)
	tests.Assert(t, reflect.DeepEqual(info.Nodes, c.Info.Nodes))
	tests.Assert(t, reflect.DeepEqual(info.Volumes, c.Info.Volumes))
}
