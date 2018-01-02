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
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/boltdb/bolt"
	"github.com/heketi/heketi/executors"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/tests"
)

func TestNewBrickEntry(t *testing.T) {

	size := uint64(10)
	tpsize := size * 2
	deviceid := "abc"
	nodeid := "def"
	ps := size
	gid := int64(1)
	volumeid := "ghi"

	b := NewBrickEntry(size, tpsize, ps, deviceid, nodeid, gid, volumeid)
	tests.Assert(t, b.Info.Id != "")
	tests.Assert(t, b.TpSize == tpsize)
	tests.Assert(t, b.PoolMetadataSize == ps)
	tests.Assert(t, b.Info.DeviceId == deviceid)
	tests.Assert(t, b.Info.NodeId == nodeid)
	tests.Assert(t, b.Info.Size == size)
	tests.Assert(t, b.gidRequested == gid)
	tests.Assert(t, b.Info.VolumeId == volumeid)
}

func TestBrickEntryMarshal(t *testing.T) {
	size := uint64(10)
	tpsize := size * 2
	deviceid := "abc"
	nodeid := "def"
	ps := size
	gid := int64(0)
	volumeid := "ghi"
	m := NewBrickEntry(size, tpsize, ps, deviceid, nodeid, gid, volumeid)

	buffer, err := m.Marshal()
	tests.Assert(t, err == nil)
	tests.Assert(t, buffer != nil)
	tests.Assert(t, len(buffer) > 0)

	um := &BrickEntry{}
	err = um.Unmarshal(buffer)
	tests.Assert(t, err == nil)

	tests.Assert(t, reflect.DeepEqual(um, m))
}

func TestNewBrickEntryFromIdNotFound(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Test for ID not found
	err := app.db.View(func(tx *bolt.Tx) error {
		_, err := NewBrickEntryFromId(tx, "123")
		return err
	})
	tests.Assert(t, err == ErrNotFound)

}

func TestNewBrickEntryFromId(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a brick
	b := NewBrickEntry(10, 20, 5, "abc", "def", 0, "ghi")

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return b.Save(tx)
	})
	tests.Assert(t, err == nil)

	var brick *BrickEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		brick, err = NewBrickEntryFromId(tx, b.Info.Id)
		return err
	})
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(brick, b))

}

func TestNewBrickEntrySaveDelete(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a brick
	b := NewBrickEntry(10, 20, 5, "abc", "def", 1000, "ghi")

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return b.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Delete entry which has devices
	var brick *BrickEntry
	err = app.db.Update(func(tx *bolt.Tx) error {
		var err error
		brick, err = NewBrickEntryFromId(tx, b.Info.Id)
		if err != nil {
			return err
		}

		err = brick.Delete(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)

	// Check brick has been deleted and is not in db
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		brick, err = NewBrickEntryFromId(tx, b.Info.Id)
		return err
	})
	tests.Assert(t, err == ErrNotFound)
}

func TestNewBrickEntryNewInfoResponse(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a brick
	b := NewBrickEntry(10, 20, 5, "abc", "def", 1000, "ghi")

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return b.Save(tx)
	})
	tests.Assert(t, err == nil)

	var info *api.BrickInfo
	err = app.db.View(func(tx *bolt.Tx) error {
		brick, err := NewBrickEntryFromId(tx, b.Id())
		if err != nil {
			return err
		}

		info, err = brick.NewInfoResponse(tx)
		return err
	})
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(*info, b.Info))
}

func TestBrickEntryDestroyCheck(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a brick
	b := NewBrickEntry(10, 20, 5, "abc", "node", 1000, "ghi")
	n := NewNodeEntry()
	n.Info.Id = "node"
	n.Info.Hostnames.Manage = []string{"manage"}
	n.Info.Hostnames.Storage = []string{"storage"}

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		err := n.Save(tx)
		tests.Assert(t, err == nil)
		return b.Save(tx)
	})
	tests.Assert(t, err == nil)

	app.xo.MockBrickDestroyCheck = func(host string, brick *executors.BrickRequest) error {
		return fmt.Errorf("MOCK error")
	}

	err = b.DestroyCheck(app.db, app.executor)
	tests.Assert(t, err != nil)

	app.xo.MockBrickDestroyCheck = func(host string, brick *executors.BrickRequest) error {
		return nil
	}

	err = b.DestroyCheck(app.db, app.executor)
	tests.Assert(t, err == nil, err)
}

func TestBrickEntryCreate(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Set test values
	size := uint64(10)
	tpsize := uint64(20)
	poolMetadataSize := uint64(5)
	deviceid := "abc"
	nodeid := "node"
	gid := int64(1000)
	volumeid := "ghi"

	// Create a brick
	b := NewBrickEntry(size, tpsize, poolMetadataSize,
		deviceid, nodeid, gid, volumeid)
	n := NewNodeEntry()
	n.Info.Id = nodeid
	n.Info.Hostnames.Manage = []string{"manage"}
	n.Info.Hostnames.Storage = []string{"storage"}

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		err := n.Save(tx)
		tests.Assert(t, err == nil)
		return b.Save(tx)
	})
	tests.Assert(t, err == nil)

	app.xo.MockBrickCreate = func(host string,
		brick *executors.BrickRequest) (*executors.BrickInfo, error) {
		bInfo := &executors.BrickInfo{
			Path: "/mockpath",
		}

		tests.Assert(t, brick.Gid == gid)
		tests.Assert(t, brick.Name == b.Info.Id)
		tests.Assert(t, brick.PoolMetadataSize == poolMetadataSize)
		tests.Assert(t, brick.Size == size)
		tests.Assert(t, brick.TpSize == tpsize)
		tests.Assert(t, brick.VgId == deviceid)

		return bInfo, nil
	}
	err = b.Create(app.db, app.executor)
	tests.Assert(t, err == nil)
}
