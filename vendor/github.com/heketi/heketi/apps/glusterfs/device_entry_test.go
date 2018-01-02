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
	"sort"
	"testing"

	"github.com/boltdb/bolt"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
)

func createSampleDeviceEntry(nodeid string, disksize uint64) *DeviceEntry {

	req := &api.DeviceAddRequest{}
	req.NodeId = nodeid
	req.Name = "/dev/" + utils.GenUUID()[:8]

	d := NewDeviceEntryFromRequest(req)
	d.StorageSet(disksize)

	return d
}

func TestNewDeviceEntry(t *testing.T) {

	d := NewDeviceEntry()
	tests.Assert(t, d != nil)
	tests.Assert(t, d.Info.Id == "")
	tests.Assert(t, d.Info.Name == "")
	tests.Assert(t, d.Info.Storage.Free == 0)
	tests.Assert(t, d.Info.Storage.Total == 0)
	tests.Assert(t, d.Info.Storage.Used == 0)
	tests.Assert(t, d.Bricks != nil)
	tests.Assert(t, len(d.Bricks) == 0)

}

func TestNewDeviceEntryFromRequest(t *testing.T) {
	req := &api.DeviceAddRequest{}
	req.NodeId = "123"
	req.Name = "/dev/" + utils.GenUUID()

	d := NewDeviceEntryFromRequest(req)
	tests.Assert(t, d != nil)
	tests.Assert(t, d.Info.Id != "")
	tests.Assert(t, d.Info.Name == req.Name)
	tests.Assert(t, d.Info.Storage.Free == 0)
	tests.Assert(t, d.Info.Storage.Total == 0)
	tests.Assert(t, d.Info.Storage.Used == 0)
	tests.Assert(t, d.NodeId == "123")
	tests.Assert(t, d.Bricks != nil)
	tests.Assert(t, len(d.Bricks) == 0)

}

func TestNewDeviceEntryMarshal(t *testing.T) {
	req := &api.DeviceAddRequest{}
	req.NodeId = "abc"
	req.Name = "/dev/" + utils.GenUUID()

	d := NewDeviceEntryFromRequest(req)
	d.Info.Storage.Free = 10
	d.Info.Storage.Total = 100
	d.Info.Storage.Used = 1000
	d.BrickAdd("abc")
	d.BrickAdd("def")

	buffer, err := d.Marshal()
	tests.Assert(t, err == nil)
	tests.Assert(t, buffer != nil)
	tests.Assert(t, len(buffer) > 0)

	um := &DeviceEntry{}
	err = um.Unmarshal(buffer)
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(um, d))

}

func TestDeviceEntryNewBrickEntry(t *testing.T) {
	req := &api.DeviceAddRequest{}
	req.NodeId = "abc"
	req.Name = "/dev/" + utils.GenUUID()

	d := NewDeviceEntryFromRequest(req)
	d.Info.Storage.Free = 900
	d.Info.Storage.Total = 1000
	d.Info.Storage.Used = 100

	// Alignment
	d.ExtentSize = 8

	// Too large
	brick := d.NewBrickEntry(1000000000, 1.5, 1000, "abc")
	tests.Assert(t, brick == nil)

	// --- Now check with a real value ---

	// Check newly created brick
	size := 201
	tpsize := uint64(float32(size) * 1.5)

	// Alignment
	tpsize += d.ExtentSize - (tpsize % d.ExtentSize)

	// Calculate metadatasize
	metadatasize := d.poolMetadataSize(tpsize)

	// Alignment
	metadatasize += d.ExtentSize - (metadatasize % d.ExtentSize)
	total := tpsize + metadatasize

	brick = d.NewBrickEntry(200, 1.5, 1000, "abc")
	tests.Assert(t, brick != nil)
	tests.Assert(t, brick.TpSize == tpsize)
	tests.Assert(t, brick.PoolMetadataSize == metadatasize, brick.PoolMetadataSize, metadatasize)
	tests.Assert(t, brick.Info.Size == 200)
	tests.Assert(t, brick.gidRequested == 1000)
	tests.Assert(t, brick.Info.VolumeId == "abc")

	// Check it was subtracted from device storage
	tests.Assert(t, d.Info.Storage.Used == 100+total)
	tests.Assert(t, d.Info.Storage.Free == 900-total)
	tests.Assert(t, d.Info.Storage.Total == 1000)
}

func TestDeviceEntryAddDeleteBricks(t *testing.T) {
	d := NewDeviceEntry()
	tests.Assert(t, len(d.Bricks) == 0)

	d.BrickAdd("123")
	tests.Assert(t, utils.SortedStringHas(d.Bricks, "123"))
	tests.Assert(t, len(d.Bricks) == 1)
	d.BrickAdd("abc")
	tests.Assert(t, utils.SortedStringHas(d.Bricks, "123"))
	tests.Assert(t, utils.SortedStringHas(d.Bricks, "abc"))
	tests.Assert(t, len(d.Bricks) == 2)

	d.BrickDelete("123")
	tests.Assert(t, !utils.SortedStringHas(d.Bricks, "123"))
	tests.Assert(t, utils.SortedStringHas(d.Bricks, "abc"))
	tests.Assert(t, len(d.Bricks) == 1)

	d.BrickDelete("ccc")
	tests.Assert(t, !utils.SortedStringHas(d.Bricks, "123"))
	tests.Assert(t, utils.SortedStringHas(d.Bricks, "abc"))
	tests.Assert(t, len(d.Bricks) == 1)
}

func TestNewDeviceEntryFromIdNotFound(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Test for ID not found
	err := app.db.View(func(tx *bolt.Tx) error {
		_, err := NewDeviceEntryFromId(tx, "123")
		return err
	})
	tests.Assert(t, err == ErrNotFound)

}

func TestDeviceEntryRegister(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a device
	req := &api.DeviceAddRequest{}
	req.NodeId = "abc"
	req.Name = "/dev/" + utils.GenUUID()

	d := NewDeviceEntryFromRequest(req)

	// Register device
	err := app.db.Update(func(tx *bolt.Tx) error {
		err := d.Register(tx)
		tests.Assert(t, err == nil)

		return d.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Should not be able to register again
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := d.Register(tx)
		tests.Assert(t, err != nil)

		return err
	})
	tests.Assert(t, err != nil)

	// Create another device on a different node device
	req = &api.DeviceAddRequest{}
	req.NodeId = "def"
	req.Name = "/dev/" + utils.GenUUID()

	d2 := NewDeviceEntryFromRequest(req)

	// Same device on different node should work
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := d2.Register(tx)
		tests.Assert(t, err == nil)

		return d2.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Remove d
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := d.Deregister(tx)
		tests.Assert(t, err == nil)

		return d.Delete(tx)
	})
	tests.Assert(t, err == nil)

	// Register d node again
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := d.Register(tx)
		tests.Assert(t, err == nil)

		return d.Save(tx)
	})
	tests.Assert(t, err == nil)

}

func TestDeviceEntryRegisterStaleRegistration(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a device
	req := &api.DeviceAddRequest{}
	req.NodeId = "abc"
	req.Name = "/dev/" + utils.GenUUID()

	d := NewDeviceEntryFromRequest(req)

	// Only register device but do not save it
	err := app.db.Update(func(tx *bolt.Tx) error {
		return d.Register(tx)
	})
	tests.Assert(t, err == nil)

	// Should be able to register again
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := d.Register(tx)
		tests.Assert(t, err == nil)

		return d.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Should not be able to register again
	err = app.db.Update(func(tx *bolt.Tx) error {
		return d.Register(tx)
	})
	tests.Assert(t, err != nil)

	// Remove d
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := d.Deregister(tx)
		tests.Assert(t, err == nil)

		return d.Delete(tx)
	})
	tests.Assert(t, err == nil)

	// Register d node again
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := d.Register(tx)
		tests.Assert(t, err == nil)

		return d.Save(tx)
	})
	tests.Assert(t, err == nil)

}

func TestNewDeviceEntryFromId(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a device
	req := &api.DeviceAddRequest{}
	req.NodeId = "abc"
	req.Name = "/dev/" + utils.GenUUID()

	d := NewDeviceEntryFromRequest(req)
	d.Info.Storage.Free = 10
	d.Info.Storage.Total = 100
	d.Info.Storage.Used = 1000
	d.BrickAdd("abc")
	d.BrickAdd("def")

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return d.Save(tx)
	})
	tests.Assert(t, err == nil)

	var device *DeviceEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		device, err = NewDeviceEntryFromId(tx, d.Info.Id)
		if err != nil {
			return err
		}
		return nil

	})
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(device, d))
}

func TestNewDeviceEntrySaveDelete(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a device
	req := &api.DeviceAddRequest{}
	req.NodeId = "abc"
	req.Name = "/dev/" + utils.GenUUID()

	d := NewDeviceEntryFromRequest(req)
	d.Info.Storage.Free = 10
	d.Info.Storage.Total = 100
	d.Info.Storage.Used = 1000
	d.BrickAdd("abc")
	d.BrickAdd("def")

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return d.Save(tx)
	})
	tests.Assert(t, err == nil)

	var device *DeviceEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		device, err = NewDeviceEntryFromId(tx, d.Info.Id)
		if err != nil {
			return err
		}
		return nil

	})
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(device, d))

	// Delete entry which has devices
	err = app.db.Update(func(tx *bolt.Tx) error {
		var err error
		device, err = NewDeviceEntryFromId(tx, d.Info.Id)
		if err != nil {
			return err
		}

		err = device.Delete(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == ErrConflict)

	// Delete devices in device
	device.BrickDelete("abc")
	device.BrickDelete("def")
	tests.Assert(t, len(device.Bricks) == 0)
	err = app.db.Update(func(tx *bolt.Tx) error {
		return device.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Now try to delete the device
	err = app.db.Update(func(tx *bolt.Tx) error {
		var err error
		device, err = NewDeviceEntryFromId(tx, d.Info.Id)
		if err != nil {
			return err
		}

		err = device.Delete(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)

	// Check device has been deleted and is not in db
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		device, err = NewDeviceEntryFromId(tx, d.Info.Id)
		if err != nil {
			return err
		}
		return nil

	})
	tests.Assert(t, err == ErrNotFound)
}

func TestNewDeviceEntryNewInfoResponseBadBrickIds(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a device
	req := &api.DeviceAddRequest{}
	req.NodeId = "abc"
	req.Name = "/dev/" + utils.GenUUID()

	// Add bad brick ids
	d := NewDeviceEntryFromRequest(req)
	d.Info.Storage.Free = 10
	d.Info.Storage.Total = 100
	d.Info.Storage.Used = 1000
	d.BrickAdd("abc")
	d.BrickAdd("def")

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return d.Save(tx)
	})
	tests.Assert(t, err == nil)

	var info *api.DeviceInfoResponse
	err = app.db.View(func(tx *bolt.Tx) error {
		device, err := NewDeviceEntryFromId(tx, d.Info.Id)
		if err != nil {
			return err
		}

		info, err = device.NewInfoResponse(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == ErrNotFound)
}

func TestNewDeviceEntryNewInfoResponse(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a device
	req := &api.DeviceAddRequest{}
	req.NodeId = "abc"
	req.Name = "/dev/" + utils.GenUUID()

	d := NewDeviceEntryFromRequest(req)
	d.Info.Storage.Free = 10
	d.Info.Storage.Total = 100
	d.Info.Storage.Used = 1000

	// Create a brick
	b := &BrickEntry{}
	b.Info.Id = "bbb"
	b.Info.Size = 10
	b.Info.NodeId = "abc"
	b.Info.DeviceId = d.Info.Id
	b.Info.Path = "/somepath"

	// Add brick to device
	d.BrickAdd("bbb")

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		err := d.Save(tx)
		if err != nil {
			return err
		}

		return b.Save(tx)
	})
	tests.Assert(t, err == nil)

	var info *api.DeviceInfoResponse
	err = app.db.View(func(tx *bolt.Tx) error {
		device, err := NewDeviceEntryFromId(tx, d.Info.Id)
		if err != nil {
			return err
		}

		info, err = device.NewInfoResponse(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)
	tests.Assert(t, info.Id == d.Info.Id)
	tests.Assert(t, info.Name == d.Info.Name)
	tests.Assert(t, reflect.DeepEqual(info.Storage, d.Info.Storage))
	tests.Assert(t, len(info.Bricks) == 1)
	tests.Assert(t, info.Bricks[0].Id == "bbb")
	tests.Assert(t, info.Bricks[0].Path == "/somepath")
	tests.Assert(t, info.Bricks[0].NodeId == "abc")
	tests.Assert(t, info.Bricks[0].DeviceId == d.Info.Id)
	tests.Assert(t, info.Bricks[0].Size == 10)

}

func TestDeviceEntryStorage(t *testing.T) {
	d := NewDeviceEntry()

	tests.Assert(t, d.Info.Storage.Free == 0)
	tests.Assert(t, d.Info.Storage.Total == 0)
	tests.Assert(t, d.Info.Storage.Used == 0)

	d.StorageSet(1000)
	tests.Assert(t, d.Info.Storage.Free == 1000)
	tests.Assert(t, d.Info.Storage.Total == 1000)
	tests.Assert(t, d.Info.Storage.Used == 0)

	d.StorageSet(2000)
	tests.Assert(t, d.Info.Storage.Free == 2000)
	tests.Assert(t, d.Info.Storage.Total == 2000)
	tests.Assert(t, d.Info.Storage.Used == 0)

	d.StorageAllocate(1000)
	tests.Assert(t, d.Info.Storage.Free == 1000)
	tests.Assert(t, d.Info.Storage.Total == 2000)
	tests.Assert(t, d.Info.Storage.Used == 1000)

	d.StorageAllocate(500)
	tests.Assert(t, d.Info.Storage.Free == 500)
	tests.Assert(t, d.Info.Storage.Total == 2000)
	tests.Assert(t, d.Info.Storage.Used == 1500)

	d.StorageFree(500)
	tests.Assert(t, d.Info.Storage.Free == 1000)
	tests.Assert(t, d.Info.Storage.Total == 2000)
	tests.Assert(t, d.Info.Storage.Used == 1000)

	d.StorageFree(1000)
	tests.Assert(t, d.Info.Storage.Free == 2000)
	tests.Assert(t, d.Info.Storage.Total == 2000)
	tests.Assert(t, d.Info.Storage.Used == 0)
}

func TestDeviceSetStateFailed(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create allocator
	mockAllocator := NewMockAllocator(app.db)
	app.allocator = mockAllocator

	// Create cluster entry
	c := NewClusterEntry()
	c.Info.Id = "cluster"

	// Create a node
	n := NewNodeEntry()
	tests.Assert(t, n != nil)
	tests.Assert(t, n.State == api.EntryStateOnline)

	// Initialize node
	n.Info.Id = "node"
	n.Info.ClusterId = "cluster"
	n.Devices = sort.StringSlice{"d1"}

	// Create device entry
	d := NewDeviceEntry()
	d.Info.Id = "d1"
	d.Info.Name = "/d1"
	d.NodeId = "node"

	// Add to allocator
	mockAllocator.AddDevice(c, n, d)

	// Save in db
	app.db.Update(func(tx *bolt.Tx) error {
		err := c.Save(tx)
		tests.Assert(t, err == nil)

		err = n.Save(tx)
		tests.Assert(t, err == nil)

		err = d.Save(tx)
		tests.Assert(t, err == nil)

		// Check ring
		tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 1)
		tests.Assert(t, mockAllocator.clustermap[c.Info.Id][0] == d.Info.Id)
		return nil
	})

	// Set offline
	err := d.SetState(app.db, app.executor, mockAllocator, api.EntryStateOffline)
	tests.Assert(t, d.State == api.EntryStateOffline)
	tests.Assert(t, err == nil, err)
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 0)

	// Set failed, Note: this requires the current state to be offline
	err = d.SetState(app.db, app.executor, mockAllocator, api.EntryStateFailed)
	tests.Assert(t, d.State == api.EntryStateFailed)
	tests.Assert(t, err == nil)
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 0)

	// Set failed again
	err = d.SetState(app.db, app.executor, mockAllocator, api.EntryStateFailed)
	tests.Assert(t, d.State == api.EntryStateFailed)
	tests.Assert(t, err == nil)

	// Set offline
	err = d.SetState(app.db, app.executor, mockAllocator, api.EntryStateOffline)
	tests.Assert(t, d.State == api.EntryStateFailed)
	tests.Assert(t, err != nil)
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 0)

	// Set online
	err = d.SetState(app.db, app.executor, mockAllocator, api.EntryStateOnline)
	tests.Assert(t, d.State == api.EntryStateFailed)
	tests.Assert(t, err != nil)
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 0)

}

func TestDeviceSetStateOfflineOnline(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create allocator
	mockAllocator := NewMockAllocator(app.db)
	app.allocator = mockAllocator

	// Create cluster entry
	c := NewClusterEntry()
	c.Info.Id = "cluster"

	// Create a node
	n := NewNodeEntry()
	tests.Assert(t, n != nil)
	tests.Assert(t, n.State == api.EntryStateOnline)

	// Initialize node
	n.Info.Id = "node"
	n.Info.ClusterId = "cluster"
	n.Devices = sort.StringSlice{"d1"}

	// Create device entry
	d := NewDeviceEntry()
	d.Info.Id = "d1"
	d.Info.Name = "/d1"
	d.NodeId = "node"

	// Add to allocator
	mockAllocator.AddDevice(c, n, d)

	// Save in db
	app.db.Update(func(tx *bolt.Tx) error {
		err := c.Save(tx)
		tests.Assert(t, err == nil)

		err = n.Save(tx)
		tests.Assert(t, err == nil)

		err = d.Save(tx)
		tests.Assert(t, err == nil)

		// Check ring
		tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 1)
		tests.Assert(t, mockAllocator.clustermap[c.Info.Id][0] == d.Info.Id)
		return nil
	})

	// Set offline
	err := d.SetState(app.db, app.executor, mockAllocator, api.EntryStateOffline)
	tests.Assert(t, d.State == api.EntryStateOffline)
	tests.Assert(t, err == nil)

	// Check it was removed from ring
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 0)

	// Set offline again
	err = d.SetState(app.db, app.executor, mockAllocator, api.EntryStateOffline)
	tests.Assert(t, d.State == api.EntryStateOffline)
	tests.Assert(t, err == nil)

	// Set online
	err = d.SetState(app.db, app.executor, mockAllocator, api.EntryStateOnline)
	tests.Assert(t, d.State == api.EntryStateOnline)
	tests.Assert(t, err == nil)
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 1)
	tests.Assert(t, mockAllocator.clustermap[c.Info.Id][0] == d.Info.Id)

}
