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

func createSampleNodeEntry() *NodeEntry {
	req := &api.NodeAddRequest{
		ClusterId: "123",
		Hostnames: api.HostAddresses{
			Manage:  []string{"manage" + utils.GenUUID()[:8]},
			Storage: []string{"storage" + utils.GenUUID()[:8]},
		},
		Zone: 99,
	}

	return NewNodeEntryFromRequest(req)
}

func TestNewNodeEntry(t *testing.T) {

	n := NewNodeEntry()
	tests.Assert(t, n.Info.Id == "")
	tests.Assert(t, n.Info.ClusterId == "")
	tests.Assert(t, len(n.Devices) == 0)
	tests.Assert(t, n.Devices != nil)
}

func TestNewNodeEntryFromRequest(t *testing.T) {
	req := &api.NodeAddRequest{
		ClusterId: "123",
		Hostnames: api.HostAddresses{
			Manage:  []string{"manage"},
			Storage: []string{"storage"},
		},
		Zone: 99,
	}

	n := NewNodeEntryFromRequest(req)
	tests.Assert(t, n != nil)
	tests.Assert(t, n.Info.ClusterId == req.ClusterId)
	tests.Assert(t, n.Info.Zone == req.Zone)
	tests.Assert(t, len(n.Info.Id) > 0)
	tests.Assert(t, len(n.Info.Hostnames.Manage) == len(req.Hostnames.Manage))
	tests.Assert(t, len(n.Info.Hostnames.Storage) == len(req.Hostnames.Storage))
	tests.Assert(t, n.Info.Hostnames.Manage[0] == req.Hostnames.Manage[0])
	tests.Assert(t, n.Info.Hostnames.Storage[0] == req.Hostnames.Storage[0])

}

func TestNewNodeEntryMarshal(t *testing.T) {
	req := &api.NodeAddRequest{
		ClusterId: "123",
		Hostnames: api.HostAddresses{
			Manage:  []string{"manage"},
			Storage: []string{"storage"},
		},
		Zone: 99,
	}

	n := NewNodeEntryFromRequest(req)
	n.DeviceAdd("abc")
	n.DeviceAdd("def")

	buffer, err := n.Marshal()
	tests.Assert(t, err == nil)
	tests.Assert(t, buffer != nil)
	tests.Assert(t, len(buffer) > 0)

	um := &NodeEntry{}
	err = um.Unmarshal(buffer)
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(n, um))

}

func TestNodeEntryAddDeleteDevices(t *testing.T) {
	n := NewNodeEntry()
	tests.Assert(t, len(n.Devices) == 0)

	n.DeviceAdd("123")
	tests.Assert(t, utils.SortedStringHas(n.Devices, "123"))
	tests.Assert(t, len(n.Devices) == 1)
	n.DeviceAdd("abc")
	tests.Assert(t, utils.SortedStringHas(n.Devices, "123"))
	tests.Assert(t, utils.SortedStringHas(n.Devices, "abc"))
	tests.Assert(t, len(n.Devices) == 2)

	n.DeviceDelete("123")
	tests.Assert(t, !utils.SortedStringHas(n.Devices, "123"))
	tests.Assert(t, utils.SortedStringHas(n.Devices, "abc"))
	tests.Assert(t, len(n.Devices) == 1)

	n.DeviceDelete("ccc")
	tests.Assert(t, !utils.SortedStringHas(n.Devices, "123"))
	tests.Assert(t, utils.SortedStringHas(n.Devices, "abc"))
	tests.Assert(t, len(n.Devices) == 1)
}

func TestNodeEntryRegister(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a node
	req := &api.NodeAddRequest{
		ClusterId: "123",
		Hostnames: api.HostAddresses{
			Manage:  []string{"manage"},
			Storage: []string{"storage"},
		},
		Zone: 99,
	}
	n := NewNodeEntryFromRequest(req)

	// Register node
	err := app.db.Update(func(tx *bolt.Tx) error {
		err := n.Register(tx)
		tests.Assert(t, err == nil)

		return n.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Should not be able to register again
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := n.Register(tx)
		tests.Assert(t, err != nil)

		return err
	})
	tests.Assert(t, err != nil)

	// Create a new node on *different* cluster
	req = &api.NodeAddRequest{
		ClusterId: "abc",
		Hostnames: api.HostAddresses{
			// Same name as previous
			Manage:  []string{"manage"},
			Storage: []string{"storage"},
		},
		Zone: 99,
	}
	diff_cluster_n := NewNodeEntryFromRequest(req)

	// Should not be able to register diff_cluster_n
	err = app.db.Update(func(tx *bolt.Tx) error {
		return diff_cluster_n.Register(tx)
	})
	tests.Assert(t, err != nil)

	// Add a new node
	req = &api.NodeAddRequest{
		ClusterId: "3",
		Hostnames: api.HostAddresses{
			Manage:  []string{"manage2"},
			Storage: []string{"storage2"},
		},
		Zone: 99,
	}
	n2 := NewNodeEntryFromRequest(req)

	// Register n2 node
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := n2.Register(tx)
		tests.Assert(t, err == nil)

		return n2.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Remove n
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := n.Deregister(tx)
		tests.Assert(t, err == nil)

		return n.Delete(tx)
	})
	tests.Assert(t, err == nil)

	// Register n node again
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := n.Register(tx)
		tests.Assert(t, err == nil)

		return n.Save(tx)
	})
	tests.Assert(t, err == nil)
}

func TestNodeEntryRegisterStaleRegistration(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a registration entry in the db
	// but do not create an actual node entry
	req := &api.NodeAddRequest{
		ClusterId: "123",
		Hostnames: api.HostAddresses{
			Manage:  []string{"manage"},
			Storage: []string{"storage"},
		},
		Zone: 99,
	}
	n := NewNodeEntryFromRequest(req)

	// Only save the registration
	err := app.db.Update(func(tx *bolt.Tx) error {
		return n.Register(tx)
	})
	tests.Assert(t, err == nil)

	// Register node again.  This should
	// work because a real node entry is not saved
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := n.Register(tx)
		tests.Assert(t, err == nil)

		return n.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Register again.  Should not work
	err = app.db.Update(func(tx *bolt.Tx) error {
		return n.Register(tx)
	})
	tests.Assert(t, err != nil)

	// Remove n
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := n.Deregister(tx)
		tests.Assert(t, err == nil)

		return n.Delete(tx)
	})
	tests.Assert(t, err == nil)

	// Register n node again
	err = app.db.Update(func(tx *bolt.Tx) error {
		err := n.Register(tx)
		tests.Assert(t, err == nil)

		return n.Save(tx)
	})
	tests.Assert(t, err == nil)

}

func TestNewNodeEntryFromIdNotFound(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Test for ID not found
	err := app.db.View(func(tx *bolt.Tx) error {
		_, err := NewNodeEntryFromId(tx, "123")
		return err
	})
	tests.Assert(t, err == ErrNotFound)

}

func TestNewNodeEntryFromId(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a node
	req := &api.NodeAddRequest{
		ClusterId: "123",
		Hostnames: api.HostAddresses{
			Manage:  []string{"manage"},
			Storage: []string{"storage"},
		},
		Zone: 99,
	}

	n := NewNodeEntryFromRequest(req)
	n.DeviceAdd("abc")
	n.DeviceAdd("def")

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return n.Save(tx)
	})
	tests.Assert(t, err == nil)

	var node *NodeEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		node, err = NewNodeEntryFromId(tx, n.Info.Id)
		if err != nil {
			return err
		}
		return nil

	})
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(node, n))

}

func TestNewNodeEntrySaveDelete(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a node
	req := &api.NodeAddRequest{
		ClusterId: "123",
		Hostnames: api.HostAddresses{
			Manage:  []string{"manage"},
			Storage: []string{"storage"},
		},
		Zone: 99,
	}

	n := NewNodeEntryFromRequest(req)
	n.DeviceAdd("abc")
	n.DeviceAdd("def")

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return n.Save(tx)
	})
	tests.Assert(t, err == nil)

	var node *NodeEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		node, err = NewNodeEntryFromId(tx, n.Info.Id)
		if err != nil {
			return err
		}
		return nil

	})
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(node, n))

	// Delete entry which has devices
	err = app.db.Update(func(tx *bolt.Tx) error {
		var err error
		node, err = NewNodeEntryFromId(tx, n.Info.Id)
		if err != nil {
			return err
		}

		err = node.Delete(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == ErrConflict)

	// Delete devices in node
	node.DeviceDelete("abc")
	node.DeviceDelete("def")
	tests.Assert(t, len(node.Devices) == 0)
	err = app.db.Update(func(tx *bolt.Tx) error {
		return node.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Now try to delete the node
	err = app.db.Update(func(tx *bolt.Tx) error {
		var err error
		node, err = NewNodeEntryFromId(tx, n.Info.Id)
		if err != nil {
			return err
		}

		err = node.Delete(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)

	// Check node has been deleted and is not in db
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		node, err = NewNodeEntryFromId(tx, n.Info.Id)
		if err != nil {
			return err
		}
		return nil

	})
	tests.Assert(t, err == ErrNotFound)
}

func TestNewNodeEntryNewInfoResponse(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create a node
	req := &api.NodeAddRequest{
		ClusterId: "123",
		Hostnames: api.HostAddresses{
			Manage:  []string{"manage"},
			Storage: []string{"storage"},
		},
		Zone: 99,
	}

	n := NewNodeEntryFromRequest(req)

	// Save element in database
	err := app.db.Update(func(tx *bolt.Tx) error {
		return n.Save(tx)
	})
	tests.Assert(t, err == nil)

	var info *api.NodeInfoResponse
	err = app.db.View(func(tx *bolt.Tx) error {
		node, err := NewNodeEntryFromId(tx, n.Info.Id)
		if err != nil {
			return err
		}

		info, err = node.NewInfoReponse(tx)
		if err != nil {
			return err
		}

		return nil

	})
	tests.Assert(t, err == nil)

	tests.Assert(t, info.ClusterId == n.Info.ClusterId)
	tests.Assert(t, info.Id == n.Info.Id)
	tests.Assert(t, info.Zone == n.Info.Zone)
	tests.Assert(t, len(info.Hostnames.Manage) == 1)
	tests.Assert(t, len(info.Hostnames.Storage) == 1)
	tests.Assert(t, reflect.DeepEqual(info.Hostnames.Manage, n.Info.Hostnames.Manage))
	tests.Assert(t, reflect.DeepEqual(info.Hostnames.Storage, n.Info.Hostnames.Storage))
}

func TestNodeSetStateFailed(t *testing.T) {
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

		return nil
	})

	// Check ring
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 1)
	tests.Assert(t, mockAllocator.clustermap[c.Info.Id][0] == d.Info.Id)

	// Set failed
	err := n.SetState(app.db, app.executor, mockAllocator, api.EntryStateFailed)
	tests.Assert(t, n.State == api.EntryStateOnline)
	tests.Assert(t, err != nil)

	// Set offline
	err = n.SetState(app.db, app.executor, mockAllocator, api.EntryStateOffline)
	tests.Assert(t, n.State == api.EntryStateOffline)
	tests.Assert(t, err == nil)

	// Check it was removed from ring
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 0)

	// Set failed
	err = n.SetState(app.db, app.executor, mockAllocator, api.EntryStateFailed)
	tests.Assert(t, n.State == api.EntryStateFailed)
	tests.Assert(t, err == nil)

	// Set offline
	err = n.SetState(app.db, app.executor, mockAllocator, api.EntryStateOffline)
	tests.Assert(t, n.State == api.EntryStateFailed)
	tests.Assert(t, err != nil)
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 0)

	// Set online
	err = n.SetState(app.db, app.executor, mockAllocator, api.EntryStateOnline)
	tests.Assert(t, n.State == api.EntryStateFailed)
	tests.Assert(t, err != nil)
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 0)

}

func TestNodeSetStateOfflineOnline(t *testing.T) {
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
		return nil
	})

	// Check ring
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 1)
	tests.Assert(t, mockAllocator.clustermap[c.Info.Id][0] == d.Info.Id)

	// Set offline
	err := n.SetState(app.db, app.executor, mockAllocator, api.EntryStateOffline)
	tests.Assert(t, n.State == api.EntryStateOffline)
	tests.Assert(t, err == nil)

	// Check it was removed from ring
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 0)

	// Set offline again
	err = n.SetState(app.db, app.executor, mockAllocator, api.EntryStateOffline)
	tests.Assert(t, n.State == api.EntryStateOffline)
	tests.Assert(t, err == nil)

	// Set online
	err = n.SetState(app.db, app.executor, mockAllocator, api.EntryStateOnline)
	tests.Assert(t, n.State == api.EntryStateOnline)
	tests.Assert(t, err == nil)
	tests.Assert(t, len(mockAllocator.clustermap[c.Info.Id]) == 1)
	tests.Assert(t, mockAllocator.clustermap[c.Info.Id][0] == d.Info.Id)
}

func TestGetVerifiedManageHostname(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create cluster entry
	c := NewClusterEntry()
	c.Info.Id = "cluster"

	// Create a node
	req := &api.NodeAddRequest{
		ClusterId: "cluster",
		Hostnames: api.HostAddresses{
			Manage:  []string{"manage"},
			Storage: []string{"storage"},
		},
		Zone: 99,
	}

	n := NewNodeEntryFromRequest(req)
	c.NodeAdd(n.Info.Id)

	// Save in db
	app.db.Update(func(tx *bolt.Tx) error {
		err := c.Save(tx)
		tests.Assert(t, err == nil)

		err = n.Save(tx)
		tests.Assert(t, err == nil)

		return nil
	})

	hostname, err := GetVerifiedManageHostname(app.db, app.executor, "cluster")
	tests.Assert(t, hostname == "manage")
	tests.Assert(t, err == nil)
	tests.Assert(t, n.State == api.EntryStateOnline)

	app.db.Update(func(tx *bolt.Tx) error {
		// Set offline
		n.State = api.EntryStateOffline
		tests.Assert(t, n.State == api.EntryStateOffline)

		err := n.Save(tx)
		tests.Assert(t, err == nil)

		return nil

	})

	hostname, err = GetVerifiedManageHostname(app.db, app.executor, "cluster")
	tests.Assert(t, hostname != "manage")
	tests.Assert(t, err != nil)
}
