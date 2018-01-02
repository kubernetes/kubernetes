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
	"testing"

	"github.com/boltdb/bolt"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
)

func TestNewSimpleAllocator(t *testing.T) {

	a := NewSimpleAllocator()
	tests.Assert(t, a != nil)
	tests.Assert(t, a.rings != nil)

}

func TestSimpleAllocatorEmpty(t *testing.T) {
	a := NewSimpleAllocator()
	tests.Assert(t, a != nil)

	err := a.RemoveDevice(createSampleClusterEntry(),
		createSampleNodeEntry(),
		createSampleDeviceEntry("aaa", 10))
	tests.Assert(t, err == ErrNotFound)

	err = a.RemoveCluster("aaa")
	tests.Assert(t, err == ErrNotFound)

	ch, _, errc := a.GetNodes(utils.GenUUID(), utils.GenUUID())
	for d := range ch {
		tests.Assert(t, false, d)
	}
	err = <-errc
	tests.Assert(t, err == ErrNotFound)
}

func TestSimpleAllocatorAddRemoveDevice(t *testing.T) {
	a := NewSimpleAllocator()
	tests.Assert(t, a != nil)

	cluster := createSampleClusterEntry()
	node := createSampleNodeEntry()
	node.Info.ClusterId = cluster.Info.Id
	device := createSampleDeviceEntry(node.Info.Id, 10000)

	tests.Assert(t, len(a.rings) == 0)
	err := a.AddDevice(cluster, node, device)
	tests.Assert(t, err == nil)
	tests.Assert(t, len(a.rings) == 1)
	tests.Assert(t, a.rings[cluster.Info.Id] != nil)

	// Get the nodes from the ring
	ch, _, errc := a.GetNodes(cluster.Info.Id, utils.GenUUID())

	var devices int
	for d := range ch {
		devices++
		tests.Assert(t, d == device.Info.Id)
	}
	err = <-errc
	tests.Assert(t, devices == 1)
	tests.Assert(t, err == nil)

	// Now remove the device
	err = a.RemoveDevice(cluster, node, device)
	tests.Assert(t, err == nil)
	tests.Assert(t, len(a.rings) == 1)

	// Get the nodes from the ring
	ch, _, errc = a.GetNodes(cluster.Info.Id, utils.GenUUID())

	devices = 0
	for d := range ch {
		devices++
		tests.Assert(t, false, d)
	}
	err = <-errc
	tests.Assert(t, devices == 0)
	tests.Assert(t, err == nil)

}

func TestSimpleAllocatorInitFromDb(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Setup database
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create large cluster
	err := setupSampleDbWithTopology(app,
		1,      // clusters
		10,     // nodes_per_cluster
		20,     // devices_per_node,
		600*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Get the cluster list
	var clusterId string
	err = app.db.View(func(tx *bolt.Tx) error {
		clusters, err := ClusterList(tx)
		if err != nil {
			return err
		}
		tests.Assert(t, len(clusters) == 1)
		clusterId = clusters[0]

		return nil
	})
	tests.Assert(t, err == nil)

	// Create an allocator and initialize it from the DB
	a := NewSimpleAllocatorFromDb(app.db)
	tests.Assert(t, a != nil)

	// Get the nodes from the ring
	ch, _, errc := a.GetNodes(clusterId, utils.GenUUID())

	var devices int
	for d := range ch {
		devices++
		tests.Assert(t, d != "")
	}
	err = <-errc
	tests.Assert(t, devices == 10*20)
	tests.Assert(t, err == nil)

}

func TestSimpleAllocatorInitFromDbWithOfflineDevices(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Setup database
	app := NewTestApp(tmpfile)
	defer app.Close()

	// Create large cluster
	err := setupSampleDbWithTopology(app,
		1,      // clusters
		2,      // nodes_per_cluster
		4,      // devices_per_node,
		600*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Get the cluster list
	var clusterId, nodeId string
	err = app.db.Update(func(tx *bolt.Tx) error {
		clusters, err := ClusterList(tx)
		if err != nil {
			return err
		}
		tests.Assert(t, len(clusters) == 1)
		clusterId = clusters[0]

		cluster, err := NewClusterEntryFromId(tx, clusterId)
		tests.Assert(t, err == nil)

		// make one node offline, which will mean none of its
		// devices are added to the ring
		node, err := cluster.NodeEntryFromClusterIndex(tx, 0)
		tests.Assert(t, err == nil)
		nodeId = node.Info.Id
		node.State = api.EntryStateOffline
		node.Save(tx)

		// Make only one device offline in the other node
		node, err = cluster.NodeEntryFromClusterIndex(tx, 1)
		device, err := NewDeviceEntryFromId(tx, node.Devices[0])
		device.State = api.EntryStateOffline
		device.Save(tx)

		return nil
	})
	tests.Assert(t, err == nil)

	// Create an allocator and initialize it from the DB
	a := NewSimpleAllocatorFromDb(app.db)
	tests.Assert(t, a != nil)

	// Get the nodes from the ring
	ch, _, errc := a.GetNodes(clusterId, utils.GenUUID())

	var devices int
	for d := range ch {
		devices++
		tests.Assert(t, d != "")
	}
	err = <-errc
	tests.Assert(t, err == nil)

	// Only three online devices should be in the list
	tests.Assert(t, devices == 3, devices)

}
