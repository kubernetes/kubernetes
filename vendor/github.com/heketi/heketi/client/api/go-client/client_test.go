//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), as published by the Free Software Foundation,
// or under the Apache License, Version 2.0 <LICENSE-APACHE2 or
// http://www.apache.org/licenses/LICENSE-2.0>.
//
// You may not use this file except in compliance with those terms.
//

package client

import (
	"fmt"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"

	"github.com/gorilla/mux"
	"github.com/heketi/heketi/apps/glusterfs"
	"github.com/heketi/heketi/middleware"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
	"github.com/urfave/negroni"
)

const (
	TEST_ADMIN_KEY = "adminkey"
)

func setupHeketiServer(app *glusterfs.App) *httptest.Server {
	router := mux.NewRouter()
	app.SetRoutes(router)
	n := negroni.New()

	jwtconfig := &middleware.JwtAuthConfig{}
	jwtconfig.Admin.PrivateKey = TEST_ADMIN_KEY
	jwtconfig.User.PrivateKey = "userkey"

	// Setup middleware
	n.Use(middleware.NewJwtAuth(jwtconfig))
	n.UseFunc(app.Auth)
	n.UseHandler(router)

	// Create server
	return httptest.NewServer(n)
}

func TestTopology(t *testing.T) {
	db := tests.Tempfile()
	defer os.Remove(db)

	// Create the app
	app := glusterfs.NewTestApp(db)
	defer app.Close()

	// Setup the server
	ts := setupHeketiServer(app)
	defer ts.Close()

	// Create cluster correctly
	c := NewClient(ts.URL, "admin", TEST_ADMIN_KEY)
	tests.Assert(t, c != nil)

	//Create multiple clusters
	clusteridlist := make([]api.ClusterInfoResponse, 0)
	for m := 0; m < 4; m++ {
		cluster, err := c.ClusterCreate()
		tests.Assert(t, err == nil)
		tests.Assert(t, cluster.Id != "")
		clusteridlist = append(clusteridlist, *cluster)
	}
	tests.Assert(t, len(clusteridlist) == 4)

	//Verify the topology info and then delete the clusters
	topology, err := c.TopologyInfo()
	tests.Assert(t, err == nil)
	for _, cid := range topology.ClusterList {
		clusterid := cid.Id
		err = c.ClusterDelete(clusterid)
		tests.Assert(t, err == nil)
	}

	//Create a cluster and add multiple nodes,devices and volumes
	cluster, err := c.ClusterCreate()
	tests.Assert(t, err == nil)
	tests.Assert(t, cluster.Id != "")
	tests.Assert(t, len(cluster.Nodes) == 0)
	tests.Assert(t, len(cluster.Volumes) == 0)

	// Get information about the client
	clusterinfo, err := c.ClusterInfo(cluster.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(clusterinfo, cluster))

	// Get information about the Topology and verify the cluster creation
	topology, err = c.TopologyInfo()
	tests.Assert(t, err == nil)
	tests.Assert(t, topology.ClusterList[0].Id == cluster.Id)

	// Create multiple nodes and add devices to the nodes
	nodeinfos := make([]api.NodeInfoResponse, 0)
	for n := 0; n < 4; n++ {
		nodeReq := &api.NodeAddRequest{}
		nodeReq.ClusterId = cluster.Id
		nodeReq.Hostnames.Manage = []string{"manage" + fmt.Sprintf("%v", n)}
		nodeReq.Hostnames.Storage = []string{"storage" + fmt.Sprintf("%v", n)}
		nodeReq.Zone = n + 1

		// Create node
		node, err := c.NodeAdd(nodeReq)
		nodeinfos = append(nodeinfos, *node)
		tests.Assert(t, err == nil)

		// Create a device request
		sg := utils.NewStatusGroup()
		for i := 0; i < 50; i++ {
			sg.Add(1)
			go func() {
				defer sg.Done()

				deviceReq := &api.DeviceAddRequest{}
				deviceReq.Name = "sd" + utils.GenUUID()[:8]
				deviceReq.NodeId = node.Id

				// Create device
				err := c.DeviceAdd(deviceReq)
				sg.Err(err)
			}()
		}
		tests.Assert(t, sg.Result() == nil)
	}
	tests.Assert(t, len(nodeinfos) != 0)

	// Get list of volumes
	list, err := c.VolumeList()
	tests.Assert(t, err == nil)
	tests.Assert(t, len(list.Volumes) == 0)

	//Create multiple volumes to the cluster
	volumeinfos := make([]api.VolumeInfoResponse, 0)
	for n := 0; n < 4; n++ {
		volumeReq := &api.VolumeCreateRequest{}
		volumeReq.Size = 10
		volume, err := c.VolumeCreate(volumeReq)
		tests.Assert(t, err == nil)
		tests.Assert(t, volume.Id != "")
		tests.Assert(t, volume.Size == volumeReq.Size)
		volumeinfos = append(volumeinfos, *volume)
	}
	topology, err = c.TopologyInfo()
	tests.Assert(t, err == nil)

	// Test topology have all the existing volumes
	var volumefound int
	for _, volumeid := range topology.ClusterList[0].Volumes {
		volumeInfo := volumeid
		for _, singlevolumei := range volumeinfos {
			if singlevolumei.Id == volumeInfo.Id {
				volumefound++
				break
			}
		}
	}
	tests.Assert(t, volumefound == 4)

	// Delete all the volumes
	for _, volumeid := range topology.ClusterList[0].Volumes {
		volumeInfo := volumeid
		err = c.VolumeDelete(volumeInfo.Id)
		tests.Assert(t, err == nil)

	}

	// Verify the nodes and devices info from topology info and delete the entries
	for _, nodeid := range topology.ClusterList[0].Nodes {
		nodeInfo := nodeid
		var found bool
		for _, singlenodei := range nodeinfos {
			found = false
			if singlenodei.Id == nodeInfo.Id {
				found = true
				break
			}
		}
		tests.Assert(t, found == true)

		// Delete all devices
		sg := utils.NewStatusGroup()
		for index := range nodeInfo.DevicesInfo {
			sg.Add(1)
			go func(i int) {
				defer sg.Done()
				sg.Err(c.DeviceDelete(nodeInfo.DevicesInfo[i].Id))
			}(index)
		}
		err = sg.Result()
		tests.Assert(t, err == nil, err)

		// Delete node
		err = c.NodeDelete(nodeInfo.Id)
		tests.Assert(t, err == nil)

	}

	// Delete cluster
	err = c.ClusterDelete(cluster.Id)
	tests.Assert(t, err == nil)

}

func TestClientCluster(t *testing.T) {
	db := tests.Tempfile()
	defer os.Remove(db)

	// Create the app
	app := glusterfs.NewTestApp(db)
	defer app.Close()

	// Setup the server
	ts := setupHeketiServer(app)
	defer ts.Close()

	// Create cluster with unknown user
	c := NewClient(ts.URL, "asdf", "")
	tests.Assert(t, c != nil)
	cluster, err := c.ClusterCreate()
	tests.Assert(t, err != nil)
	tests.Assert(t, cluster == nil)

	// Create cluster with bad password
	c = NewClient(ts.URL, "admin", "badpassword")
	tests.Assert(t, c != nil)
	cluster, err = c.ClusterCreate()
	tests.Assert(t, err != nil)
	tests.Assert(t, cluster == nil)

	// Create cluster correctly
	c = NewClient(ts.URL, "admin", TEST_ADMIN_KEY)
	tests.Assert(t, c != nil)
	cluster, err = c.ClusterCreate()
	tests.Assert(t, err == nil)
	tests.Assert(t, cluster.Id != "")
	tests.Assert(t, len(cluster.Nodes) == 0)
	tests.Assert(t, len(cluster.Volumes) == 0)

	// Request bad id
	info, err := c.ClusterInfo("bad")
	tests.Assert(t, err != nil)
	tests.Assert(t, info == nil)

	// Get information about the client
	info, err = c.ClusterInfo(cluster.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(info, cluster))

	// Get a list of clusters
	list, err := c.ClusterList()
	tests.Assert(t, err == nil)
	tests.Assert(t, len(list.Clusters) == 1)
	tests.Assert(t, list.Clusters[0] == info.Id)

	// Delete non-existent cluster
	err = c.ClusterDelete("badid")
	tests.Assert(t, err != nil)

	// Delete current cluster
	err = c.ClusterDelete(info.Id)
	tests.Assert(t, err == nil)
}

func TestClientNode(t *testing.T) {
	db := tests.Tempfile()
	defer os.Remove(db)

	// Create the app
	app := glusterfs.NewTestApp(db)
	defer app.Close()

	// Setup the server
	ts := setupHeketiServer(app)
	defer ts.Close()

	// Create cluster
	c := NewClient(ts.URL, "admin", TEST_ADMIN_KEY)
	tests.Assert(t, c != nil)
	cluster, err := c.ClusterCreate()
	tests.Assert(t, err == nil)
	tests.Assert(t, cluster.Id != "")
	tests.Assert(t, len(cluster.Nodes) == 0)
	tests.Assert(t, len(cluster.Volumes) == 0)

	// Add node to unknown cluster
	nodeReq := &api.NodeAddRequest{}
	nodeReq.ClusterId = "badid"
	nodeReq.Hostnames.Manage = []string{"manage"}
	nodeReq.Hostnames.Storage = []string{"storage"}
	nodeReq.Zone = 10
	_, err = c.NodeAdd(nodeReq)
	tests.Assert(t, err != nil)

	// Create node request packet
	nodeReq.ClusterId = cluster.Id
	node, err := c.NodeAdd(nodeReq)
	tests.Assert(t, err == nil)
	tests.Assert(t, node.Zone == nodeReq.Zone)
	tests.Assert(t, node.State == api.EntryStateOnline)
	tests.Assert(t, node.Id != "")
	tests.Assert(t, reflect.DeepEqual(nodeReq.Hostnames, node.Hostnames))
	tests.Assert(t, len(node.DevicesInfo) == 0)

	// Info on invalid id
	info, err := c.NodeInfo("badid")
	tests.Assert(t, err != nil)
	tests.Assert(t, info == nil)

	// Set offline
	err = c.NodeState(node.Id, &api.StateRequest{
		State: api.EntryStateOffline,
	})
	tests.Assert(t, err == nil)

	// Get node info
	info, err = c.NodeInfo(node.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, info.State == api.EntryStateOffline)

	// Set online
	err = c.NodeState(node.Id, &api.StateRequest{
		State: api.EntryStateOnline,
	})
	tests.Assert(t, err == nil)

	// Get node info
	info, err = c.NodeInfo(node.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, info.State == api.EntryStateOnline)
	tests.Assert(t, reflect.DeepEqual(info, node))

	// Delete invalid node
	err = c.NodeDelete("badid")
	tests.Assert(t, err != nil)

	// Can't delete cluster with a node
	err = c.ClusterDelete(cluster.Id)
	tests.Assert(t, err != nil)

	// Delete node
	err = c.NodeDelete(node.Id)
	tests.Assert(t, err == nil)

	// Delete cluster
	err = c.ClusterDelete(cluster.Id)
	tests.Assert(t, err == nil)

}

func TestClientDevice(t *testing.T) {
	db := tests.Tempfile()
	defer os.Remove(db)

	// Create the app
	app := glusterfs.NewTestApp(db)
	defer app.Close()

	// Setup the server
	ts := setupHeketiServer(app)
	defer ts.Close()

	// Create cluster
	c := NewClient(ts.URL, "admin", TEST_ADMIN_KEY)
	tests.Assert(t, c != nil)
	cluster, err := c.ClusterCreate()
	tests.Assert(t, err == nil)

	// Create node request packet
	nodeReq := &api.NodeAddRequest{}
	nodeReq.ClusterId = cluster.Id
	nodeReq.Hostnames.Manage = []string{"manage"}
	nodeReq.Hostnames.Storage = []string{"storage"}
	nodeReq.Zone = 10

	// Create node
	node, err := c.NodeAdd(nodeReq)
	tests.Assert(t, err == nil)

	// Create a device request
	deviceReq := &api.DeviceAddRequest{}
	deviceReq.Name = "sda"
	deviceReq.NodeId = node.Id

	// Create device
	err = c.DeviceAdd(deviceReq)
	tests.Assert(t, err == nil)

	// Get node information
	info, err := c.NodeInfo(node.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, len(info.DevicesInfo) == 1)
	tests.Assert(t, len(info.DevicesInfo[0].Bricks) == 0)
	tests.Assert(t, info.DevicesInfo[0].Name == deviceReq.Name)
	tests.Assert(t, info.DevicesInfo[0].Id != "")

	// Get info from an unknown id
	_, err = c.DeviceInfo("badid")
	tests.Assert(t, err != nil)

	// Get device information
	deviceId := info.DevicesInfo[0].Id
	deviceInfo, err := c.DeviceInfo(deviceId)
	tests.Assert(t, err == nil)
	tests.Assert(t, deviceInfo.State == api.EntryStateOnline)
	tests.Assert(t, reflect.DeepEqual(*deviceInfo, info.DevicesInfo[0]))

	// Set offline
	err = c.DeviceState(deviceId, &api.StateRequest{
		State: api.EntryStateOffline,
	})
	tests.Assert(t, err == nil)
	deviceInfo, err = c.DeviceInfo(deviceId)
	tests.Assert(t, err == nil)
	tests.Assert(t, deviceInfo.State == api.EntryStateOffline)

	// Set online
	err = c.DeviceState(deviceId, &api.StateRequest{
		State: api.EntryStateOnline,
	})
	tests.Assert(t, err == nil)
	deviceInfo, err = c.DeviceInfo(deviceId)
	tests.Assert(t, err == nil)
	tests.Assert(t, deviceInfo.State == api.EntryStateOnline)

	// Try to delete node, and will not until we delete the device
	err = c.NodeDelete(node.Id)
	tests.Assert(t, err != nil)

	// Delete unknown device
	err = c.DeviceDelete("badid")
	tests.Assert(t, err != nil)

	// Delete device
	err = c.DeviceDelete(deviceInfo.Id)
	tests.Assert(t, err == nil)

	// Delete node
	err = c.NodeDelete(node.Id)
	tests.Assert(t, err == nil)

	// Delete cluster
	err = c.ClusterDelete(cluster.Id)
	tests.Assert(t, err == nil)

}

func TestClientVolume(t *testing.T) {
	db := tests.Tempfile()
	defer os.Remove(db)

	// Create the app
	app := glusterfs.NewTestApp(db)
	defer app.Close()

	// Setup the server
	ts := setupHeketiServer(app)
	defer ts.Close()

	// Create cluster
	c := NewClient(ts.URL, "admin", TEST_ADMIN_KEY)
	tests.Assert(t, c != nil)
	cluster, err := c.ClusterCreate()
	tests.Assert(t, err == nil)

	// Create node request packet
	for n := 0; n < 4; n++ {
		nodeReq := &api.NodeAddRequest{}
		nodeReq.ClusterId = cluster.Id
		nodeReq.Hostnames.Manage = []string{"manage" + fmt.Sprintf("%v", n)}
		nodeReq.Hostnames.Storage = []string{"storage" + fmt.Sprintf("%v", n)}
		nodeReq.Zone = n + 1

		// Create node
		node, err := c.NodeAdd(nodeReq)
		tests.Assert(t, err == nil)

		// Create a device request
		sg := utils.NewStatusGroup()
		for i := 0; i < 50; i++ {
			sg.Add(1)
			go func() {
				defer sg.Done()

				deviceReq := &api.DeviceAddRequest{}
				deviceReq.Name = "sd" + utils.GenUUID()[:8]
				deviceReq.NodeId = node.Id

				// Create device
				err := c.DeviceAdd(deviceReq)
				sg.Err(err)

			}()
		}
		tests.Assert(t, sg.Result() == nil)
	}

	// Get list of volumes
	list, err := c.VolumeList()
	tests.Assert(t, err == nil)
	tests.Assert(t, len(list.Volumes) == 0)

	// Create a volume
	volumeReq := &api.VolumeCreateRequest{}
	volumeReq.Size = 10
	volume, err := c.VolumeCreate(volumeReq)
	tests.Assert(t, err == nil)
	tests.Assert(t, volume.Id != "")
	tests.Assert(t, volume.Size == volumeReq.Size)

	// Get list of volumes
	list, err = c.VolumeList()
	tests.Assert(t, err == nil)
	tests.Assert(t, len(list.Volumes) == 1)
	tests.Assert(t, list.Volumes[0] == volume.Id)

	// Get info on incorrect id
	info, err := c.VolumeInfo("badid")
	tests.Assert(t, err != nil)

	// Get info
	info, err = c.VolumeInfo(volume.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, reflect.DeepEqual(info, volume))

	// Expand volume with a bad id
	expandReq := &api.VolumeExpandRequest{}
	expandReq.Size = 10
	volumeInfo, err := c.VolumeExpand("badid", expandReq)
	tests.Assert(t, err != nil)

	// Expand volume
	volumeInfo, err = c.VolumeExpand(volume.Id, expandReq)
	tests.Assert(t, err == nil)
	tests.Assert(t, volumeInfo.Size == 20)

	// Delete bad id
	err = c.VolumeDelete("badid")
	tests.Assert(t, err != nil)

	// Delete volume
	err = c.VolumeDelete(volume.Id)
	tests.Assert(t, err == nil)

	clusterInfo, err := c.ClusterInfo(cluster.Id)
	for _, nodeid := range clusterInfo.Nodes {
		// Get node information
		nodeInfo, err := c.NodeInfo(nodeid)
		tests.Assert(t, err == nil)

		// Delete all devices
		sg := utils.NewStatusGroup()
		for index := range nodeInfo.DevicesInfo {
			sg.Add(1)
			go func(i int) {
				defer sg.Done()
				sg.Err(c.DeviceDelete(nodeInfo.DevicesInfo[i].Id))
			}(index)
		}
		err = sg.Result()
		tests.Assert(t, err == nil, err)

		// Delete node
		err = c.NodeDelete(nodeid)
		tests.Assert(t, err == nil)

	}

	// Delete cluster
	err = c.ClusterDelete(cluster.Id)
	tests.Assert(t, err == nil)

}
