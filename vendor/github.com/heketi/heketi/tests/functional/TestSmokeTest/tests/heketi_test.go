// +build functional

//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//
package functional

import (
	"fmt"
	"net/http"
	"testing"

	client "github.com/heketi/heketi/client/api/go-client"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/heketi/pkg/utils/ssh"
	"github.com/heketi/tests"
)

// These are the settings for the vagrant file
const (

	// The heketi server must be running on the host
	heketiUrl = "http://localhost:8080"

	// VMs
	storage0 = "192.168.10.100"
	storage1 = "192.168.10.101"
	storage2 = "192.168.10.102"
	storage3 = "192.168.10.103"
)

var (
	// Heketi client
	heketi = client.NewClientNoAuth(heketiUrl)
	logger = utils.NewLogger("[test]", utils.LEVEL_DEBUG)

	// Storage systems
	storagevms = []string{
		storage0,
		storage1,
		storage2,
		storage3,
	}

	// Disks on each system
	disks = []string{
		"/dev/vdb",
		"/dev/vdc",
		"/dev/vdd",
		"/dev/vde",

		"/dev/vdf",
		"/dev/vdg",
		"/dev/vdh",
		"/dev/vdi",
	}
)

func setupCluster(t *testing.T, numNodes int, numDisks int) {
	tests.Assert(t, heketi != nil)

	// Create a cluster
	cluster, err := heketi.ClusterCreate()
	tests.Assert(t, err == nil)

	// hardcoded limits from the lists above
	// possible TODO: generalize
	tests.Assert(t, numNodes <= 4)
	tests.Assert(t, numDisks <= 8)

	// Add nodes
	for index, hostname := range storagevms[:numNodes] {
		nodeReq := &api.NodeAddRequest{}
		nodeReq.ClusterId = cluster.Id
		nodeReq.Hostnames.Manage = []string{hostname}
		nodeReq.Hostnames.Storage = []string{hostname}
		nodeReq.Zone = index%2 + 1

		node, err := heketi.NodeAdd(nodeReq)
		tests.Assert(t, err == nil)

		// Add devices
		sg := utils.NewStatusGroup()
		for _, disk := range disks[:numDisks] {
			sg.Add(1)
			go func(d string) {
				defer sg.Done()

				driveReq := &api.DeviceAddRequest{}
				driveReq.Name = d
				driveReq.NodeId = node.Id

				err := heketi.DeviceAdd(driveReq)
				sg.Err(err)
			}(disk)
		}

		err = sg.Result()
		tests.Assert(t, err == nil)
	}
}

func teardownCluster(t *testing.T) {
	clusters, err := heketi.ClusterList()
	tests.Assert(t, err == nil, err)

	for _, cluster := range clusters.Clusters {

		clusterInfo, err := heketi.ClusterInfo(cluster)
		tests.Assert(t, err == nil)

		// Delete volumes in this cluster
		for _, volume := range clusterInfo.Volumes {
			err := heketi.VolumeDelete(volume)
			tests.Assert(t, err == nil)
		}

		// Delete nodes
		for _, node := range clusterInfo.Nodes {

			// Get node information
			nodeInfo, err := heketi.NodeInfo(node)
			tests.Assert(t, err == nil)

			// Delete each device
			sg := utils.NewStatusGroup()
			for _, device := range nodeInfo.DevicesInfo {
				sg.Add(1)
				go func(id string) {
					defer sg.Done()

					err := heketi.DeviceDelete(id)
					sg.Err(err)

				}(device.Id)
			}
			err = sg.Result()
			tests.Assert(t, err == nil)

			// Delete node
			err = heketi.NodeDelete(node)
			tests.Assert(t, err == nil)
		}

		// Delete cluster
		err = heketi.ClusterDelete(cluster)
		tests.Assert(t, err == nil)
	}
}

func TestConnection(t *testing.T) {
	r, err := http.Get(heketiUrl + "/hello")
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusOK)
}

func TestHeketiSmokeTest(t *testing.T) {

	// Setup the VM storage topology
	teardownCluster(t)
	setupCluster(t, 4, 8)
	defer teardownCluster(t)

	// Create a volume and delete a few time to test garbage collection
	for i := 0; i < 2; i++ {

		volReq := &api.VolumeCreateRequest{}
		volReq.Size = 4000
		volReq.Snapshot.Enable = true
		volReq.Snapshot.Factor = 1.5
		volReq.Durability.Type = api.DurabilityReplicate

		volInfo, err := heketi.VolumeCreate(volReq)
		tests.Assert(t, err == nil)
		tests.Assert(t, volInfo.Size == 4000)
		tests.Assert(t, volInfo.Mount.GlusterFS.MountPoint != "")
		tests.Assert(t, volInfo.Name != "")

		volumes, err := heketi.VolumeList()
		tests.Assert(t, err == nil)
		tests.Assert(t, len(volumes.Volumes) == 1)
		tests.Assert(t, volumes.Volumes[0] == volInfo.Id)

		err = heketi.VolumeDelete(volInfo.Id)
		tests.Assert(t, err == nil)
	}

	// Create a 1TB volume
	volReq := &api.VolumeCreateRequest{}
	volReq.Size = 1024
	volReq.Snapshot.Enable = true
	volReq.Snapshot.Factor = 1.5
	volReq.Durability.Type = api.DurabilityReplicate

	simplevol, err := heketi.VolumeCreate(volReq)
	tests.Assert(t, err == nil)

	// Create a 12TB volume with 6TB of snapshot space
	// There should be no space
	volReq = &api.VolumeCreateRequest{}
	volReq.Size = 12 * 1024
	volReq.Snapshot.Enable = true
	volReq.Snapshot.Factor = 1.5
	volReq.Durability.Type = api.DurabilityReplicate

	_, err = heketi.VolumeCreate(volReq)
	tests.Assert(t, err != nil)

	// Check there is only one
	volumes, err := heketi.VolumeList()
	tests.Assert(t, err == nil)
	tests.Assert(t, len(volumes.Volumes) == 1)

	// Create a 100G volume with replica 3
	volReq = &api.VolumeCreateRequest{}
	volReq.Size = 100
	volReq.Durability.Type = api.DurabilityReplicate
	volReq.Durability.Replicate.Replica = 3

	volInfo, err := heketi.VolumeCreate(volReq)
	tests.Assert(t, err == nil)
	tests.Assert(t, volInfo.Size == 100)
	tests.Assert(t, volInfo.Mount.GlusterFS.MountPoint != "")
	tests.Assert(t, volInfo.Name != "")
	tests.Assert(t, len(volInfo.Bricks) == 3, len(volInfo.Bricks))

	// Check there are two volumes
	volumes, err = heketi.VolumeList()
	tests.Assert(t, err == nil)
	tests.Assert(t, len(volumes.Volumes) == 2)

	// Expand volume
	volExpReq := &api.VolumeExpandRequest{}
	volExpReq.Size = 2000

	volInfo, err = heketi.VolumeExpand(simplevol.Id, volExpReq)
	tests.Assert(t, err == nil)
	tests.Assert(t, volInfo.Size == simplevol.Size+2000)

	// Delete volume
	err = heketi.VolumeDelete(volInfo.Id)
	tests.Assert(t, err == nil)
}

func TestHeketiCreateVolumeWithGid(t *testing.T) {
	// Setup the VM storage topology
	teardownCluster(t)
	setupCluster(t, 4, 8)
	defer teardownCluster(t)

	// Create a volume
	volReq := &api.VolumeCreateRequest{}
	volReq.Size = 1024
	volReq.Durability.Type = api.DurabilityReplicate
	volReq.Durability.Replicate.Replica = 3
	volReq.Snapshot.Enable = true
	volReq.Snapshot.Factor = 1.5

	// Set to the vagrant gid
	volReq.Gid = 1000

	// Create the volume
	volInfo, err := heketi.VolumeCreate(volReq)
	tests.Assert(t, err == nil)

	// SSH into system and create two writers belonging to writegroup gid
	vagrantexec := ssh.NewSshExecWithKeyFile(logger, "vagrant", "../config/insecure_private_key")
	cmd := []string{
		"sudo groupadd writegroup",
		"sudo useradd writer1 -G writegroup -p$6$WBG5yf03$3DvyE41cicXEZDW.HDeJg3S4oEoELqKWoS/n6l28vorNxhIlcBe2SLQFDhqq6.Pq",
		"sudo useradd writer2 -G writegroup -p$6$WBG5yf03$3DvyE41cicXEZDW.HDeJg3S4oEoELqKWoS/n6l28vorNxhIlcBe2SLQFDhqq6.Pq",
		fmt.Sprintf("sudo mount -t glusterfs %v /mnt", volInfo.Mount.GlusterFS.MountPoint),
	}
	_, err = vagrantexec.ConnectAndExec("192.168.10.100:22", cmd, 10, true)
	tests.Assert(t, err == nil, err)

	writer1exec := ssh.NewSshExecWithPassword(logger, "writer1", "$6$WBG5yf03$3DvyE41cicXEZDW.HDeJg3S4oEoELqKWoS/n6l28vorNxhIlcBe2SLQFDhqq6.Pq")
	cmd = []string{
		"touch /mnt/writer1testfile",
		"mkdir /mnt/writer1dir",
		"touch /mnt/writer1dir/testfile",
	}
	_, err = writer1exec.ConnectAndExec("192.168.10.100:22", cmd, 10, false)
	tests.Assert(t, err == nil, err)

	writer2exec := ssh.NewSshExecWithPassword(logger, "writer2", "$6$WBG5yf03$3DvyE41cicXEZDW.HDeJg3S4oEoELqKWoS/n6l28vorNxhIlcBe2SLQFDhqq6.Pq")
	cmd = []string{
		"touch /mnt/writer2testfile",
		"mkdir /mnt/writer2dir",
		"touch /mnt/writer2dir/testfile",
	}
	_, err = writer2exec.ConnectAndExec("192.168.10.100:22", cmd, 10, false)
	tests.Assert(t, err == nil, err)
	cmd = []string{
		"mkdir /mnt/writer1dir/writer2subdir",
		"touch /mnt/writer1dir/writer2testfile",
	}
	_, err = writer2exec.ConnectAndExec("192.168.10.100:22", cmd, 10, false)
	tests.Assert(t, err == nil, err)

}

func TestRemoveDevice(t *testing.T) {

	// Setup the VM storage topology
	teardownCluster(t)
	setupCluster(t, 3, 2)
	defer teardownCluster(t)

	// We have 2 disks of 500GB on every node
	// Total space per node is 1TB
	// We have 3 Nodes, so total space is 3TB

	// vol1: 300 ==> 1 replica set
	// vol2: 600 ==> 4 replica sets of 150 each
	//               on each node:
	//               1 brick on the already used disk
	//               3 bricks on the previously unused disk
	//
	//             n1d1   n2d1   n3d1
	//       -------------------------
	//       r1: [ r1b1 , r1b2, r1b3 ]
	//
	//             n1d2   n2d2   n3d2
	//       -------------------------
	//       r2  [ r2b1,  r2b2,  r2b3 ]
	//       r3  [ r3b1,  r3b2   r3b4 ]
	//       r4  [ r4b1   r4b2   r4b3 ]

	volReq := &api.VolumeCreateRequest{}
	volReq.Size = 300
	volReq.Durability.Type = api.DurabilityReplicate
	volReq.Durability.Replicate.Replica = 3
	vol1, err := heketi.VolumeCreate(volReq)
	tests.Assert(t, err == nil)

	// Check there is only one
	volumes, err := heketi.VolumeList()
	tests.Assert(t, err == nil)
	tests.Assert(t, len(volumes.Volumes) == 1)

	volReq = &api.VolumeCreateRequest{}
	volReq.Size = 600
	volReq.Durability.Type = api.DurabilityReplicate
	volReq.Durability.Replicate.Replica = 3
	vol2, err := heketi.VolumeCreate(volReq)
	tests.Assert(t, err == nil)

	deviceOccurence := make(map[string]int)
	maxBricksPerDevice := 0
	var deviceToRemove string
	var diskNode string
	for _, brick := range vol2.Bricks {
		deviceOccurence[brick.DeviceId]++
		if deviceOccurence[brick.DeviceId] > maxBricksPerDevice {
			maxBricksPerDevice = deviceOccurence[brick.DeviceId]
			deviceToRemove = brick.DeviceId
			diskNode = brick.NodeId
		}
	}

	for device, _ := range deviceOccurence {
		logger.Info("Key: %v , Value: %v", device, deviceOccurence[device])
	}

	// if this fails, it's a problem with the test ...
	tests.Assert(t, maxBricksPerDevice > 1, "Problem: failed to produce a disk with multiple bricks from one volume!")

	// Add a replacement disk
	driveReq := &api.DeviceAddRequest{}
	driveReq.Name = disks[2]
	driveReq.NodeId = diskNode
	err = heketi.DeviceAdd(driveReq)
	tests.Assert(t, err == nil, err)

	stateReq := &api.StateRequest{}
	stateReq.State = api.EntryStateOffline
	err = heketi.DeviceState(deviceToRemove, stateReq)
	tests.Assert(t, err == nil)

	stateReq = &api.StateRequest{}
	stateReq.State = api.EntryStateFailed
	err = heketi.DeviceState(deviceToRemove, stateReq)
	tests.Assert(t, err == nil)

	logger.Info("%v %v", vol1, vol2)
	// Delete volumes
	err = heketi.VolumeDelete(vol1.Id)
	tests.Assert(t, err == nil)
	err = heketi.VolumeDelete(vol2.Id)
	tests.Assert(t, err == nil)
}

func TestRemoveDeviceVsVolumeCreate(t *testing.T) {

	// Setup the VM storage topology
	teardownCluster(t)
	setupCluster(t, 4, 1)
	defer teardownCluster(t)

	var newDevice string
	var deviceToRemove string

	volReq := &api.VolumeCreateRequest{}
	volReq.Size = 300
	volReq.Durability.Type = api.DurabilityReplicate
	volReq.Durability.Replicate.Replica = 3
	_, err := heketi.VolumeCreate(volReq)
	tests.Assert(t, err == nil)
	// Check there is only one
	volumes, err := heketi.VolumeList()
	tests.Assert(t, err == nil)
	tests.Assert(t, len(volumes.Volumes) == 1)

	clusters, err := heketi.ClusterList()
	tests.Assert(t, err == nil, err)
	for _, cluster := range clusters.Clusters {
		clusterInfo, err := heketi.ClusterInfo(cluster)
		tests.Assert(t, err == nil)

		for _, node := range clusterInfo.Nodes {

			// Get node information
			nodeInfo, err := heketi.NodeInfo(node)
			tests.Assert(t, err == nil)
			for _, device := range nodeInfo.DevicesInfo {
				if len(device.Bricks) == 0 {
					newDevice = device.Id
				} else {
					deviceToRemove = device.Id
				}
			}
		}
	}

	stateReq := &api.StateRequest{}
	stateReq.State = api.EntryStateOffline
	err = heketi.DeviceState(deviceToRemove, stateReq)
	tests.Assert(t, err == nil)

	sgDeviceRemove := utils.NewStatusGroup()
	sgDeviceRemove.Add(1)
	go func() {
		defer sgDeviceRemove.Done()
		stateReq = &api.StateRequest{}
		stateReq.State = api.EntryStateFailed
		err = heketi.DeviceState(deviceToRemove, stateReq)
		sgDeviceRemove.Err(err)
	}()

	sgVolumeCreate := utils.NewStatusGroup()
	for i := 0; i < 15; i++ {
		sgVolumeCreate.Add(1)
		go func() {
			defer sgVolumeCreate.Done()
			volReq = &api.VolumeCreateRequest{}
			volReq.Size = 10
			volReq.Durability.Type = api.DurabilityReplicate
			volReq.Durability.Replicate.Replica = 3
			_, err := heketi.VolumeCreate(volReq)
			sgVolumeCreate.Err(err)
		}()
	}

	err = sgVolumeCreate.Result()
	tests.Assert(t, err == nil)
	err = sgDeviceRemove.Result()
	tests.Assert(t, err == nil)
	// At this point, we should have one brick moved to new device as a result of remove device
	// and 15 bricks created on new device as a result of 15 volume creates
	newDeviceResponse, err := heketi.DeviceInfo(newDevice)
	tests.Assert(t, len(newDeviceResponse.Bricks) == 16, "device entry not consistent")

}

func TestHeketiVolumeExpandWithGid(t *testing.T) {
	// Setup the VM storage topology
	teardownCluster(t)
	setupCluster(t, 4, 8)
	defer teardownCluster(t)

	// Create a volume
	volReq := &api.VolumeCreateRequest{}
	volReq.Size = 300
	volReq.Durability.Type = api.DurabilityReplicate
	volReq.Durability.Replicate.Replica = 3
	volReq.Snapshot.Enable = true
	volReq.Snapshot.Factor = 1.5

	// Set to the vagrant gid
	volReq.Gid = 2333

	// Create the volume
	volInfo, err := heketi.VolumeCreate(volReq)
	tests.Assert(t, err == nil)

	// Expand volume
	volExpReq := &api.VolumeExpandRequest{}
	volExpReq.Size = 300

	newVolInfo, err := heketi.VolumeExpand(volInfo.Id, volExpReq)
	tests.Assert(t, err == nil)
	tests.Assert(t, newVolInfo.Size == volInfo.Size+300)

	// SSH into system and check gid of bricks
	vagrantexec := ssh.NewSshExecWithKeyFile(logger, "vagrant", "../config/insecure_private_key")
	cmd := []string{
		fmt.Sprintf("sudo ls -l /var/lib/heketi/mounts/vg_*/brick_*/  | grep  -e \"^d\" | cut -d\" \" -f4 | grep -q %v", volReq.Gid),
	}
	_, err = vagrantexec.ConnectAndExec("192.168.10.100:22", cmd, 10, true)
	tests.Assert(t, err == nil, "Brick found with different Gid")
	_, err = vagrantexec.ConnectAndExec("192.168.10.101:22", cmd, 10, true)
	tests.Assert(t, err == nil, "Brick found with different Gid")
	_, err = vagrantexec.ConnectAndExec("192.168.10.102:22", cmd, 10, true)
	tests.Assert(t, err == nil, "Brick found with different Gid")
	_, err = vagrantexec.ConnectAndExec("192.168.10.103:22", cmd, 10, true)
	tests.Assert(t, err == nil, "Brick found with different Gid")

}

func TestHeketiVolumeCreateWithOptions(t *testing.T) {
	// Setup the VM storage topology
	teardownCluster(t)
	setupCluster(t, 2, 2)
	defer teardownCluster(t)

	// Create a volume
	volReq := &api.VolumeCreateRequest{}
	volReq.Size = 10
	volReq.Durability.Type = api.DurabilityReplicate
	volReq.Durability.Replicate.Replica = 2
	volReq.Snapshot.Enable = true
	volReq.Snapshot.Factor = 1.5
	volReq.GlusterVolumeOptions = []string{"performance.rda-cache-limit 10MB","performance.nl-cache-positive-entry no"}

	// Set to the vagrant gid
	volReq.Gid = 2333

	// Create the volume
	volInfo, err := heketi.VolumeCreate(volReq)
	tests.Assert(t, err == nil)
	tests.Assert(t, len(volInfo.GlusterVolumeOptions) > 0)

	// SSH into system and check volume options.
	vagrantexec := ssh.NewSshExecWithKeyFile(logger, "vagrant", "../config/insecure_private_key")
	cmd := []string{
		fmt.Sprintf("sudo gluster v info %v | grep performance.rda-cache-limit | grep 10MB", volInfo.Name),
		fmt.Sprintf("sudo gluster v info %v | grep performance.nl-cache-positive-entry | grep no", volInfo.Name),
	}
	_, err = vagrantexec.ConnectAndExec("192.168.10.100:22", cmd, 10, true)
	tests.Assert(t, err == nil, "Volume Created with specified options")
}
