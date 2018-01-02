// +build acceptance blockstorage

package extensions

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v2/volumes"

	blockstorage "github.com/gophercloud/gophercloud/acceptance/openstack/blockstorage/v2"
	compute "github.com/gophercloud/gophercloud/acceptance/openstack/compute/v2"
)

func TestVolumeActionsUploadImageDestroy(t *testing.T) {
	blockClient, err := clients.NewBlockStorageV2Client()
	if err != nil {
		t.Fatalf("Unable to create a blockstorage client: %v", err)
	}
	computeClient, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	volume, err := blockstorage.CreateVolume(t, blockClient)
	if err != nil {
		t.Fatalf("Unable to create volume: %v", err)
	}
	defer blockstorage.DeleteVolume(t, blockClient, volume)

	volumeImage, err := CreateUploadImage(t, blockClient, volume)
	if err != nil {
		t.Fatalf("Unable to upload volume-backed image: %v", err)
	}

	tools.PrintResource(t, volumeImage)

	err = DeleteUploadedImage(t, computeClient, volumeImage.ImageName)
	if err != nil {
		t.Fatalf("Unable to delete volume-backed image: %v", err)
	}
}

func TestVolumeActionsAttachCreateDestroy(t *testing.T) {
	blockClient, err := clients.NewBlockStorageV2Client()
	if err != nil {
		t.Fatalf("Unable to create a blockstorage client: %v", err)
	}

	computeClient, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	server, err := compute.CreateServer(t, computeClient)
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}
	defer compute.DeleteServer(t, computeClient, server)

	volume, err := blockstorage.CreateVolume(t, blockClient)
	if err != nil {
		t.Fatalf("Unable to create volume: %v", err)
	}
	defer blockstorage.DeleteVolume(t, blockClient, volume)

	err = CreateVolumeAttach(t, blockClient, volume, server)
	if err != nil {
		t.Fatalf("Unable to attach volume: %v", err)
	}

	newVolume, err := volumes.Get(blockClient, volume.ID).Extract()
	if err != nil {
		t.Fatal("Unable to get updated volume information: %v", err)
	}

	DeleteVolumeAttach(t, blockClient, newVolume)
}

func TestVolumeActionsReserveUnreserve(t *testing.T) {
	client, err := clients.NewBlockStorageV2Client()
	if err != nil {
		t.Fatalf("Unable to create blockstorage client: %v", err)
	}

	volume, err := blockstorage.CreateVolume(t, client)
	if err != nil {
		t.Fatalf("Unable to create volume: %v", err)
	}
	defer blockstorage.DeleteVolume(t, client, volume)

	err = CreateVolumeReserve(t, client, volume)
	if err != nil {
		t.Fatalf("Unable to create volume reserve: %v", err)
	}
	defer DeleteVolumeReserve(t, client, volume)
}

func TestVolumeActionsExtendSize(t *testing.T) {
	blockClient, err := clients.NewBlockStorageV2Client()
	if err != nil {
		t.Fatalf("Unable to create a blockstorage client: %v", err)
	}

	volume, err := blockstorage.CreateVolume(t, blockClient)
	if err != nil {
		t.Fatalf("Unable to create volume: %v", err)
	}
	defer blockstorage.DeleteVolume(t, blockClient, volume)

	tools.PrintResource(t, volume)

	err = ExtendVolumeSize(t, blockClient, volume)
	if err != nil {
		t.Fatalf("Unable to resize volume: %v", err)
	}

	newVolume, err := volumes.Get(blockClient, volume.ID).Extract()
	if err != nil {
		t.Fatal("Unable to get updated volume information: %v", err)
	}

	tools.PrintResource(t, newVolume)
}

// Note(jtopjian): I plan to work on this at some point, but it requires
// setting up a server with iscsi utils.
/*
func TestVolumeConns(t *testing.T) {
    client, err := newClient()
    th.AssertNoErr(t, err)

    t.Logf("Creating volume")
    cv, err := volumes.Create(client, &volumes.CreateOpts{
        Size: 1,
        Name: "blockv2-volume",
    }).Extract()
    th.AssertNoErr(t, err)

    defer func() {
        err = volumes.WaitForStatus(client, cv.ID, "available", 60)
        th.AssertNoErr(t, err)

        t.Logf("Deleting volume")
        err = volumes.Delete(client, cv.ID).ExtractErr()
        th.AssertNoErr(t, err)
    }()

    err = volumes.WaitForStatus(client, cv.ID, "available", 60)
    th.AssertNoErr(t, err)

    connOpts := &volumeactions.ConnectorOpts{
        IP:        "127.0.0.1",
        Host:      "stack",
        Initiator: "iqn.1994-05.com.redhat:17cf566367d2",
        Multipath: false,
        Platform:  "x86_64",
        OSType:    "linux2",
    }

    t.Logf("Initializing connection")
    _, err = volumeactions.InitializeConnection(client, cv.ID, connOpts).Extract()
    th.AssertNoErr(t, err)

    t.Logf("Terminating connection")
    err = volumeactions.TerminateConnection(client, cv.ID, connOpts).ExtractErr()
    th.AssertNoErr(t, err)
}
*/
