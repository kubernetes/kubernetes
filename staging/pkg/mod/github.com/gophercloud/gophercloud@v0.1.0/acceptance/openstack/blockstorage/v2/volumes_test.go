// +build acceptance blockstorage

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/extensions/volumeactions"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v2/snapshots"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v2/volumes"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestVolumesCreateDestroy(t *testing.T) {
	clients.RequireLong(t)

	client, err := clients.NewBlockStorageV2Client()
	th.AssertNoErr(t, err)

	volume, err := CreateVolume(t, client)
	th.AssertNoErr(t, err)
	defer DeleteVolume(t, client, volume)

	newVolume, err := volumes.Get(client, volume.ID).Extract()
	th.AssertNoErr(t, err)

	// Update volume
	updatedVolumeName := ""
	updatedVolumeDescription := ""
	updateOpts := volumes.UpdateOpts{
		Name:        &updatedVolumeName,
		Description: &updatedVolumeDescription,
	}
	updatedVolume, err := volumes.Update(client, volume.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, updatedVolume)
	th.AssertEquals(t, updatedVolume.Name, updatedVolumeName)
	th.AssertEquals(t, updatedVolume.Description, updatedVolumeDescription)

	allPages, err := volumes.List(client, volumes.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)

	allVolumes, err := volumes.ExtractVolumes(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, v := range allVolumes {
		tools.PrintResource(t, volume)
		if v.ID == newVolume.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}

func TestVolumesCreateForceDestroy(t *testing.T) {
	clients.RequireLong(t)

	client, err := clients.NewBlockStorageV2Client()
	th.AssertNoErr(t, err)

	volume, err := CreateVolume(t, client)
	th.AssertNoErr(t, err)

	newVolume, err := volumes.Get(client, volume.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newVolume)

	err = volumeactions.ForceDelete(client, newVolume.ID).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestVolumesCascadeDelete(t *testing.T) {
	clients.RequireLong(t)

	client, err := clients.NewBlockStorageV3Client()
	th.AssertNoErr(t, err)

	vol, err := CreateVolume(t, client)
	th.AssertNoErr(t, err)

	err = volumes.WaitForStatus(client, vol.ID, "available", 60)
	th.AssertNoErr(t, err)

	snapshot1, err := CreateSnapshot(t, client, vol)
	th.AssertNoErr(t, err)

	snapshot2, err := CreateSnapshot(t, client, vol)
	th.AssertNoErr(t, err)

	t.Logf("Attempting to delete volume: %s", vol.ID)

	deleteOpts := volumes.DeleteOpts{Cascade: true}
	err = volumes.Delete(client, vol.ID, deleteOpts).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete volume %s: %v", vol.ID, err)
	}

	for _, sid := range []string{snapshot1.ID, snapshot2.ID} {
		err := gophercloud.WaitFor(120, func() (bool, error) {
			_, err := snapshots.Get(client, sid).Extract()
			if err != nil {
				return true, nil
			}
			return false, nil
		})
		th.AssertNoErr(t, err)
		t.Logf("Successfully deleted snapshot: %s", sid)
	}

	err = gophercloud.WaitFor(120, func() (bool, error) {
		_, err := volumes.Get(client, vol.ID).Extract()
		if err != nil {
			return true, nil
		}
		return false, nil
	})
	th.AssertNoErr(t, err)

	t.Logf("Successfully deleted volume: %s", vol.ID)

}
