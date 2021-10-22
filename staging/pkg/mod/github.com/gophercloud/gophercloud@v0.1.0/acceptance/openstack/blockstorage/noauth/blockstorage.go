// Package noauth contains common functions for creating block storage based
// resources for use in acceptance tests. See the `*_test.go` files for
// example usages.
package noauth

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v2/snapshots"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v2/volumes"
)

// CreateVolume will create a volume with a random name and size of 1GB. An
// error will be returned if the volume was unable to be created.
func CreateVolume(t *testing.T, client *gophercloud.ServiceClient) (*volumes.Volume, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires volume creation in short mode.")
	}

	volumeName := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create volume: %s", volumeName)

	createOpts := volumes.CreateOpts{
		Size: 1,
		Name: volumeName,
	}

	volume, err := volumes.Create(client, createOpts).Extract()
	if err != nil {
		return volume, err
	}

	err = volumes.WaitForStatus(client, volume.ID, "available", 60)
	if err != nil {
		return volume, err
	}

	return volume, nil
}

// CreateVolumeFromImage will create a volume from with a random name and size of
// 1GB. An error will be returned if the volume was unable to be created.
func CreateVolumeFromImage(t *testing.T, client *gophercloud.ServiceClient) (*volumes.Volume, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires volume creation in short mode.")
	}

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	volumeName := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create volume: %s", volumeName)

	createOpts := volumes.CreateOpts{
		Size:    1,
		Name:    volumeName,
		ImageID: choices.ImageID,
	}

	volume, err := volumes.Create(client, createOpts).Extract()
	if err != nil {
		return volume, err
	}

	err = volumes.WaitForStatus(client, volume.ID, "available", 60)
	if err != nil {
		return volume, err
	}

	return volume, nil
}

// DeleteVolume will delete a volume. A fatal error will occur if the volume
// failed to be deleted. This works best when used as a deferred function.
func DeleteVolume(t *testing.T, client *gophercloud.ServiceClient, volume *volumes.Volume) {
	err := volumes.Delete(client, volume.ID, volumes.DeleteOpts{}).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete volume %s: %v", volume.ID, err)
	}

	t.Logf("Deleted volume: %s", volume.ID)
}

// CreateSnapshot will create a snapshot of the specified volume.
// Snapshot will be assigned a random name and description.
func CreateSnapshot(t *testing.T, client *gophercloud.ServiceClient, volume *volumes.Volume) (*snapshots.Snapshot, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires snapshot creation in short mode.")
	}

	snapshotName := tools.RandomString("ACPTTEST", 16)
	snapshotDescription := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create snapshot: %s", snapshotName)

	createOpts := snapshots.CreateOpts{
		VolumeID:    volume.ID,
		Name:        snapshotName,
		Description: snapshotDescription,
	}

	snapshot, err := snapshots.Create(client, createOpts).Extract()
	if err != nil {
		return snapshot, err
	}

	err = snapshots.WaitForStatus(client, snapshot.ID, "available", 60)
	if err != nil {
		return snapshot, err
	}

	return snapshot, nil
}

// DeleteSnapshot will delete a snapshot. A fatal error will occur if the
// snapshot failed to be deleted.
func DeleteSnapshot(t *testing.T, client *gophercloud.ServiceClient, snapshot *snapshots.Snapshot) {
	err := snapshots.Delete(client, snapshot.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete snapshot %s: %+v", snapshot.ID, err)
	}

	// Volumes can't be deleted until their snapshots have been,
	// so block up to 120 seconds for the snapshot to delete.
	err = gophercloud.WaitFor(120, func() (bool, error) {
		_, err := snapshots.Get(client, snapshot.ID).Extract()
		if err != nil {
			return true, nil
		}

		return false, nil
	})
	if err != nil {
		t.Fatalf("Error waiting for snapshot to delete: %v", err)
	}

	t.Logf("Deleted snapshot: %s", snapshot.ID)
}
