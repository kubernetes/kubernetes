// Package v1 contains common functions for creating block storage based
// resources for use in acceptance tests. See the `*_test.go` files for
// example usages.
package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v1/snapshots"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v1/volumes"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v1/volumetypes"
)

// CreateSnapshot will create a volume snapshot based off of a given volume and
// with a random name. An error will be returned if the snapshot failed to be
// created.
func CreateSnapshot(t *testing.T, client *gophercloud.ServiceClient, volume *volumes.Volume) (*snapshots.Snapshot, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires snapshot creation in short mode.")
	}

	snapshotName := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create snapshot %s based on volume %s", snapshotName, volume.ID)

	createOpts := snapshots.CreateOpts{
		Name:     snapshotName,
		VolumeID: volume.ID,
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

// CreateVolume will create a volume with a random name and size of 1GB. An
// error will be returned if the volume was unable to be created.
func CreateVolume(t *testing.T, client *gophercloud.ServiceClient) (*volumes.Volume, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires volume creation in short mode.")
	}

	volumeName := tools.RandomString("ACPTTEST", 16)
	volumeDescription := tools.RandomString("ACPTTEST-DESC", 16)
	t.Logf("Attempting to create volume: %s", volumeName)

	createOpts := volumes.CreateOpts{
		Size:        1,
		Name:        volumeName,
		Description: volumeDescription,
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

// CreateVolumeType will create a volume type with a random name. An error will
// be returned if the volume type was unable to be created.
func CreateVolumeType(t *testing.T, client *gophercloud.ServiceClient) (*volumetypes.VolumeType, error) {
	volumeTypeName := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create volume type: %s", volumeTypeName)

	createOpts := volumetypes.CreateOpts{
		Name: volumeTypeName,
		ExtraSpecs: map[string]interface{}{
			"capabilities": "ssd",
			"priority":     3,
		},
	}

	volumeType, err := volumetypes.Create(client, createOpts).Extract()
	if err != nil {
		return volumeType, err
	}

	return volumeType, nil
}

// DeleteSnapshot will delete a snapshot. A fatal error will occur if the
// snapshot failed to be deleted. This works best when used as a deferred
// function.
func DeleteSnapshotshot(t *testing.T, client *gophercloud.ServiceClient, snapshot *snapshots.Snapshot) {
	err := snapshots.Delete(client, snapshot.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete snapshot %s: %v", snapshot.ID, err)
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
		t.Fatalf("Unable to wait for snapshot to delete: %v", err)
	}

	t.Logf("Deleted snapshot: %s", snapshot.ID)
}

// DeleteVolume will delete a volume. A fatal error will occur if the volume
// failed to be deleted. This works best when used as a deferred function.
func DeleteVolume(t *testing.T, client *gophercloud.ServiceClient, volume *volumes.Volume) {
	err := volumes.Delete(client, volume.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete volume %s: %v", volume.ID, err)
	}

	t.Logf("Deleted volume: %s", volume.ID)
}

// DeleteVolumeType will delete a volume type. A fatal error will occur if the
// volume type failed to be deleted. This works best when used as a deferred
// function.
func DeleteVolumeType(t *testing.T, client *gophercloud.ServiceClient, volumeType *volumetypes.VolumeType) {
	err := volumetypes.Delete(client, volumeType.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete volume type %s: %v", volumeType.ID, err)
	}

	t.Logf("Deleted volume type: %s", volumeType.ID)
}
