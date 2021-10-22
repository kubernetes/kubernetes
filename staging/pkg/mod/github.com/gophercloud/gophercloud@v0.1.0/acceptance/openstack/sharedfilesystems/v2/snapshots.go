package v2

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/snapshots"
)

// CreateSnapshot will create a snapshot from the share ID with a name. An error will
// be returned if the snapshot could not be created
func CreateSnapshot(t *testing.T, client *gophercloud.ServiceClient, shareID string) (*snapshots.Snapshot, error) {
	if testing.Short() {
		t.Skip("Skipping test that requres share creation in short mode.")
	}

	createOpts := snapshots.CreateOpts{
		ShareID:     shareID,
		Name:        "My Test Snapshot",
		Description: "My Test Description",
	}

	snapshot, err := snapshots.Create(client, createOpts).Extract()
	if err != nil {
		t.Logf("Failed to create snapshot")
		return nil, err
	}

	err = waitForSnapshotStatus(t, client, snapshot.ID, "available", 600)
	if err != nil {
		t.Logf("Failed to get %s snapshot status", snapshot.ID)
		return snapshot, err
	}

	return snapshot, nil
}

// ListSnapshots lists all snapshots that belong to this tenant's project.
// An error will be returned if the snapshots could not be listed..
func ListSnapshots(t *testing.T, client *gophercloud.ServiceClient) ([]snapshots.Snapshot, error) {
	r, err := snapshots.ListDetail(client, &snapshots.ListOpts{}).AllPages()
	if err != nil {
		return nil, err
	}

	return snapshots.ExtractSnapshots(r)
}

// DeleteSnapshot will delete a snapshot. A fatal error will occur if the snapshot
// failed to be deleted. This works best when used as a deferred function.
func DeleteSnapshot(t *testing.T, client *gophercloud.ServiceClient, snapshot *snapshots.Snapshot) {
	err := snapshots.Delete(client, snapshot.ID).ExtractErr()
	if err != nil {
		t.Errorf("Unable to delete snapshot %s: %v", snapshot.ID, err)
	}

	err = waitForSnapshotStatus(t, client, snapshot.ID, "deleted", 600)
	if err != nil {
		t.Errorf("Failed to wait for 'deleted' status for %s snapshot: %v", snapshot.ID, err)
	} else {
		t.Logf("Deleted snapshot: %s", snapshot.ID)
	}
}

func waitForSnapshotStatus(t *testing.T, c *gophercloud.ServiceClient, id, status string, secs int) error {
	err := gophercloud.WaitFor(secs, func() (bool, error) {
		current, err := snapshots.Get(c, id).Extract()
		if err != nil {
			if _, ok := err.(gophercloud.ErrDefault404); ok {
				switch status {
				case "deleted":
					return true, nil
				default:
					return false, err
				}
			}
			return false, err
		}

		if current.Status == status {
			return true, nil
		}

		if strings.Contains(current.Status, "error") {
			return true, fmt.Errorf("An error occurred, wrong status: %s", current.Status)
		}

		return false, nil
	})

	if err != nil {
		mErr := PrintMessages(t, c, id)
		if mErr != nil {
			return fmt.Errorf("Snapshot status is '%s' and unable to get manila messages: %s", err, mErr)
		}
	}

	return err
}
