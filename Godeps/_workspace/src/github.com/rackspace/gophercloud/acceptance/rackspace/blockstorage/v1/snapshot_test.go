// +build acceptance blockstorage snapshots

package v1

import (
	"testing"
	"time"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/blockstorage/v1/snapshots"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestSnapshots(t *testing.T) {
	client := setup(t)
	volID := testVolumeCreate(t, client)

	t.Log("Creating snapshots")
	s := testSnapshotCreate(t, client, volID)
	id := s.ID

	t.Log("Listing snapshots")
	testSnapshotList(t, client)

	t.Logf("Getting snapshot %s", id)
	testSnapshotGet(t, client, id)

	t.Logf("Updating snapshot %s", id)
	testSnapshotUpdate(t, client, id)

	t.Logf("Deleting snapshot %s", id)
	testSnapshotDelete(t, client, id)
	s.WaitUntilDeleted(client, -1)

	t.Logf("Deleting volume %s", volID)
	testVolumeDelete(t, client, volID)
}

func testSnapshotCreate(t *testing.T, client *gophercloud.ServiceClient, volID string) *snapshots.Snapshot {
	opts := snapshots.CreateOpts{VolumeID: volID, Name: "snapshot-001"}
	s, err := snapshots.Create(client, opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created snapshot %s", s.ID)

	t.Logf("Waiting for new snapshot to become available...")
	start := time.Now().Second()
	s.WaitUntilComplete(client, -1)
	t.Logf("Snapshot completed after %ds", time.Now().Second()-start)

	return s
}

func testSnapshotList(t *testing.T, client *gophercloud.ServiceClient) {
	snapshots.List(client).EachPage(func(page pagination.Page) (bool, error) {
		sList, err := snapshots.ExtractSnapshots(page)
		th.AssertNoErr(t, err)

		for _, s := range sList {
			t.Logf("Snapshot: ID [%s] Name [%s] Volume ID [%s] Progress [%s] Created [%s]",
				s.ID, s.Name, s.VolumeID, s.Progress, s.CreatedAt)
		}

		return true, nil
	})
}

func testSnapshotGet(t *testing.T, client *gophercloud.ServiceClient, id string) {
	_, err := snapshots.Get(client, id).Extract()
	th.AssertNoErr(t, err)
}

func testSnapshotUpdate(t *testing.T, client *gophercloud.ServiceClient, id string) {
	_, err := snapshots.Update(client, id, snapshots.UpdateOpts{Name: "new_name"}).Extract()
	th.AssertNoErr(t, err)
}

func testSnapshotDelete(t *testing.T, client *gophercloud.ServiceClient, id string) {
	res := snapshots.Delete(client, id)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted snapshot %s", id)
}
