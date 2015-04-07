// +build acceptance

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/blockstorage/v1/snapshots"
	"github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestSnapshots(t *testing.T) {

	client, err := newClient(t)
	th.AssertNoErr(t, err)

	v, err := volumes.Create(client, &volumes.CreateOpts{
		Name: "gophercloud-test-volume",
		Size: 1,
	}).Extract()
	th.AssertNoErr(t, err)

	err = volumes.WaitForStatus(client, v.ID, "available", 120)
	th.AssertNoErr(t, err)

	t.Logf("Created volume: %v\n", v)

	ss, err := snapshots.Create(client, &snapshots.CreateOpts{
		Name:     "gophercloud-test-snapshot",
		VolumeID: v.ID,
	}).Extract()
	th.AssertNoErr(t, err)

	err = snapshots.WaitForStatus(client, ss.ID, "available", 120)
	th.AssertNoErr(t, err)

	t.Logf("Created snapshot: %+v\n", ss)

	err = snapshots.Delete(client, ss.ID).ExtractErr()
	th.AssertNoErr(t, err)

	err = gophercloud.WaitFor(120, func() (bool, error) {
		_, err := snapshots.Get(client, ss.ID).Extract()
		if err != nil {
			return true, nil
		}

		return false, nil
	})
	th.AssertNoErr(t, err)

	t.Log("Deleted snapshot\n")

	err = volumes.Delete(client, v.ID).ExtractErr()
	th.AssertNoErr(t, err)

	err = gophercloud.WaitFor(120, func() (bool, error) {
		_, err := volumes.Get(client, v.ID).Extract()
		if err != nil {
			return true, nil
		}

		return false, nil
	})
	th.AssertNoErr(t, err)

	t.Log("Deleted volume\n")
}
