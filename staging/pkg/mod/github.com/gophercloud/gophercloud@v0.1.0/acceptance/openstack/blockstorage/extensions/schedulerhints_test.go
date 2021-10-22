// +build acceptance blockstorage

package extensions

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/extensions/schedulerhints"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v3/volumes"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestSchedulerHints(t *testing.T) {
	clients.RequireLong(t)

	client, err := clients.NewBlockStorageV3Client()
	th.AssertNoErr(t, err)

	volumeName := tools.RandomString("ACPTTEST", 16)
	createOpts := volumes.CreateOpts{
		Size: 1,
		Name: volumeName,
	}

	volume1, err := volumes.Create(client, createOpts).Extract()
	th.AssertNoErr(t, err)

	err = volumes.WaitForStatus(client, volume1.ID, "available", 60)
	th.AssertNoErr(t, err)
	defer volumes.Delete(client, volume1.ID, volumes.DeleteOpts{})

	volumeName = tools.RandomString("ACPTTEST", 16)
	base := volumes.CreateOpts{
		Size: 1,
		Name: volumeName,
	}

	schedulerHints := schedulerhints.SchedulerHints{
		SameHost: []string{
			volume1.ID,
		},
	}

	createOptsWithHints := schedulerhints.CreateOptsExt{
		VolumeCreateOptsBuilder: base,
		SchedulerHints:          schedulerHints,
	}

	volume2, err := volumes.Create(client, createOptsWithHints).Extract()
	th.AssertNoErr(t, err)

	err = volumes.WaitForStatus(client, volume2.ID, "available", 60)
	th.AssertNoErr(t, err)

	err = volumes.Delete(client, volume2.ID, volumes.DeleteOpts{}).ExtractErr()
	th.AssertNoErr(t, err)
}
