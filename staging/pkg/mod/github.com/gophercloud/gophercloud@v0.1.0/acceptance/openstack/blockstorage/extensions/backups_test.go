// +build acceptance blockstorage

package extensions

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/extensions/backups"

	blockstorage "github.com/gophercloud/gophercloud/acceptance/openstack/blockstorage/v3"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestBackupsCRUD(t *testing.T) {
	blockClient, err := clients.NewBlockStorageV3Client()
	th.AssertNoErr(t, err)

	volume, err := blockstorage.CreateVolume(t, blockClient)
	th.AssertNoErr(t, err)
	defer blockstorage.DeleteVolume(t, blockClient, volume)

	backup, err := CreateBackup(t, blockClient, volume.ID)
	th.AssertNoErr(t, err)
	defer DeleteBackup(t, blockClient, backup.ID)

	allPages, err := backups.List(blockClient, nil).AllPages()
	th.AssertNoErr(t, err)

	allBackups, err := backups.ExtractBackups(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, v := range allBackups {
		if backup.Name == v.Name {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}
