// +build acceptance db rackspace

package v1

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/rackspace/gophercloud/rackspace/db/v1/backups"
	"github.com/rackspace/gophercloud/rackspace/db/v1/instances"
)

func (c *context) createBackup() {
	opts := backups.CreateOpts{
		Name:       tools.RandomString("backup_", 5),
		InstanceID: c.instanceID,
	}

	backup, err := backups.Create(c.client, opts).Extract()

	c.Logf("Created backup %#v", backup)
	c.AssertNoErr(err)

	err = gophercloud.WaitFor(60, func() (bool, error) {
		b, err := backups.Get(c.client, backup.ID).Extract()
		if err != nil {
			return false, err
		}
		if b.Status == "COMPLETED" {
			return true, nil
		}
		return false, nil
	})
	c.AssertNoErr(err)

	c.backupID = backup.ID
}

func (c *context) getBackup() {
	backup, err := backups.Get(c.client, c.backupID).Extract()
	c.AssertNoErr(err)
	c.Logf("Getting backup %s", backup.ID)
}

func (c *context) listAllBackups() {
	c.Logf("Listing backups")

	err := backups.List(c.client, nil).EachPage(func(page pagination.Page) (bool, error) {
		backupList, err := backups.ExtractBackups(page)
		c.AssertNoErr(err)

		for _, b := range backupList {
			c.Logf("Backup: %#v", b)
		}

		return true, nil
	})

	c.AssertNoErr(err)
}

func (c *context) listInstanceBackups() {
	c.Logf("Listing backups for instance %s", c.instanceID)

	err := instances.ListBackups(c.client, c.instanceID).EachPage(func(page pagination.Page) (bool, error) {
		backupList, err := backups.ExtractBackups(page)
		c.AssertNoErr(err)

		for _, b := range backupList {
			c.Logf("Backup: %#v", b)
		}

		return true, nil
	})

	c.AssertNoErr(err)
}

func (c *context) deleteBackup() {
	err := backups.Delete(c.client, c.backupID).ExtractErr()
	c.AssertNoErr(err)
	c.Logf("Deleted backup %s", c.backupID)
}
