// +build acceptance db rackspace

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/db/v1/instances"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestRunner(t *testing.T) {
	c := newContext(t)

	// FLAVOR tests
	c.listFlavors()
	c.getFlavor()

	// INSTANCE tests
	c.createInstance()
	c.listInstances()
	c.getInstance()
	c.isRootEnabled()
	c.enableRootUser()
	c.isRootEnabled()
	c.restartInstance()
	c.resizeInstance()
	c.resizeVol()
	c.getDefaultConfig()

	// REPLICA tests
	c.createReplica()
	c.detachReplica()

	// BACKUP tests
	c.createBackup()
	c.getBackup()
	c.listAllBackups()
	c.listInstanceBackups()
	c.deleteBackup()

	// CONFIG GROUP tests
	c.createConfigGrp()
	c.getConfigGrp()
	c.updateConfigGrp()
	c.replaceConfigGrp()
	c.associateInstanceWithConfigGrp()
	c.listConfigGrpInstances()
	c.detachInstanceFromGrp()
	c.deleteConfigGrp()

	// DATABASE tests
	c.createDBs()
	c.listDBs()

	// USER tests
	c.createUsers()
	c.listUsers()
	c.changeUserPwd()
	c.getUser()
	c.updateUser()
	c.listUserAccess()
	c.revokeUserAccess()
	c.grantUserAccess()

	// TEARDOWN
	c.deleteUsers()
	c.deleteDBs()

	c.restartInstance()
	c.WaitUntilActive(c.instanceID)

	c.deleteInstance(c.replicaID)
	c.deleteInstance(c.instanceID)
}

func (c *context) createInstance() {
	opts := instances.CreateOpts{
		FlavorRef: "1",
		Size:      1,
		Name:      tools.RandomString("gopher_db", 5),
	}

	instance, err := instances.Create(c.client, opts).Extract()
	th.AssertNoErr(c.test, err)

	c.Logf("Creating %s. Waiting...", instance.ID)
	c.WaitUntilActive(instance.ID)
	c.Logf("Created instance %s", instance.ID)

	c.instanceID = instance.ID
}

func (c *context) listInstances() {
	c.Logf("Listing instances")

	err := instances.List(c.client, nil).EachPage(func(page pagination.Page) (bool, error) {
		instanceList, err := instances.ExtractInstances(page)
		c.AssertNoErr(err)

		for _, i := range instanceList {
			c.Logf("Instance: ID [%s] Name [%s] Status [%s] VolSize [%d] Datastore Type [%s]",
				i.ID, i.Name, i.Status, i.Volume.Size, i.Datastore.Type)
		}

		return true, nil
	})

	c.AssertNoErr(err)
}

func (c *context) getInstance() {
	instance, err := instances.Get(c.client, c.instanceID).Extract()
	c.AssertNoErr(err)
	c.Logf("Getting instance: %#v", instance)
}

func (c *context) deleteInstance(id string) {
	err := instances.Delete(c.client, id).ExtractErr()
	c.AssertNoErr(err)
	c.Logf("Deleted instance %s", id)
}

func (c *context) enableRootUser() {
	_, err := instances.EnableRootUser(c.client, c.instanceID).Extract()
	c.AssertNoErr(err)
	c.Logf("Enabled root user on %s", c.instanceID)
}

func (c *context) isRootEnabled() {
	enabled, err := instances.IsRootEnabled(c.client, c.instanceID)
	c.AssertNoErr(err)
	c.Logf("Is root enabled? %s", enabled)
}

func (c *context) restartInstance() {
	id := c.instanceID
	err := instances.Restart(c.client, id).ExtractErr()
	c.AssertNoErr(err)
	c.Logf("Restarting %s. Waiting...", id)
	c.WaitUntilActive(id)
	c.Logf("Restarted %s", id)
}

func (c *context) resizeInstance() {
	id := c.instanceID
	err := instances.Resize(c.client, id, "2").ExtractErr()
	c.AssertNoErr(err)
	c.Logf("Resizing %s. Waiting...", id)
	c.WaitUntilActive(id)
	c.Logf("Resized %s with flavorRef %s", id, "2")
}

func (c *context) resizeVol() {
	id := c.instanceID
	err := instances.ResizeVolume(c.client, id, 2).ExtractErr()
	c.AssertNoErr(err)
	c.Logf("Resizing volume of %s. Waiting...", id)
	c.WaitUntilActive(id)
	c.Logf("Resized the volume of %s to %d GB", id, 2)
}

func (c *context) getDefaultConfig() {
	config, err := instances.GetDefaultConfig(c.client, c.instanceID).Extract()
	c.Logf("Default config group for instance %s: %#v", c.instanceID, config)
	c.AssertNoErr(err)
}
