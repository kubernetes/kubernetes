// +build acceptance db

package v1

import (
	"os"
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/db/v1/instances"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

const envDSType = "DATASTORE_TYPE_ID"

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
	//c.resizeInstance()
	//c.resizeVol()

	// DATABASE tests
	c.createDBs()
	c.listDBs()

	// USER tests
	c.createUsers()
	c.listUsers()

	// TEARDOWN
	c.deleteUsers()
	c.deleteDBs()
	c.deleteInstance()
}

func (c context) createInstance() {
	if os.Getenv(envDSType) == "" {
		c.test.Fatalf("%s must be set as an environment var", envDSType)
	}

	opts := instances.CreateOpts{
		FlavorRef: "2",
		Size:      5,
		Name:      tools.RandomString("gopher_db", 5),
		Datastore: &instances.DatastoreOpts{Type: os.Getenv(envDSType)},
	}

	instance, err := instances.Create(c.client, opts).Extract()
	th.AssertNoErr(c.test, err)

	c.Logf("Restarting %s. Waiting...", instance.ID)
	c.WaitUntilActive(instance.ID)
	c.Logf("Created Instance %s", instance.ID)

	c.instanceID = instance.ID
}

func (c context) listInstances() {
	c.Logf("Listing instances")

	err := instances.List(c.client).EachPage(func(page pagination.Page) (bool, error) {
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

func (c context) getInstance() {
	instance, err := instances.Get(c.client, c.instanceID).Extract()
	c.AssertNoErr(err)
	c.Logf("Getting instance: %s", instance.ID)
}

func (c context) deleteInstance() {
	err := instances.Delete(c.client, c.instanceID).ExtractErr()
	c.AssertNoErr(err)
	c.Logf("Deleted instance %s", c.instanceID)
}

func (c context) enableRootUser() {
	_, err := instances.EnableRootUser(c.client, c.instanceID).Extract()
	c.AssertNoErr(err)
	c.Logf("Enabled root user on %s", c.instanceID)
}

func (c context) isRootEnabled() {
	enabled, err := instances.IsRootEnabled(c.client, c.instanceID)
	c.AssertNoErr(err)
	c.Logf("Is root enabled? %d", enabled)
}

func (c context) restartInstance() {
	id := c.instanceID
	err := instances.Restart(c.client, id).ExtractErr()
	c.AssertNoErr(err)
	c.Logf("Restarting %s. Waiting...", id)
	c.WaitUntilActive(id)
	c.Logf("Restarted %s", id)
}

func (c context) resizeInstance() {
	id := c.instanceID
	err := instances.Resize(c.client, id, "3").ExtractErr()
	c.AssertNoErr(err)
	c.Logf("Resizing %s. Waiting...", id)
	c.WaitUntilActive(id)
	c.Logf("Resized %s with flavorRef %s", id, "2")
}

func (c context) resizeVol() {
	id := c.instanceID
	err := instances.ResizeVolume(c.client, id, 4).ExtractErr()
	c.AssertNoErr(err)
	c.Logf("Resizing volume of %s. Waiting...", id)
	c.WaitUntilActive(id)
	c.Logf("Resized the volume of %s to %d GB", id, 2)
}
