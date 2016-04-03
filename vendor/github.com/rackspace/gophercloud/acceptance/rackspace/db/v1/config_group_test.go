// +build acceptance db rackspace

package v1

import (
	"github.com/rackspace/gophercloud/acceptance/tools"
	os "github.com/rackspace/gophercloud/openstack/db/v1/configurations"
	"github.com/rackspace/gophercloud/pagination"
	config "github.com/rackspace/gophercloud/rackspace/db/v1/configurations"
	"github.com/rackspace/gophercloud/rackspace/db/v1/instances"
)

func (c *context) createConfigGrp() {
	opts := os.CreateOpts{
		Name: tools.RandomString("config_", 5),
		Values: map[string]interface{}{
			"connect_timeout":  300,
			"join_buffer_size": 900000,
		},
	}

	cg, err := config.Create(c.client, opts).Extract()

	c.AssertNoErr(err)
	c.Logf("Created config group %#v", cg)

	c.configGroupID = cg.ID
}

func (c *context) getConfigGrp() {
	cg, err := config.Get(c.client, c.configGroupID).Extract()
	c.Logf("Getting config group: %#v", cg)
	c.AssertNoErr(err)
}

func (c *context) updateConfigGrp() {
	opts := os.UpdateOpts{
		Name: tools.RandomString("new_name_", 5),
		Values: map[string]interface{}{
			"connect_timeout": 250,
		},
	}
	err := config.Update(c.client, c.configGroupID, opts).ExtractErr()
	c.Logf("Updated config group %s", c.configGroupID)
	c.AssertNoErr(err)
}

func (c *context) replaceConfigGrp() {
	opts := os.UpdateOpts{
		Values: map[string]interface{}{
			"big_tables": 1,
		},
	}

	err := config.Replace(c.client, c.configGroupID, opts).ExtractErr()
	c.Logf("Replaced values for config group %s", c.configGroupID)
	c.AssertNoErr(err)
}

func (c *context) associateInstanceWithConfigGrp() {
	err := instances.AssociateWithConfigGroup(c.client, c.instanceID, c.configGroupID).ExtractErr()
	c.Logf("Associated instance %s with config group %s", c.instanceID, c.configGroupID)
	c.AssertNoErr(err)
}

func (c *context) listConfigGrpInstances() {
	c.Logf("Listing all instances associated with config group %s", c.configGroupID)

	err := config.ListInstances(c.client, c.configGroupID).EachPage(func(page pagination.Page) (bool, error) {
		instanceList, err := instances.ExtractInstances(page)
		c.AssertNoErr(err)

		for _, instance := range instanceList {
			c.Logf("Instance: %#v", instance)
		}

		return true, nil
	})

	c.AssertNoErr(err)
}

func (c *context) deleteConfigGrp() {
	err := config.Delete(c.client, c.configGroupID).ExtractErr()
	c.Logf("Deleted config group %s", c.configGroupID)
	c.AssertNoErr(err)
}

func (c *context) detachInstanceFromGrp() {
	err := instances.DetachFromConfigGroup(c.client, c.instanceID).ExtractErr()
	c.Logf("Detached instance %s from config groups", c.instanceID)
	c.AssertNoErr(err)
}
