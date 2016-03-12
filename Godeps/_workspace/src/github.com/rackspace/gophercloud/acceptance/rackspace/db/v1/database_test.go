// +build acceptance db rackspace

package v1

import (
	"github.com/rackspace/gophercloud/acceptance/tools"
	db "github.com/rackspace/gophercloud/openstack/db/v1/databases"
	"github.com/rackspace/gophercloud/pagination"
)

func (c *context) createDBs() {
	dbs := []string{
		tools.RandomString("db_", 5),
		tools.RandomString("db_", 5),
		tools.RandomString("db_", 5),
	}

	opts := db.BatchCreateOpts{
		db.CreateOpts{Name: dbs[0]},
		db.CreateOpts{Name: dbs[1]},
		db.CreateOpts{Name: dbs[2]},
	}

	err := db.Create(c.client, c.instanceID, opts).ExtractErr()
	c.Logf("Created three databases on instance %s: %s, %s, %s", c.instanceID, dbs[0], dbs[1], dbs[2])
	c.AssertNoErr(err)

	c.DBIDs = dbs
}

func (c *context) listDBs() {
	c.Logf("Listing databases on instance %s", c.instanceID)

	err := db.List(c.client, c.instanceID).EachPage(func(page pagination.Page) (bool, error) {
		dbList, err := db.ExtractDBs(page)
		c.AssertNoErr(err)

		for _, db := range dbList {
			c.Logf("DB: %#v", db)
		}

		return true, nil
	})

	c.AssertNoErr(err)
}

func (c *context) deleteDBs() {
	for _, id := range c.DBIDs {
		err := db.Delete(c.client, c.instanceID, id).ExtractErr()
		c.AssertNoErr(err)
		c.Logf("Deleted DB %s", id)
	}
}
