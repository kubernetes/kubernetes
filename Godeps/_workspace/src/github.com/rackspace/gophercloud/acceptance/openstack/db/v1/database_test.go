// +build acceptance db

package v1

import (
	db "github.com/rackspace/gophercloud/openstack/db/v1/databases"
	"github.com/rackspace/gophercloud/pagination"
)

func (c context) createDBs() {
	opts := db.BatchCreateOpts{
		db.CreateOpts{Name: "db1"},
		db.CreateOpts{Name: "db2"},
		db.CreateOpts{Name: "db3"},
	}

	err := db.Create(c.client, c.instanceID, opts).ExtractErr()
	c.AssertNoErr(err)
	c.Logf("Created three databases on instance %s: db1, db2, db3", c.instanceID)
}

func (c context) listDBs() {
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

func (c context) deleteDBs() {
	for _, id := range []string{"db1", "db2", "db3"} {
		err := db.Delete(c.client, c.instanceID, id).ExtractErr()
		c.AssertNoErr(err)
		c.Logf("Deleted DB %s", id)
	}
}
