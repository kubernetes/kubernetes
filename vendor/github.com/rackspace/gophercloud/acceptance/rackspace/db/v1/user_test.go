// +build acceptance db rackspace

package v1

import (
	"github.com/rackspace/gophercloud/acceptance/tools"
	db "github.com/rackspace/gophercloud/openstack/db/v1/databases"
	os "github.com/rackspace/gophercloud/openstack/db/v1/users"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/db/v1/users"
)

func (c *context) createUsers() {
	c.users = []string{
		tools.RandomString("user_", 5),
		tools.RandomString("user_", 5),
		tools.RandomString("user_", 5),
	}

	db1 := db.CreateOpts{Name: c.DBIDs[0]}
	db2 := db.CreateOpts{Name: c.DBIDs[1]}
	db3 := db.CreateOpts{Name: c.DBIDs[2]}

	opts := os.BatchCreateOpts{
		os.CreateOpts{
			Name:      c.users[0],
			Password:  tools.RandomString("db_", 5),
			Databases: db.BatchCreateOpts{db1, db2, db3},
		},
		os.CreateOpts{
			Name:      c.users[1],
			Password:  tools.RandomString("db_", 5),
			Databases: db.BatchCreateOpts{db1, db2},
		},
		os.CreateOpts{
			Name:      c.users[2],
			Password:  tools.RandomString("db_", 5),
			Databases: db.BatchCreateOpts{db3},
		},
	}

	err := users.Create(c.client, c.instanceID, opts).ExtractErr()
	c.Logf("Created three users on instance %s: %s, %s, %s", c.instanceID, c.users[0], c.users[1], c.users[2])
	c.AssertNoErr(err)
}

func (c *context) listUsers() {
	c.Logf("Listing users on instance %s", c.instanceID)

	err := os.List(c.client, c.instanceID).EachPage(func(page pagination.Page) (bool, error) {
		uList, err := os.ExtractUsers(page)
		c.AssertNoErr(err)

		for _, u := range uList {
			c.Logf("User: %#v", u)
		}

		return true, nil
	})

	c.AssertNoErr(err)
}

func (c *context) deleteUsers() {
	for _, id := range c.users {
		err := users.Delete(c.client, c.instanceID, id).ExtractErr()
		c.AssertNoErr(err)
		c.Logf("Deleted user %s", id)
	}
}

func (c *context) changeUserPwd() {
	opts := os.BatchCreateOpts{}

	for _, name := range c.users[:1] {
		opts = append(opts, os.CreateOpts{Name: name, Password: tools.RandomString("", 5)})
	}

	err := users.ChangePassword(c.client, c.instanceID, opts).ExtractErr()
	c.Logf("Updated 2 users' passwords")
	c.AssertNoErr(err)
}

func (c *context) getUser() {
	user, err := users.Get(c.client, c.instanceID, c.users[0]).Extract()
	c.Logf("Getting user %s", user)
	c.AssertNoErr(err)
}

func (c *context) updateUser() {
	opts := users.UpdateOpts{Name: tools.RandomString("new_name_", 5)}
	err := users.Update(c.client, c.instanceID, c.users[0], opts).ExtractErr()
	c.Logf("Updated user %s", c.users[0])
	c.AssertNoErr(err)
	c.users[0] = opts.Name
}

func (c *context) listUserAccess() {
	err := users.ListAccess(c.client, c.instanceID, c.users[0]).EachPage(func(page pagination.Page) (bool, error) {
		dbList, err := users.ExtractDBs(page)
		c.AssertNoErr(err)

		for _, db := range dbList {
			c.Logf("User %s has access to DB: %#v", db)
		}

		return true, nil
	})

	c.AssertNoErr(err)
}

func (c *context) grantUserAccess() {
	opts := db.BatchCreateOpts{db.CreateOpts{Name: c.DBIDs[0]}}
	err := users.GrantAccess(c.client, c.instanceID, c.users[0], opts).ExtractErr()
	c.Logf("Granted access for user %s to DB %s", c.users[0], c.DBIDs[0])
	c.AssertNoErr(err)
}

func (c *context) revokeUserAccess() {
	dbName, userName := c.DBIDs[0], c.users[0]
	err := users.RevokeAccess(c.client, c.instanceID, userName, dbName).ExtractErr()
	c.Logf("Revoked access for user %s to DB %s", userName, dbName)
	c.AssertNoErr(err)
}
