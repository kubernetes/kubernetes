package users

import (
	"testing"

	db "github.com/rackspace/gophercloud/openstack/db/v1/databases"
	os "github.com/rackspace/gophercloud/openstack/db/v1/users"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

const instanceID = "{instanceID}"

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleCreate(t)

	opts := os.BatchCreateOpts{
		os.CreateOpts{
			Databases: db.BatchCreateOpts{
				db.CreateOpts{Name: "databaseA"},
			},
			Name:     "dbuser3",
			Password: "secretsecret",
		},
		os.CreateOpts{
			Databases: db.BatchCreateOpts{
				db.CreateOpts{Name: "databaseB"},
				db.CreateOpts{Name: "databaseC"},
			},
			Name:     "dbuser4",
			Password: "secretsecret",
		},
	}

	res := Create(fake.ServiceClient(), instanceID, opts)
	th.AssertNoErr(t, res.Err)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleDelete(t)

	res := Delete(fake.ServiceClient(), instanceID, "{userName}")
	th.AssertNoErr(t, res.Err)
}
