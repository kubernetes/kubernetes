package users

import (
	"testing"

	db "github.com/rackspace/gophercloud/openstack/db/v1/databases"
	os "github.com/rackspace/gophercloud/openstack/db/v1/users"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
	"github.com/rackspace/gophercloud/testhelper/fixture"
)

var (
	userName = "{userName}"
	_rootURL = "/instances/" + instanceID + "/users"
	_userURL = _rootURL + "/" + userName
	_dbURL   = _userURL + "/databases"
)

func TestChangeUserPassword(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, _rootURL, "PUT", changePwdReq, "", 202)

	opts := os.BatchCreateOpts{
		os.CreateOpts{Name: "dbuser1", Password: "newpassword"},
		os.CreateOpts{Name: "dbuser2", Password: "anotherpassword"},
	}

	err := ChangePassword(fake.ServiceClient(), instanceID, opts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestUpdateUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, _userURL, "PUT", updateReq, "", 202)

	opts := UpdateOpts{
		Name:     "new_username",
		Password: "new_password",
	}

	err := Update(fake.ServiceClient(), instanceID, userName, opts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestGetUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, _userURL, "GET", "", getResp, 200)

	user, err := Get(fake.ServiceClient(), instanceID, userName).Extract()

	th.AssertNoErr(t, err)

	expected := &User{
		Name: "exampleuser",
		Host: "foo",
		Databases: []db.Database{
			db.Database{Name: "databaseA"},
			db.Database{Name: "databaseB"},
		},
	}

	th.AssertDeepEquals(t, expected, user)
}

func TestUserAccessList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, _userURL+"/databases", "GET", "", listUserAccessResp, 200)

	expectedDBs := []db.Database{
		db.Database{Name: "databaseE"},
	}

	pages := 0
	err := ListAccess(fake.ServiceClient(), instanceID, userName).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := ExtractDBs(page)
		if err != nil {
			return false, err
		}

		th.CheckDeepEquals(t, expectedDBs, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
}

func TestUserList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	fixture.SetupHandler(t, "/instances/"+instanceID+"/users", "GET", "", listResp, 200)

	expectedUsers := []User{
		User{
			Databases: []db.Database{
				db.Database{Name: "databaseA"},
			},
			Name: "dbuser1",
			Host: "localhost",
		},
		User{
			Databases: []db.Database{
				db.Database{Name: "databaseB"},
				db.Database{Name: "databaseC"},
			},
			Name: "dbuser2",
			Host: "localhost",
		},
	}

	pages := 0
	err := List(fake.ServiceClient(), instanceID).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := ExtractUsers(page)
		if err != nil {
			return false, err
		}

		th.CheckDeepEquals(t, expectedUsers, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
}

func TestGrantAccess(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, _dbURL, "PUT", grantUserAccessReq, "", 202)

	opts := db.BatchCreateOpts{db.CreateOpts{Name: "databaseE"}}
	err := GrantAccess(fake.ServiceClient(), instanceID, userName, opts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestRevokeAccess(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, _dbURL+"/{dbName}", "DELETE", "", "", 202)

	err := RevokeAccess(fake.ServiceClient(), instanceID, userName, "{dbName}").ExtractErr()
	th.AssertNoErr(t, err)
}
