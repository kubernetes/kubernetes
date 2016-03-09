package instances

import (
	"testing"

	db "github.com/rackspace/gophercloud/openstack/db/v1/databases"
	"github.com/rackspace/gophercloud/openstack/db/v1/users"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreate(t)

	opts := CreateOpts{
		Name:      "json_rack_instance",
		FlavorRef: "1",
		Databases: db.BatchCreateOpts{
			db.CreateOpts{CharSet: "utf8", Collate: "utf8_general_ci", Name: "sampledb"},
			db.CreateOpts{Name: "nextround"},
		},
		Users: users.BatchCreateOpts{
			users.CreateOpts{
				Name:     "demouser",
				Password: "demopassword",
				Databases: db.BatchCreateOpts{
					db.CreateOpts{Name: "sampledb"},
				},
			},
		},
		Size: 2,
	}

	instance, err := Create(fake.ServiceClient(), opts).Extract()

	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, &expectedInstance, instance)
}

func TestInstanceList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleList(t)

	pages := 0
	err := List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := ExtractInstances(page)
		if err != nil {
			return false, err
		}

		th.CheckDeepEquals(t, []Instance{expectedInstance}, actual)
		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
}

func TestGetInstance(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGet(t)

	instance, err := Get(fake.ServiceClient(), instanceID).Extract()

	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, &expectedInstance, instance)
}

func TestDeleteInstance(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDelete(t)

	res := Delete(fake.ServiceClient(), instanceID)
	th.AssertNoErr(t, res.Err)
}

func TestEnableRootUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleEnableRoot(t)

	expected := &users.User{Name: "root", Password: "secretsecret"}
	user, err := EnableRootUser(fake.ServiceClient(), instanceID).Extract()

	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, user)
}

func TestIsRootEnabled(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleIsRootEnabled(t)

	isEnabled, err := IsRootEnabled(fake.ServiceClient(), instanceID)

	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, isEnabled)
}

func TestRestart(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRestart(t)

	res := Restart(fake.ServiceClient(), instanceID)
	th.AssertNoErr(t, res.Err)
}

func TestResize(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleResize(t)

	res := Resize(fake.ServiceClient(), instanceID, "2")
	th.AssertNoErr(t, res.Err)
}

func TestResizeVolume(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleResizeVol(t)

	res := ResizeVolume(fake.ServiceClient(), instanceID, 4)
	th.AssertNoErr(t, res.Err)
}
