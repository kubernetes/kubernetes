package testing

import (
	"testing"

	db "github.com/gophercloud/gophercloud/openstack/db/v1/databases"
	"github.com/gophercloud/gophercloud/openstack/db/v1/instances"
	"github.com/gophercloud/gophercloud/openstack/db/v1/users"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreate(t)

	opts := instances.CreateOpts{
		Name:      "json_rack_instance",
		FlavorRef: "1",
		Databases: db.BatchCreateOpts{
			{CharSet: "utf8", Collate: "utf8_general_ci", Name: "sampledb"},
			{Name: "nextround"},
		},
		Users: users.BatchCreateOpts{
			{
				Name:     "demouser",
				Password: "demopassword",
				Databases: db.BatchCreateOpts{
					{Name: "sampledb"},
				},
			},
		},
		Size: 2,
	}

	instance, err := instances.Create(fake.ServiceClient(), opts).Extract()

	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, &expectedInstance, instance)
}

func TestInstanceList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleList(t)

	pages := 0
	err := instances.List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := instances.ExtractInstances(page)
		if err != nil {
			return false, err
		}

		th.CheckDeepEquals(t, []instances.Instance{expectedInstance}, actual)
		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
}

func TestGetInstance(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGet(t)

	instance, err := instances.Get(fake.ServiceClient(), instanceID).Extract()

	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, &expectedInstance, instance)
}

func TestDeleteInstance(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDelete(t)

	res := instances.Delete(fake.ServiceClient(), instanceID)
	th.AssertNoErr(t, res.Err)
}

func TestEnableRootUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleEnableRoot(t)

	expected := &users.User{Name: "root", Password: "secretsecret"}
	user, err := instances.EnableRootUser(fake.ServiceClient(), instanceID).Extract()

	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, user)
}

func TestIsRootEnabled(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleIsRootEnabled(t)

	isEnabled, err := instances.IsRootEnabled(fake.ServiceClient(), instanceID).Extract()

	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, isEnabled)
}

func TestRestart(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRestart(t)

	res := instances.Restart(fake.ServiceClient(), instanceID)
	th.AssertNoErr(t, res.Err)
}

func TestResize(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleResize(t)

	res := instances.Resize(fake.ServiceClient(), instanceID, "2")
	th.AssertNoErr(t, res.Err)
}

func TestResizeVolume(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleResizeVol(t)

	res := instances.ResizeVolume(fake.ServiceClient(), instanceID, 4)
	th.AssertNoErr(t, res.Err)
}
