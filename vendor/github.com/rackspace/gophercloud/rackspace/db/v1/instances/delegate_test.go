package instances

import (
	"testing"

	osDBs "github.com/rackspace/gophercloud/openstack/db/v1/databases"
	os "github.com/rackspace/gophercloud/openstack/db/v1/instances"
	osUsers "github.com/rackspace/gophercloud/openstack/db/v1/users"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
	"github.com/rackspace/gophercloud/testhelper/fixture"
)

var (
	_rootURL = "/instances"
	resURL   = "/instances/" + instanceID
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, _rootURL, "POST", createReq, createResp, 200)

	opts := CreateOpts{
		Name:      "json_rack_instance",
		FlavorRef: "1",
		Databases: osDBs.BatchCreateOpts{
			osDBs.CreateOpts{CharSet: "utf8", Collate: "utf8_general_ci", Name: "sampledb"},
			osDBs.CreateOpts{Name: "nextround"},
		},
		Users: osUsers.BatchCreateOpts{
			osUsers.CreateOpts{
				Name:     "demouser",
				Password: "demopassword",
				Databases: osDBs.BatchCreateOpts{
					osDBs.CreateOpts{Name: "sampledb"},
				},
			},
		},
		Size:         2,
		RestorePoint: "1234567890",
	}

	instance, err := Create(fake.ServiceClient(), opts).Extract()

	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expectedInstance, instance)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, resURL, "GET", "", getResp, 200)

	instance, err := Get(fake.ServiceClient(), instanceID).Extract()

	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expectedInstance, instance)
}

func TestDeleteInstance(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleDelete(t)

	res := Delete(fake.ServiceClient(), instanceID)
	th.AssertNoErr(t, res.Err)
}

func TestEnableRootUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleEnableRoot(t)

	expected := &osUsers.User{Name: "root", Password: "secretsecret"}

	user, err := EnableRootUser(fake.ServiceClient(), instanceID).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, user)
}

func TestRestart(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleRestart(t)

	res := Restart(fake.ServiceClient(), instanceID)
	th.AssertNoErr(t, res.Err)
}

func TestResize(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleResize(t)

	res := Resize(fake.ServiceClient(), instanceID, "2")
	th.AssertNoErr(t, res.Err)
}

func TestResizeVolume(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleResizeVol(t)

	res := ResizeVolume(fake.ServiceClient(), instanceID, 4)
	th.AssertNoErr(t, res.Err)
}
