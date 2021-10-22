package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/keymanager/v1/acls"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestGetSecretACL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSecretACLSuccessfully(t)

	actual, err := acls.GetSecretACL(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedACL, *actual)
}

func TestGetContainerACL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetContainerACLSuccessfully(t)

	actual, err := acls.GetContainerACL(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedACL, *actual)
}

func TestSetSecretACL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleSetSecretACLSuccessfully(t)

	users := []string{"GG27dVwR9gBMnsOaRoJ1DFJmZfdVjIdW"}
	iFalse := false
	setOpts := acls.SetOpts{
		Type:          "read",
		Users:         &users,
		ProjectAccess: &iFalse,
	}

	actual, err := acls.SetSecretACL(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c", setOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedSecretACLRef, *actual)
}

func TestSetContainerACL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleSetContainerACLSuccessfully(t)

	users := []string{"GG27dVwR9gBMnsOaRoJ1DFJmZfdVjIdW"}
	iFalse := false
	setOpts := acls.SetOpts{
		Type:          "read",
		Users:         &users,
		ProjectAccess: &iFalse,
	}

	actual, err := acls.SetContainerACL(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c", setOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedContainerACLRef, *actual)
}

func TestDeleteSecretACL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteSecretACLSuccessfully(t)

	res := acls.DeleteSecretACL(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c")
	th.AssertNoErr(t, res.Err)
}

func TestDeleteContainerACL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteContainerACLSuccessfully(t)

	res := acls.DeleteContainerACL(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c")
	th.AssertNoErr(t, res.Err)
}

func TestUpdateSecretACL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateSecretACLSuccessfully(t)

	newUsers := []string{}
	updateOpts := acls.SetOpts{
		Type:  "read",
		Users: &newUsers,
	}

	actual, err := acls.UpdateSecretACL(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c", updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedSecretACLRef, *actual)
}

func TestUpdateContainerACL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateContainerACLSuccessfully(t)

	newUsers := []string{}
	updateOpts := acls.SetOpts{
		Type:  "read",
		Users: &newUsers,
	}

	actual, err := acls.UpdateContainerACL(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c", updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedContainerACLRef, *actual)
}
