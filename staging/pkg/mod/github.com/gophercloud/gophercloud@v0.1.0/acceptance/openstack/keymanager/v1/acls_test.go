// +build acceptance keymanager acls

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/keymanager/v1/acls"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestACLCRUD(t *testing.T) {
	client, err := clients.NewKeyManagerV1Client()
	th.AssertNoErr(t, err)

	payload := tools.RandomString("SUPERSECRET-", 8)
	secret, err := CreateSecretWithPayload(t, client, payload)
	th.AssertNoErr(t, err)
	secretID, err := ParseID(secret.SecretRef)
	th.AssertNoErr(t, err)
	defer DeleteSecret(t, client, secretID)

	user := tools.RandomString("", 32)
	users := []string{user}
	iFalse := false
	setOpts := acls.SetOpts{
		Type:          "read",
		Users:         &users,
		ProjectAccess: &iFalse,
	}

	aclRef, err := acls.SetSecretACL(client, secretID, setOpts).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, aclRef)
	defer func() {
		err := acls.DeleteSecretACL(client, secretID).ExtractErr()
		th.AssertNoErr(t, err)
		acl, err := acls.GetSecretACL(client, secretID).Extract()
		th.AssertNoErr(t, err)
		tools.PrintResource(t, acl)
	}()

	acl, err := acls.GetSecretACL(client, secretID).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, acl)
	tools.PrintResource(t, (*acl)["read"].Created)
	th.AssertEquals(t, len((*acl)["read"].Users), 1)
	th.AssertEquals(t, (*acl)["read"].ProjectAccess, false)

	newUsers := []string{}
	updateOpts := acls.SetOpts{
		Type:  "read",
		Users: &newUsers,
	}

	aclRef, err = acls.UpdateSecretACL(client, secretID, updateOpts).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, aclRef)

	acl, err = acls.GetSecretACL(client, secretID).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, acl)
	tools.PrintResource(t, (*acl)["read"].Created)
	th.AssertEquals(t, len((*acl)["read"].Users), 0)
	th.AssertEquals(t, (*acl)["read"].ProjectAccess, false)

	container, err := CreateGenericContainer(t, client, secret)
	th.AssertNoErr(t, err)
	containerID, err := ParseID(container.ContainerRef)
	th.AssertNoErr(t, err)
	defer DeleteContainer(t, client, containerID)

	aclRef, err = acls.SetContainerACL(client, containerID, setOpts).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, aclRef)
	defer func() {
		err := acls.DeleteContainerACL(client, containerID).ExtractErr()
		th.AssertNoErr(t, err)
		acl, err := acls.GetContainerACL(client, containerID).Extract()
		th.AssertNoErr(t, err)
		tools.PrintResource(t, acl)
	}()

	acl, err = acls.GetContainerACL(client, containerID).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, acl)
	tools.PrintResource(t, (*acl)["read"].Created)
	th.AssertEquals(t, len((*acl)["read"].Users), 1)
	th.AssertEquals(t, (*acl)["read"].ProjectAccess, false)

	aclRef, err = acls.UpdateContainerACL(client, containerID, updateOpts).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, aclRef)

	acl, err = acls.GetContainerACL(client, containerID).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, acl)
	tools.PrintResource(t, (*acl)["read"].Created)
	th.AssertEquals(t, len((*acl)["read"].Users), 0)
	th.AssertEquals(t, (*acl)["read"].ProjectAccess, false)
}
