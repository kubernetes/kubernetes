// +build acceptance identity

package v2

import (
	"strconv"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack/identity/v2/tenants"
	"github.com/rackspace/gophercloud/openstack/identity/v2/users"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestUsers(t *testing.T) {
	client := authenticatedClient(t)

	tenantID := findTenant(t, client)

	userID := createUser(t, client, tenantID)

	listUsers(t, client)

	getUser(t, client, userID)

	updateUser(t, client, userID)

	listUserRoles(t, client, tenantID, userID)

	deleteUser(t, client, userID)
}

func findTenant(t *testing.T, client *gophercloud.ServiceClient) string {
	var tenantID string
	err := tenants.List(client, nil).EachPage(func(page pagination.Page) (bool, error) {
		tenantList, err := tenants.ExtractTenants(page)
		th.AssertNoErr(t, err)

		for _, t := range tenantList {
			tenantID = t.ID
			break
		}

		return true, nil
	})
	th.AssertNoErr(t, err)

	return tenantID
}

func createUser(t *testing.T, client *gophercloud.ServiceClient, tenantID string) string {
	t.Log("Creating user")

	opts := users.CreateOpts{
		Name:     tools.RandomString("user_", 5),
		Enabled:  users.Disabled,
		TenantID: tenantID,
		Email:    "new_user@foo.com",
	}

	user, err := users.Create(client, opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created user %s on tenant %s", user.ID, tenantID)

	return user.ID
}

func listUsers(t *testing.T, client *gophercloud.ServiceClient) {
	err := users.List(client).EachPage(func(page pagination.Page) (bool, error) {
		userList, err := users.ExtractUsers(page)
		th.AssertNoErr(t, err)

		for _, user := range userList {
			t.Logf("Listing user: ID [%s] Name [%s] Email [%s] Enabled? [%s]",
				user.ID, user.Name, user.Email, strconv.FormatBool(user.Enabled))
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
}

func getUser(t *testing.T, client *gophercloud.ServiceClient, userID string) {
	_, err := users.Get(client, userID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Getting user %s", userID)
}

func updateUser(t *testing.T, client *gophercloud.ServiceClient, userID string) {
	opts := users.UpdateOpts{Name: tools.RandomString("new_name", 5), Email: "new@foo.com"}
	user, err := users.Update(client, userID, opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Updated user %s: Name [%s] Email [%s]", userID, user.Name, user.Email)
}

func listUserRoles(t *testing.T, client *gophercloud.ServiceClient, tenantID, userID string) {
	count := 0
	err := users.ListRoles(client, tenantID, userID).EachPage(func(page pagination.Page) (bool, error) {
		count++

		roleList, err := users.ExtractRoles(page)
		th.AssertNoErr(t, err)

		t.Logf("Listing roles for user %s", userID)

		for _, r := range roleList {
			t.Logf("- %s (%s)", r.Name, r.ID)
		}

		return true, nil
	})

	if count == 0 {
		t.Logf("No roles for user %s", userID)
	}

	th.AssertNoErr(t, err)
}

func deleteUser(t *testing.T, client *gophercloud.ServiceClient, userID string) {
	res := users.Delete(client, userID)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted user %s", userID)
}
