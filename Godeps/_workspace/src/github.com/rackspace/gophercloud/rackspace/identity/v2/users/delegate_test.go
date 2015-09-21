package users

import (
	"testing"

	os "github.com/rackspace/gophercloud/openstack/identity/v2/users"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockListResponse(t)

	count := 0

	err := List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		users, err := os.ExtractUsers(page)

		th.AssertNoErr(t, err)
		th.AssertEquals(t, "u1000", users[0].ID)
		th.AssertEquals(t, "u1001", users[1].ID)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, count)
}

func TestCreateUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockCreateUser(t)

	opts := CreateOpts{
		Username: "new_user",
		Enabled:  os.Disabled,
		Email:    "new_user@foo.com",
		Password: "foo",
	}

	user, err := Create(client.ServiceClient(), opts).Extract()

	th.AssertNoErr(t, err)

	th.AssertEquals(t, "123456", user.ID)
	th.AssertEquals(t, "5830280", user.DomainID)
	th.AssertEquals(t, "DFW", user.DefaultRegion)
}

func TestGetUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockGetUser(t)

	user, err := Get(client.ServiceClient(), "new_user").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, true, user.Enabled)
	th.AssertEquals(t, true, user.MultiFactorEnabled)
}

func TestUpdateUser(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockUpdateUser(t)

	id := "c39e3de9be2d4c779f1dfd6abacc176d"

	opts := UpdateOpts{
		Enabled: os.Enabled,
		Email:   "new_email@foo.com",
	}

	user, err := Update(client.ServiceClient(), id, opts).Extract()

	th.AssertNoErr(t, err)

	th.AssertEquals(t, true, user.Enabled)
	th.AssertEquals(t, "new_email@foo.com", user.Email)
}

func TestDeleteServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDeleteUser(t)

	res := Delete(client.ServiceClient(), "c39e3de9be2d4c779f1dfd6abacc176d")
	th.AssertNoErr(t, res.Err)
}

func TestResetAPIKey(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockResetAPIKey(t)

	apiKey, err := ResetAPIKey(client.ServiceClient(), "99").Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, "joesmith", apiKey.Username)
	th.AssertEquals(t, "mooH1eiLahd5ahYood7r", apiKey.APIKey)
}
