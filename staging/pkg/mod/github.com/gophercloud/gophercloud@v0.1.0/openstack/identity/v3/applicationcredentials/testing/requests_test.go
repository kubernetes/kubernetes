package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/applicationcredentials"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListApplicationCredentials(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListApplicationCredentialsSuccessfully(t)

	count := 0
	err := applicationcredentials.List(client.ServiceClient(), userID, nil).EachPage(func(page pagination.Page) (bool, error) {
		count++

		actual, err := applicationcredentials.ExtractApplicationCredentials(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ExpectedApplicationCredentialsSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestListApplicationCredentialsAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListApplicationCredentialsSuccessfully(t)

	allPages, err := applicationcredentials.List(client.ServiceClient(), userID, nil).AllPages()
	th.AssertNoErr(t, err)
	actual, err := applicationcredentials.ExtractApplicationCredentials(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedApplicationCredentialsSlice, actual)
	th.AssertDeepEquals(t, ExpectedApplicationCredentialsSlice[0].Roles, []applicationcredentials.Role{{ID: "31f87923ae4a4d119aa0b85dcdbeed13", Name: "compute_viewer"}})
	th.AssertDeepEquals(t, ExpectedApplicationCredentialsSlice[1].Roles, []applicationcredentials.Role{applicationcredentials.Role{ID: "31f87923ae4a4d119aa0b85dcdbeed13", Name: "compute_viewer"}, applicationcredentials.Role{ID: "4494bc5bea1a4105ad7fbba6a7eb9ef4", Name: "network_viewer"}})
}

func TestGetApplicationCredential(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetApplicationCredentialSuccessfully(t)

	actual, err := applicationcredentials.Get(client.ServiceClient(), userID, applicationCredentialID).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ApplicationCredential, *actual)
}

func TestCreateApplicationCredential(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateApplicationCredentialSuccessfully(t)

	createOpts := applicationcredentials.CreateOpts{
		Name:   "test",
		Secret: "mysecret",
		Roles: []applicationcredentials.Role{
			applicationcredentials.Role{ID: "31f87923ae4a4d119aa0b85dcdbeed13"},
		},
	}

	ApplicationCredentialResponse := ApplicationCredential
	ApplicationCredentialResponse.Secret = "mysecret"

	actual, err := applicationcredentials.Create(client.ServiceClient(), userID, createOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ApplicationCredentialResponse, *actual)
}

func TestCreateNoSecretApplicationCredential(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateNoSecretApplicationCredentialSuccessfully(t)

	createOpts := applicationcredentials.CreateOpts{
		Name: "test1",
		Roles: []applicationcredentials.Role{
			applicationcredentials.Role{ID: "31f87923ae4a4d119aa0b85dcdbeed13"},
		},
	}

	actual, err := applicationcredentials.Create(client.ServiceClient(), userID, createOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ApplicationCredentialNoSecretResponse, *actual)
}

func TestCreateUnrestrictedApplicationCredential(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateUnrestrictedApplicationCredentialSuccessfully(t)

	createOpts := applicationcredentials.CreateOpts{
		Name:         "test2",
		Unrestricted: true,
		Roles: []applicationcredentials.Role{
			applicationcredentials.Role{ID: "31f87923ae4a4d119aa0b85dcdbeed13"},
			applicationcredentials.Role{ID: "4494bc5bea1a4105ad7fbba6a7eb9ef4"},
		},
		ExpiresAt: "2019-03-12T12:12:12.000000",
	}

	UnrestrictedApplicationCredentialResponse := UnrestrictedApplicationCredential
	UnrestrictedApplicationCredentialResponse.Secret = "generated_secret"

	actual, err := applicationcredentials.Create(client.ServiceClient(), userID, createOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, UnrestrictedApplicationCredentialResponse, *actual)
}

func TestDeleteApplicationCredential(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteApplicationCredentialSuccessfully(t)

	res := applicationcredentials.Delete(client.ServiceClient(), userID, applicationCredentialID)
	th.AssertNoErr(t, res.Err)
}
