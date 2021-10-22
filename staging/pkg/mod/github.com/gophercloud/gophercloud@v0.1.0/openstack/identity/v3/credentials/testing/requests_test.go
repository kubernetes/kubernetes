package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/credentials"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListCredentials(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListCredentialsSuccessfully(t)

	count := 0
	err := credentials.List(client.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++

		actual, err := credentials.ExtractCredentials(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ExpectedCredentialsSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestListCredentialsAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListCredentialsSuccessfully(t)

	allPages, err := credentials.List(client.ServiceClient(), nil).AllPages()
	th.AssertNoErr(t, err)
	actual, err := credentials.ExtractCredentials(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedCredentialsSlice, actual)
	th.AssertDeepEquals(t, ExpectedCredentialsSlice[0].Blob, "{\"access\":\"181920\",\"secret\":\"secretKey\"}")
	th.AssertDeepEquals(t, ExpectedCredentialsSlice[1].Blob, "{\"access\":\"7da79ff0aa364e1396f067e352b9b79a\",\"secret\":\"7a18d68ba8834b799d396f3ff6f1e98c\"}")
}

func TestGetCredential(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetCredentialSuccessfully(t)

	actual, err := credentials.Get(client.ServiceClient(), credentialID).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, Credential, *actual)
}

func TestCreateCredential(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateCredentialSuccessfully(t)

	createOpts := credentials.CreateOpts{
		ProjectID: projectID,
		Type:      "ec2",
		UserID:    userID,
		Blob:      "{\"access\":\"181920\",\"secret\":\"secretKey\"}",
	}

	CredentialResponse := Credential

	actual, err := credentials.Create(client.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, CredentialResponse, *actual)
}

func TestDeleteCredential(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteCredentialSuccessfully(t)

	res := credentials.Delete(client.ServiceClient(), credentialID)
	th.AssertNoErr(t, res.Err)
}

func TestUpdateCredential(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateCredentialSuccessfully(t)

	updateOpts := credentials.UpdateOpts{
		ProjectID: "731fc6f265cd486d900f16e84c5cb594",
		Type:      "ec2",
		UserID:    "bb5476fd12884539b41d5a88f838d773",
		Blob:      "{\"access\":\"181920\",\"secret\":\"secretKey\"}",
	}

	actual, err := credentials.Update(client.ServiceClient(), "2441494e52ab6d594a34d74586075cb299489bdd1e9389e3ab06467a4f460609", updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, SecondCredentialUpdated, *actual)
}
