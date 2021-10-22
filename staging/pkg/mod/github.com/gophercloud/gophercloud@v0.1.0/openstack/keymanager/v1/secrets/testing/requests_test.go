package testing

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/keymanager/v1/secrets"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListSecrets(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListSecretsSuccessfully(t)

	count := 0
	err := secrets.List(client.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++

		actual, err := secrets.ExtractSecrets(page)
		th.AssertNoErr(t, err)

		th.AssertDeepEquals(t, ExpectedSecretsSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.AssertEquals(t, count, 1)
}

func TestListSecretsAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListSecretsSuccessfully(t)

	allPages, err := secrets.List(client.ServiceClient(), nil).AllPages()
	th.AssertNoErr(t, err)
	actual, err := secrets.ExtractSecrets(allPages)
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedSecretsSlice, actual)
}

func TestGetSecret(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSecretSuccessfully(t)

	actual, err := secrets.Get(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, FirstSecret, *actual)
}

func TestCreateSecret(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateSecretSuccessfully(t)

	expiration := time.Date(2028, 6, 21, 2, 49, 48, 0, time.UTC)
	createOpts := secrets.CreateOpts{
		Algorithm:          "aes",
		BitLength:          256,
		Mode:               "cbc",
		Name:               "mysecret",
		Payload:            "foobar",
		PayloadContentType: "text/plain",
		SecretType:         secrets.OpaqueSecret,
		Expiration:         &expiration,
	}

	actual, err := secrets.Create(client.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCreateResult, *actual)
}

func TestDeleteSecret(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteSecretSuccessfully(t)

	res := secrets.Delete(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c")
	th.AssertNoErr(t, res.Err)
}

func TestUpdateSecret(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateSecretSuccessfully(t)

	updateOpts := secrets.UpdateOpts{
		Payload: "foobar",
	}

	err := secrets.Update(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c", updateOpts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestGetPayloadSecret(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetPayloadSuccessfully(t)

	res := secrets.GetPayload(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c", nil)
	th.AssertNoErr(t, res.Err)
	payload, err := res.Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, GetPayloadResponse, string(payload))
}

func TestGetMetadataSuccessfully(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetMetadataSuccessfully(t)

	actual, err := secrets.GetMetadata(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedMetadata, actual)
}

func TestCreateMetadataSuccessfully(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateMetadataSuccessfully(t)

	createOpts := secrets.MetadataOpts{
		"foo":       "bar",
		"something": "something else",
	}

	actual, err := secrets.CreateMetadata(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c", createOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCreateMetadataResult, actual)
}

func TestGetMetadatumSuccessfully(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetMetadatumSuccessfully(t)

	actual, err := secrets.GetMetadatum(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c", "foo").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedMetadatum, *actual)
}

func TestCreateMetadatumSuccessfully(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateMetadatumSuccessfully(t)

	createOpts := secrets.MetadatumOpts{
		Key:   "foo",
		Value: "bar",
	}

	err := secrets.CreateMetadatum(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c", createOpts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestUpdateMetadatumSuccessfully(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateMetadatumSuccessfully(t)

	updateOpts := secrets.MetadatumOpts{
		Key:   "foo",
		Value: "bar",
	}

	actual, err := secrets.UpdateMetadatum(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c", updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedMetadatum, *actual)
}

func TestDeleteMetadatumSuccessfully(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteMetadatumSuccessfully(t)

	err := secrets.DeleteMetadatum(client.ServiceClient(), "1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c", "foo").ExtractErr()
	th.AssertNoErr(t, err)
}
