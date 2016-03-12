package accounts

import (
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestUpdateAccount(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateAccountSuccessfully(t)

	options := &UpdateOpts{Metadata: map[string]string{"gophercloud-test": "accounts"}}
	res := Update(fake.ServiceClient(), options)
	th.AssertNoErr(t, res.Err)
}

func TestGetAccount(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetAccountSuccessfully(t)

	expectedMetadata := map[string]string{"Subject": "books"}
	res := Get(fake.ServiceClient(), &GetOpts{})
	th.AssertNoErr(t, res.Err)
	actualMetadata, _ := res.ExtractMetadata()
	th.CheckDeepEquals(t, expectedMetadata, actualMetadata)
	//headers, err := res.Extract()
	//th.AssertNoErr(t, err)
}
