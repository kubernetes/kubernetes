package testing

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/objectstorage/v1/accounts"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

var (
	loc, _ = time.LoadLocation("GMT")
)

func TestUpdateAccount(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateAccountSuccessfully(t)

	options := &accounts.UpdateOpts{Metadata: map[string]string{"gophercloud-test": "accounts"}}
	res := accounts.Update(fake.ServiceClient(), options)
	th.AssertNoErr(t, res.Err)

	expected := &accounts.UpdateHeader{
		Date: gophercloud.JSONRFC1123(time.Date(2014, time.January, 17, 16, 9, 56, 0, loc)), // Fri, 17 Jan 2014 16:09:56 GMT
	}
	actual, err := res.Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, expected, actual)
}

func TestGetAccount(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetAccountSuccessfully(t)

	expectedMetadata := map[string]string{"Subject": "books"}
	res := accounts.Get(fake.ServiceClient(), &accounts.GetOpts{})
	th.AssertNoErr(t, res.Err)
	actualMetadata, _ := res.ExtractMetadata()
	th.CheckDeepEquals(t, expectedMetadata, actualMetadata)
	_, err := res.Extract()
	th.AssertNoErr(t, err)

	expected := &accounts.GetHeader{
		ContainerCount: 2,
		ObjectCount:    5,
		BytesUsed:      14,
		Date:           gophercloud.JSONRFC1123(time.Date(2014, time.January, 17, 16, 9, 56, 0, loc)), // Fri, 17 Jan 2014 16:09:56 GMT
	}
	actual, err := res.Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, expected, actual)
}
