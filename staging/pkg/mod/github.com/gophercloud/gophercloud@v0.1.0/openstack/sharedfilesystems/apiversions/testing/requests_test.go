package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/apiversions"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListAPIVersions(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockListResponse(t)

	allVersions, err := apiversions.List(client.ServiceClient()).AllPages()
	th.AssertNoErr(t, err)

	actual, err := apiversions.ExtractAPIVersions(allVersions)
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, ManilaAllAPIVersionResults, actual)
}

func TestGetAPIVersion(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGetResponse(t)

	actual, err := apiversions.Get(client.ServiceClient(), "v2").Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, ManilaAPIVersion2Result, *actual)
}

func TestGetNoAPIVersion(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGetNoResponse(t)

	_, err := apiversions.Get(client.ServiceClient(), "v2").Extract()
	th.AssertEquals(t, err.Error(), "Unable to find requested API version")
}

func TestGetMultipleAPIVersion(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGetMultipleResponses(t)

	_, err := apiversions.Get(client.ServiceClient(), "v2").Extract()
	th.AssertEquals(t, err.Error(), "Found 2 API versions")
}
