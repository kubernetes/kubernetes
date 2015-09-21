package cdncontainers

import (
	"testing"

	os "github.com/rackspace/gophercloud/openstack/objectstorage/v1/containers"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestListCDNContainers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleListContainerNamesSuccessfully(t)

	count := 0
	err := List(fake.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractNames(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, os.ExpectedListNames, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestGetCDNContainer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleGetContainerSuccessfully(t)

	_, err := Get(fake.ServiceClient(), "testContainer").ExtractMetadata()
	th.CheckNoErr(t, err)

}

func TestUpdateCDNContainer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleUpdateContainerSuccessfully(t)

	options := &UpdateOpts{TTL: 3600}
	res := Update(fake.ServiceClient(), "testContainer", options)
	th.CheckNoErr(t, res.Err)

}
