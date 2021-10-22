package testing

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/objectstorage/v1/containers"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

var (
	metadata = map[string]string{"gophercloud-test": "containers"}
	loc, _   = time.LoadLocation("GMT")
)

func TestListContainerInfo(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListContainerInfoSuccessfully(t)

	count := 0
	err := containers.List(fake.ServiceClient(), &containers.ListOpts{Full: true}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := containers.ExtractInfo(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ExpectedListInfo, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestListAllContainerInfo(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListContainerInfoSuccessfully(t)

	allPages, err := containers.List(fake.ServiceClient(), &containers.ListOpts{Full: true}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := containers.ExtractInfo(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedListInfo, actual)
}

func TestListContainerNames(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListContainerNamesSuccessfully(t)

	count := 0
	err := containers.List(fake.ServiceClient(), &containers.ListOpts{Full: false}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := containers.ExtractNames(page)
		if err != nil {
			t.Errorf("Failed to extract container names: %v", err)
			return false, err
		}

		th.CheckDeepEquals(t, ExpectedListNames, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestListAllContainerNames(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListContainerNamesSuccessfully(t)

	allPages, err := containers.List(fake.ServiceClient(), &containers.ListOpts{Full: false}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := containers.ExtractNames(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedListNames, actual)
}

func TestCreateContainer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateContainerSuccessfully(t)

	options := containers.CreateOpts{ContentType: "application/json", Metadata: map[string]string{"foo": "bar"}}
	res := containers.Create(fake.ServiceClient(), "testContainer", options)
	th.CheckEquals(t, "bar", res.Header["X-Container-Meta-Foo"][0])

	expected := &containers.CreateHeader{
		ContentLength: 0,
		ContentType:   "text/html; charset=UTF-8",
		Date:          time.Date(2016, time.August, 17, 19, 25, 43, 0, loc), //Wed, 17 Aug 2016 19:25:43 GMT
		TransID:       "tx554ed59667a64c61866f1-0058b4ba37",
	}
	actual, err := res.Extract()
	th.CheckNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestDeleteContainer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteContainerSuccessfully(t)

	res := containers.Delete(fake.ServiceClient(), "testContainer")
	th.CheckNoErr(t, res.Err)
}

func TestUpdateContainer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateContainerSuccessfully(t)

	options := &containers.UpdateOpts{Metadata: map[string]string{"foo": "bar"}}
	res := containers.Update(fake.ServiceClient(), "testContainer", options)
	th.CheckNoErr(t, res.Err)
}

func TestGetContainer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetContainerSuccessfully(t)

	getOpts := containers.GetOpts{
		Newest: true,
	}
	res := containers.Get(fake.ServiceClient(), "testContainer", getOpts)
	_, err := res.ExtractMetadata()
	th.CheckNoErr(t, err)

	expected := &containers.GetHeader{
		AcceptRanges:  "bytes",
		BytesUsed:     100,
		ContentType:   "application/json; charset=utf-8",
		Date:          time.Date(2016, time.August, 17, 19, 25, 43, 0, loc), //Wed, 17 Aug 2016 19:25:43 GMT
		ObjectCount:   4,
		Read:          []string{"test"},
		TransID:       "tx554ed59667a64c61866f1-0057b4ba37",
		Write:         []string{"test2", "user4"},
		StoragePolicy: "test_policy",
	}
	actual, err := res.Extract()
	th.CheckNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}
