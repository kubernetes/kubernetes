package testing

import (
	"sort"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/orchestration/v1/stackresources"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestFindResources(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleFindSuccessfully(t, FindOutput)

	actual, err := stackresources.Find(fake.ServiceClient(), "hello_world").Extract()
	th.AssertNoErr(t, err)

	expected := FindExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestListResources(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListSuccessfully(t, ListOutput)

	count := 0
	err := stackresources.List(fake.ServiceClient(), "hello_world", "49181cd6-169a-4130-9455-31185bbfc5bf", nil).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := stackresources.ExtractResources(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ListExpected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestGetResource(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSuccessfully(t, GetOutput)

	actual, err := stackresources.Get(fake.ServiceClient(), "teststack", "0b1771bd-9336-4f2b-ae86-a80f971faf1e", "wordpress_instance").Extract()
	th.AssertNoErr(t, err)

	expected := GetExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestResourceMetadata(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleMetadataSuccessfully(t, MetadataOutput)

	actual, err := stackresources.Metadata(fake.ServiceClient(), "teststack", "0b1771bd-9336-4f2b-ae86-a80f971faf1e", "wordpress_instance").Extract()
	th.AssertNoErr(t, err)

	expected := MetadataExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestListResourceTypes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListTypesSuccessfully(t, ListTypesOutput)

	count := 0
	err := stackresources.ListTypes(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := stackresources.ExtractResourceTypes(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ListTypesExpected, actual)
		// test if sorting works
		sort.Sort(actual)
		th.CheckDeepEquals(t, SortedListTypesExpected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, count)
}

func TestGetResourceSchema(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSchemaSuccessfully(t, GetSchemaOutput)

	actual, err := stackresources.Schema(fake.ServiceClient(), "OS::Heat::AResourceName").Extract()
	th.AssertNoErr(t, err)

	expected := GetSchemaExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestGetResourceTemplate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetTemplateSuccessfully(t, GetTemplateOutput)

	actual, err := stackresources.Template(fake.ServiceClient(), "OS::Heat::AResourceName").Extract()
	th.AssertNoErr(t, err)

	expected := GetTemplateExpected
	th.AssertDeepEquals(t, expected, string(actual))
}
