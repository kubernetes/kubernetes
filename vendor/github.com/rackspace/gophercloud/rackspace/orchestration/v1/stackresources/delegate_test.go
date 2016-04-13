package stackresources

import (
	"testing"

	os "github.com/rackspace/gophercloud/openstack/orchestration/v1/stackresources"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestFindResources(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleFindSuccessfully(t, os.FindOutput)

	actual, err := Find(fake.ServiceClient(), "hello_world").Extract()
	th.AssertNoErr(t, err)

	expected := os.FindExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestListResources(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleListSuccessfully(t, os.ListOutput)

	count := 0
	err := List(fake.ServiceClient(), "hello_world", "49181cd6-169a-4130-9455-31185bbfc5bf", nil).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := os.ExtractResources(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, os.ListExpected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestGetResource(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleGetSuccessfully(t, os.GetOutput)

	actual, err := Get(fake.ServiceClient(), "teststack", "0b1771bd-9336-4f2b-ae86-a80f971faf1e", "wordpress_instance").Extract()
	th.AssertNoErr(t, err)

	expected := os.GetExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestResourceMetadata(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleMetadataSuccessfully(t, os.MetadataOutput)

	actual, err := Metadata(fake.ServiceClient(), "teststack", "0b1771bd-9336-4f2b-ae86-a80f971faf1e", "wordpress_instance").Extract()
	th.AssertNoErr(t, err)

	expected := os.MetadataExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestListResourceTypes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleListTypesSuccessfully(t, os.ListTypesOutput)

	count := 0
	err := ListTypes(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := os.ExtractResourceTypes(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, os.ListTypesExpected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, count)
}

func TestGetResourceSchema(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleGetSchemaSuccessfully(t, os.GetSchemaOutput)

	actual, err := Schema(fake.ServiceClient(), "OS::Heat::AResourceName").Extract()
	th.AssertNoErr(t, err)

	expected := os.GetSchemaExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestGetResourceTemplate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleGetTemplateSuccessfully(t, os.GetTemplateOutput)

	actual, err := Template(fake.ServiceClient(), "OS::Heat::AResourceName").Extract()
	th.AssertNoErr(t, err)

	expected := os.GetTemplateExpected
	th.AssertDeepEquals(t, expected, string(actual))
}
