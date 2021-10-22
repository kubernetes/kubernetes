package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/keymanager/v1/containers"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListContainers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListContainersSuccessfully(t)

	count := 0
	err := containers.List(client.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++

		actual, err := containers.ExtractContainers(page)
		th.AssertNoErr(t, err)

		th.AssertDeepEquals(t, ExpectedContainersSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.AssertEquals(t, count, 1)
}

func TestListContainersAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListContainersSuccessfully(t)

	allPages, err := containers.List(client.ServiceClient(), nil).AllPages()
	th.AssertNoErr(t, err)
	actual, err := containers.ExtractContainers(allPages)
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedContainersSlice, actual)
}

func TestGetContainer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetContainerSuccessfully(t)

	actual, err := containers.Get(client.ServiceClient(), "dfdb88f3-4ddb-4525-9da6-066453caa9b0").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, FirstContainer, *actual)
}

func TestCreateContainer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateContainerSuccessfully(t)

	createOpts := containers.CreateOpts{
		Type: containers.GenericContainer,
		Name: "mycontainer",
		SecretRefs: []containers.SecretRef{
			{
				Name:      "mysecret",
				SecretRef: "http://barbican:9311/v1/secrets/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c",
			},
		},
	}

	actual, err := containers.Create(client.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, FirstContainer, *actual)
}

func TestDeleteContainer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteContainerSuccessfully(t)

	res := containers.Delete(client.ServiceClient(), "dfdb88f3-4ddb-4525-9da6-066453caa9b0")
	th.AssertNoErr(t, res.Err)
}

func TestListConsumers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListConsumersSuccessfully(t)

	count := 0
	err := containers.ListConsumers(client.ServiceClient(), "dfdb88f3-4ddb-4525-9da6-066453caa9b0", nil).EachPage(func(page pagination.Page) (bool, error) {
		count++

		actual, err := containers.ExtractConsumers(page)
		th.AssertNoErr(t, err)

		th.AssertDeepEquals(t, ExpectedConsumersSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.AssertEquals(t, count, 1)
}

func TestListConsumersAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListConsumersSuccessfully(t)

	allPages, err := containers.ListConsumers(client.ServiceClient(), "dfdb88f3-4ddb-4525-9da6-066453caa9b0", nil).AllPages()
	th.AssertNoErr(t, err)
	actual, err := containers.ExtractConsumers(allPages)
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedConsumersSlice, actual)
}

func TestCreateConsumer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateConsumerSuccessfully(t)

	createOpts := containers.CreateConsumerOpts{
		Name: "CONSUMER-LZILN1zq",
		URL:  "http://example.com",
	}

	actual, err := containers.CreateConsumer(client.ServiceClient(), "dfdb88f3-4ddb-4525-9da6-066453caa9b0", createOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCreatedConsumer, *actual)
}

func TestDeleteConsumer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteConsumerSuccessfully(t)

	deleteOpts := containers.DeleteConsumerOpts{
		Name: "CONSUMER-LZILN1zq",
		URL:  "http://example.com",
	}

	actual, err := containers.DeleteConsumer(client.ServiceClient(), "dfdb88f3-4ddb-4525-9da6-066453caa9b0", deleteOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, FirstContainer, *actual)
}
