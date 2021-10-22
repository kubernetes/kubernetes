package testing

import (
	"testing"

	common "github.com/gophercloud/gophercloud/openstack/common/extensions"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListExtensionsSuccessfully(t)

	count := 0
	extensions.List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := extensions.ExtractExtensions(page)
		th.AssertNoErr(t, err)

		expected := []common.Extension{
			common.Extension{
				Updated:     "2013-01-20T00:00:00-00:00",
				Name:        "Neutron Service Type Management",
				Links:       []interface{}{},
				Namespace:   "http://docs.openstack.org/ext/neutron/service-type/api/v1.0",
				Alias:       "service-type",
				Description: "API for retrieving service providers for Neutron advanced services",
			},
		}
		th.AssertDeepEquals(t, expected, actual)

		return true, nil
	})
	th.CheckEquals(t, 1, count)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGetExtensionsSuccessfully(t)

	ext, err := extensions.Get(client.ServiceClient(), "agent").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, ext.Updated, "2013-02-03T10:00:00-00:00")
	th.AssertEquals(t, ext.Name, "agent")
	th.AssertEquals(t, ext.Namespace, "http://docs.openstack.org/ext/agent/api/v2.0")
	th.AssertEquals(t, ext.Alias, "agent")
	th.AssertEquals(t, ext.Description, "The agent management extension.")
}
