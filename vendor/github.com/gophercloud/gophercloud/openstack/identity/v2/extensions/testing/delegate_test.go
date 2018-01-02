package testing

import (
	"testing"

	common "github.com/gophercloud/gophercloud/openstack/common/extensions/testing"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/extensions"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListExtensionsSuccessfully(t)

	count := 0
	err := extensions.List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := extensions.ExtractExtensions(page)
		th.AssertNoErr(t, err)
		th.CheckDeepEquals(t, common.ExpectedExtensions, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, count)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	common.HandleGetExtensionSuccessfully(t)

	actual, err := extensions.Get(client.ServiceClient(), "agent").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, common.SingleExtension, actual)
}
