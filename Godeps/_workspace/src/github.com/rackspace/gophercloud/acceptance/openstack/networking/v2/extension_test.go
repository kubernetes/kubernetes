// +build acceptance networking

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestListExts(t *testing.T) {
	Setup(t)
	defer Teardown()

	pager := extensions.List(Client)
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page ---")

		exts, err := extensions.ExtractExtensions(page)
		th.AssertNoErr(t, err)

		for _, ext := range exts {
			t.Logf("Extension: Name [%s] Description [%s]", ext.Name, ext.Description)
		}

		return true, nil
	})
	th.CheckNoErr(t, err)
}

func TestGetExt(t *testing.T) {
	Setup(t)
	defer Teardown()

	ext, err := extensions.Get(Client, "service-type").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, ext.Updated, "2013-01-20T00:00:00-00:00")
	th.AssertEquals(t, ext.Name, "Neutron Service Type Management")
	th.AssertEquals(t, ext.Namespace, "http://docs.openstack.org/ext/neutron/service-type/api/v1.0")
	th.AssertEquals(t, ext.Alias, "service-type")
	th.AssertEquals(t, ext.Description, "API for retrieving service providers for Neutron advanced services")
}
