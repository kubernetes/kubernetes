package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/usage"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestGetTenant(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSingleTenantSuccessfully(t)

	count := 0
	err := usage.SingleTenant(client.ServiceClient(), FirstTenantID, nil).EachPage(func(page pagination.Page) (bool, error) {
		count++

		actual, err := usage.ExtractSingleTenant(page)
		th.AssertNoErr(t, err)
		th.AssertDeepEquals(t, &SingleTenantUsageResults, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.AssertEquals(t, count, 1)
}

func TestAllTenants(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetAllTenantsSuccessfully(t)

	getOpts := usage.AllTenantsOpts{
		Detailed: true,
	}

	count := 0
	err := usage.AllTenants(client.ServiceClient(), getOpts).EachPage(func(page pagination.Page) (bool, error) {
		count++

		actual, err := usage.ExtractAllTenants(page)
		th.AssertNoErr(t, err)
		th.AssertDeepEquals(t, AllTenantsUsageResult, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.AssertEquals(t, count, 1)
}
