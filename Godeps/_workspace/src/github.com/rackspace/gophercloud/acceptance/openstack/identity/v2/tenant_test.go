// +build acceptance identity

package v2

import (
	"testing"

	tenants2 "github.com/rackspace/gophercloud/openstack/identity/v2/tenants"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestEnumerateTenants(t *testing.T) {
	service := authenticatedClient(t)

	t.Logf("Tenants to which your current token grants access:")
	count := 0
	err := tenants2.List(service, nil).EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page %02d ---", count)

		tenants, err := tenants2.ExtractTenants(page)
		th.AssertNoErr(t, err)
		for i, tenant := range tenants {
			t.Logf("[%02d] name=[%s] id=[%s] description=[%s] enabled=[%v]",
				i, tenant.Name, tenant.ID, tenant.Description, tenant.Enabled)
		}

		count++
		return true, nil
	})
	th.AssertNoErr(t, err)
}
