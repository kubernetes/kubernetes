// +build acceptance

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	rstenants "github.com/rackspace/gophercloud/rackspace/identity/v2/tenants"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestTenants(t *testing.T) {
	service := authenticatedClient(t)

	t.Logf("Tenants available to the currently issued token:")
	count := 0
	err := rstenants.List(service, nil).EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page %02d ---", count)

		tenants, err := rstenants.ExtractTenants(page)
		th.AssertNoErr(t, err)

		for i, tenant := range tenants {
			t.Logf("[%02d]      id=[%s]", i, tenant.ID)
			t.Logf("        name=[%s] enabled=[%v]", i, tenant.Name, tenant.Enabled)
			t.Logf(" description=[%s]", tenant.Description)
		}

		count++
		return true, nil
	})
	th.AssertNoErr(t, err)
	if count == 0 {
		t.Errorf("No tenants listed for your current token.")
	}
}
