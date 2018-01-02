// +build acceptance identity

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/tenants"
)

func TestTenantsList(t *testing.T) {
	client, err := clients.NewIdentityV2Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v")
	}

	allPages, err := tenants.List(client, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to list tenants: %v", err)
	}

	allTenants, err := tenants.ExtractTenants(allPages)
	if err != nil {
		t.Fatalf("Unable to extract tenants: %v", err)
	}

	for _, tenant := range allTenants {
		tools.PrintResource(t, tenant)
	}
}

func TestTenantsCRUD(t *testing.T) {
	client, err := clients.NewIdentityV2AdminClient()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v")
	}

	tenant, err := CreateTenant(t, client, nil)
	if err != nil {
		t.Fatalf("Unable to create tenant: %v", err)
	}
	defer DeleteTenant(t, client, tenant.ID)

	tenant, err = tenants.Get(client, tenant.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get tenant: %v", err)
	}

	tools.PrintResource(t, tenant)

	updateOpts := tenants.UpdateOpts{
		Description: "some tenant",
	}

	newTenant, err := tenants.Update(client, tenant.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update tenant: %v", err)
	}

	tools.PrintResource(t, newTenant)
}
