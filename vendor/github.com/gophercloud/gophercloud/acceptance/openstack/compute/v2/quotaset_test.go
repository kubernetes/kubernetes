// +build acceptance compute quotasets

package v2

import (
	"fmt"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/quotasets"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/tenants"
)

func TestQuotasetGet(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	identityClient, err := clients.NewIdentityV2Client()
	if err != nil {
		t.Fatalf("Unable to get a new identity client: %v", err)
	}

	tenantID, err := getTenantID(t, identityClient)
	if err != nil {
		t.Fatal(err)
	}

	quotaSet, err := quotasets.Get(client, tenantID).Extract()
	if err != nil {
		t.Fatal(err)
	}

	PrintQuotaSet(t, quotaSet)
}

func getTenantID(t *testing.T, client *gophercloud.ServiceClient) (string, error) {
	allPages, err := tenants.List(client, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to get list of tenants: %v", err)
	}

	allTenants, err := tenants.ExtractTenants(allPages)
	if err != nil {
		t.Fatalf("Unable to extract tenants: %v", err)
	}

	for _, tenant := range allTenants {
		return tenant.ID, nil
	}

	return "", fmt.Errorf("Unable to get tenant ID")
}
