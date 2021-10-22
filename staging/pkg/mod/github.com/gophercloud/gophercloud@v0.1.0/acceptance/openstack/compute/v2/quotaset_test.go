// +build acceptance compute quotasets

package v2

import (
	"fmt"
	"os"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/quotasets"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/tenants"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestQuotasetGet(t *testing.T) {
	clients.SkipRelease(t, "master")
	clients.SkipRelease(t, "stable/queens")
	clients.SkipRelease(t, "stable/rocky")
	clients.SkipRelease(t, "stable/stein")

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	identityClient, err := clients.NewIdentityV2Client()
	th.AssertNoErr(t, err)

	tenantID, err := getTenantID(t, identityClient)
	th.AssertNoErr(t, err)

	quotaSet, err := quotasets.Get(client, tenantID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, quotaSet)

	th.AssertEquals(t, quotaSet.FixedIPs, -1)
}

func getTenantID(t *testing.T, client *gophercloud.ServiceClient) (string, error) {
	allPages, err := tenants.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allTenants, err := tenants.ExtractTenants(allPages)
	th.AssertNoErr(t, err)

	for _, tenant := range allTenants {
		return tenant.ID, nil
	}

	return "", fmt.Errorf("Unable to get tenant ID")
}

func getTenantIDByName(t *testing.T, client *gophercloud.ServiceClient, name string) (string, error) {
	allPages, err := tenants.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allTenants, err := tenants.ExtractTenants(allPages)
	th.AssertNoErr(t, err)

	for _, tenant := range allTenants {
		if tenant.Name == name {
			return tenant.ID, nil
		}
	}

	return "", fmt.Errorf("Unable to get tenant ID")
}

// What will be sent as desired Quotas to the Server
var UpdateQuotaOpts = quotasets.UpdateOpts{
	FixedIPs:                 gophercloud.IntToPointer(10),
	FloatingIPs:              gophercloud.IntToPointer(10),
	InjectedFileContentBytes: gophercloud.IntToPointer(10240),
	InjectedFilePathBytes:    gophercloud.IntToPointer(255),
	InjectedFiles:            gophercloud.IntToPointer(5),
	KeyPairs:                 gophercloud.IntToPointer(10),
	MetadataItems:            gophercloud.IntToPointer(128),
	RAM:                      gophercloud.IntToPointer(20000),
	SecurityGroupRules:       gophercloud.IntToPointer(20),
	SecurityGroups:           gophercloud.IntToPointer(10),
	Cores:                    gophercloud.IntToPointer(10),
	Instances:                gophercloud.IntToPointer(4),
	ServerGroups:             gophercloud.IntToPointer(2),
	ServerGroupMembers:       gophercloud.IntToPointer(3),
}

// What the Server hopefully returns as the new Quotas
var UpdatedQuotas = quotasets.QuotaSet{
	FixedIPs:                 10,
	FloatingIPs:              10,
	InjectedFileContentBytes: 10240,
	InjectedFilePathBytes:    255,
	InjectedFiles:            5,
	KeyPairs:                 10,
	MetadataItems:            128,
	RAM:                      20000,
	SecurityGroupRules:       20,
	SecurityGroups:           10,
	Cores:                    10,
	Instances:                4,
	ServerGroups:             2,
	ServerGroupMembers:       3,
}

func TestQuotasetUpdateDelete(t *testing.T) {
	clients.SkipRelease(t, "master")
	clients.SkipRelease(t, "stable/queens")
	clients.SkipRelease(t, "stable/rocky")
	clients.SkipRelease(t, "stable/stein")

	clients.RequireAdmin(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	idclient, err := clients.NewIdentityV2Client()
	th.AssertNoErr(t, err)

	tenantid, err := getTenantIDByName(t, idclient, os.Getenv("OS_TENANT_NAME"))
	th.AssertNoErr(t, err)

	// save original quotas
	orig, err := quotasets.Get(client, tenantid).Extract()
	th.AssertNoErr(t, err)

	// Test Update
	res, err := quotasets.Update(client, tenantid, UpdateQuotaOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, UpdatedQuotas, *res)

	// Test Delete
	_, err = quotasets.Delete(client, tenantid).Extract()
	th.AssertNoErr(t, err)

	// We dont know the default quotas, so just check if the quotas are not the same as before
	newres, err := quotasets.Get(client, tenantid).Extract()
	th.AssertNoErr(t, err)
	if newres.RAM == res.RAM {
		t.Fatalf("Failed to update quotas")
	}

	restore := quotasets.UpdateOpts{}
	FillUpdateOptsFromQuotaSet(*orig, &restore)

	// restore original quotas
	res, err = quotasets.Update(client, tenantid, restore).Extract()
	th.AssertNoErr(t, err)

	orig.ID = ""
	th.AssertDeepEquals(t, orig, res)
}
