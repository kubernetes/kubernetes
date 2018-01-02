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

	tools.PrintResource(t, quotaSet)
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

func getTenantIDByName(t *testing.T, client *gophercloud.ServiceClient, name string) (string, error) {
	allPages, err := tenants.List(client, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to get list of tenants: %v", err)
	}

	allTenants, err := tenants.ExtractTenants(allPages)
	if err != nil {
		t.Fatalf("Unable to extract tenants: %v", err)
	}

	for _, tenant := range allTenants {
		if tenant.Name == name {
			return tenant.ID, nil
		}
	}

	return "", fmt.Errorf("Unable to get tenant ID")
}

//What will be sent as desired Quotas to the Server
var UpdatQuotaOpts = quotasets.UpdateOpts{
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

//What the Server hopefully returns as the new Quotas
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
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	idclient, err := clients.NewIdentityV2Client()
	if err != nil {
		t.Fatalf("Could not create IdentityClient to look up tenant id!")
	}

	tenantid, err := getTenantIDByName(t, idclient, os.Getenv("OS_TENANT_NAME"))
	if err != nil {
		t.Fatalf("Id for Tenant named '%' not found. Please set OS_TENANT_NAME appropriately", os.Getenv("OS_TENANT_NAME"))
	}

	//save original quotas
	orig, err := quotasets.Get(client, tenantid).Extract()
	th.AssertNoErr(t, err)

	//Test Update
	res, err := quotasets.Update(client, tenantid, UpdatQuotaOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, UpdatedQuotas, *res)

	//Test Delete
	_, err = quotasets.Delete(client, tenantid).Extract()
	th.AssertNoErr(t, err)
	//We dont know the default quotas, so just check if the quotas are not the same as before
	newres, err := quotasets.Get(client, tenantid).Extract()
	if newres == res {
		t.Fatalf("Quotas after delete equal quotas before delete!")
	}

	restore := quotasets.UpdateOpts{}
	FillUpdateOptsFromQuotaSet(*orig, &restore)

	//restore original quotas
	res, err = quotasets.Update(client, tenantid, restore).Extract()
	th.AssertNoErr(t, err)

	orig.ID = ""
	th.AssertEquals(t, *orig, *res)

}

// Makes sure that the FillUpdateOptsFromQuotaSet() helper function works properly
func TestFillFromQuotaSetHelperFunction(t *testing.T) {
	op := &quotasets.UpdateOpts{}
	expected := `
	{
	"fixed_ips": 10,
	"floating_ips": 10,
	"injected_file_content_bytes": 10240,
	"injected_file_path_bytes": 255,
	"injected_files": 5,
	"key_pairs": 10,
	"metadata_items": 128,
	"ram": 20000,
	"security_group_rules": 20,
	"security_groups": 10,
	"cores": 10,
	"instances": 4,
	"server_groups": 2,
	"server_group_members": 3
	}`
	FillUpdateOptsFromQuotaSet(UpdatedQuotas, op)
	th.AssertJSONEquals(t, expected, op)
}
