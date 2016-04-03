// +build acceptance compute

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/quotasets"
	"github.com/rackspace/gophercloud/openstack/identity/v2/tenants"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestGetQuotaset(t *testing.T) {
	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	idclient := openstack.NewIdentityV2(client.ProviderClient)
	quotaset, err := quotasets.Get(client, findTenant(t, idclient)).Extract()
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("QuotaSet details:\n")
	t.Logf("                   instances=[%d]\n", quotaset.Instances)
	t.Logf("                       cores=[%d]\n", quotaset.Cores)
	t.Logf("                         ram=[%d]\n", quotaset.Ram)
	t.Logf("                   key_pairs=[%d]\n", quotaset.KeyPairs)
	t.Logf("              metadata_items=[%d]\n", quotaset.MetadataItems)
	t.Logf("             security_groups=[%d]\n", quotaset.SecurityGroups)
	t.Logf("        security_group_rules=[%d]\n", quotaset.SecurityGroupRules)
	t.Logf("                   fixed_ips=[%d]\n", quotaset.FixedIps)
	t.Logf("                floating_ips=[%d]\n", quotaset.FloatingIps)
	t.Logf(" injected_file_content_bytes=[%d]\n", quotaset.InjectedFileContentBytes)
	t.Logf("    injected_file_path_bytes=[%d]\n", quotaset.InjectedFilePathBytes)
	t.Logf("              injected_files=[%d]\n", quotaset.InjectedFiles)

}

func findTenant(t *testing.T, client *gophercloud.ServiceClient) string {
	var tenantID string
	err := tenants.List(client, nil).EachPage(func(page pagination.Page) (bool, error) {
		tenantList, err := tenants.ExtractTenants(page)
		th.AssertNoErr(t, err)

		for _, t := range tenantList {
			tenantID = t.ID
			break
		}

		return true, nil
	})
	th.AssertNoErr(t, err)

	return tenantID
}
