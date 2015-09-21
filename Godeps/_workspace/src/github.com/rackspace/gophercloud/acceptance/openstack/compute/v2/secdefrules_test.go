// +build acceptance compute defsecrules

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	dsr "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/defsecrules"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestSecDefRules(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	id := createDefRule(t, client)

	listDefRules(t, client)

	getDefRule(t, client, id)

	deleteDefRule(t, client, id)
}

func createDefRule(t *testing.T, client *gophercloud.ServiceClient) string {
	opts := dsr.CreateOpts{
		FromPort:   tools.RandomInt(80, 89),
		ToPort:     tools.RandomInt(90, 99),
		IPProtocol: "TCP",
		CIDR:       "0.0.0.0/0",
	}

	rule, err := dsr.Create(client, opts).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Created default rule %s", rule.ID)

	return rule.ID
}

func listDefRules(t *testing.T, client *gophercloud.ServiceClient) {
	err := dsr.List(client).EachPage(func(page pagination.Page) (bool, error) {
		drList, err := dsr.ExtractDefaultRules(page)
		th.AssertNoErr(t, err)

		for _, dr := range drList {
			t.Logf("Listing default rule %s: Name [%s] From Port [%s] To Port [%s] Protocol [%s]",
				dr.ID, dr.FromPort, dr.ToPort, dr.IPProtocol)
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
}

func getDefRule(t *testing.T, client *gophercloud.ServiceClient, id string) {
	rule, err := dsr.Get(client, id).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Getting rule %s: %#v", id, rule)
}

func deleteDefRule(t *testing.T, client *gophercloud.ServiceClient, id string) {
	err := dsr.Delete(client, id).ExtractErr()
	th.AssertNoErr(t, err)

	t.Logf("Deleted rule %s", id)
}
