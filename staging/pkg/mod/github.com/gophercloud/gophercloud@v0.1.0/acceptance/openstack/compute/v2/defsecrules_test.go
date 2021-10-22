// +build acceptance compute defsecrules

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	dsr "github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/defsecrules"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestDefSecRulesList(t *testing.T) {
	clients.RequireAdmin(t)
	clients.RequireNovaNetwork(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	allPages, err := dsr.List(client).AllPages()
	th.AssertNoErr(t, err)

	allDefaultRules, err := dsr.ExtractDefaultRules(allPages)
	th.AssertNoErr(t, err)

	for _, defaultRule := range allDefaultRules {
		tools.PrintResource(t, defaultRule)
	}
}

func TestDefSecRulesCreate(t *testing.T) {
	clients.RequireAdmin(t)
	clients.RequireNovaNetwork(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	defaultRule, err := CreateDefaultRule(t, client)
	th.AssertNoErr(t, err)
	defer DeleteDefaultRule(t, client, defaultRule)

	tools.PrintResource(t, defaultRule)
}

func TestDefSecRulesGet(t *testing.T) {
	clients.RequireAdmin(t)
	clients.RequireNovaNetwork(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	defaultRule, err := CreateDefaultRule(t, client)
	th.AssertNoErr(t, err)
	defer DeleteDefaultRule(t, client, defaultRule)

	newDefaultRule, err := dsr.Get(client, defaultRule.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newDefaultRule)
}
