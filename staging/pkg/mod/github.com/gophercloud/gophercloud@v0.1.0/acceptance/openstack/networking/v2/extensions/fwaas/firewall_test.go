// +build acceptance networking fwaas

package fwaas

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	layer3 "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2/extensions/layer3"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/fwaas/firewalls"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/fwaas/routerinsertion"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestFirewallCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	router, err := layer3.CreateExternalRouter(t, client)
	th.AssertNoErr(t, err)
	defer layer3.DeleteRouter(t, client, router.ID)

	rule, err := CreateRule(t, client)
	th.AssertNoErr(t, err)
	defer DeleteRule(t, client, rule.ID)

	tools.PrintResource(t, rule)

	policy, err := CreatePolicy(t, client, rule.ID)
	th.AssertNoErr(t, err)
	defer DeletePolicy(t, client, policy.ID)

	tools.PrintResource(t, policy)

	firewall, err := CreateFirewall(t, client, policy.ID)
	th.AssertNoErr(t, err)
	defer DeleteFirewall(t, client, firewall.ID)

	tools.PrintResource(t, firewall)

	fwName := ""
	fwDescription := ""
	fwUpdateOpts := firewalls.UpdateOpts{
		Name:        &fwName,
		Description: &fwDescription,
		PolicyID:    policy.ID,
	}

	_, err = firewalls.Update(client, firewall.ID, fwUpdateOpts).Extract()
	th.AssertNoErr(t, err)

	newFirewall, err := firewalls.Get(client, firewall.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newFirewall)
	th.AssertEquals(t, newFirewall.Name, fwName)
	th.AssertEquals(t, newFirewall.Description, fwDescription)
	th.AssertEquals(t, newFirewall.PolicyID, policy.ID)

	allPages, err := firewalls.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allFirewalls, err := firewalls.ExtractFirewalls(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, firewall := range allFirewalls {
		if firewall.ID == newFirewall.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}

func TestFirewallCRUDRouter(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	router, err := layer3.CreateExternalRouter(t, client)
	th.AssertNoErr(t, err)
	defer layer3.DeleteRouter(t, client, router.ID)

	rule, err := CreateRule(t, client)
	th.AssertNoErr(t, err)
	defer DeleteRule(t, client, rule.ID)

	tools.PrintResource(t, rule)

	policy, err := CreatePolicy(t, client, rule.ID)
	th.AssertNoErr(t, err)
	defer DeletePolicy(t, client, policy.ID)

	tools.PrintResource(t, policy)

	firewall, err := CreateFirewallOnRouter(t, client, policy.ID, router.ID)
	th.AssertNoErr(t, err)
	defer DeleteFirewall(t, client, firewall.ID)

	tools.PrintResource(t, firewall)

	router2, err := layer3.CreateExternalRouter(t, client)
	th.AssertNoErr(t, err)
	defer layer3.DeleteRouter(t, client, router2.ID)

	description := "Some firewall description"
	firewallUpdateOpts := firewalls.UpdateOpts{
		PolicyID:    policy.ID,
		Description: &description,
	}

	updateOpts := routerinsertion.UpdateOptsExt{
		firewallUpdateOpts,
		[]string{router2.ID},
	}

	_, err = firewalls.Update(client, firewall.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	newFirewall, err := firewalls.Get(client, firewall.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newFirewall)
}

func TestFirewallCRUDRemoveRouter(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	router, err := layer3.CreateExternalRouter(t, client)
	th.AssertNoErr(t, err)
	defer layer3.DeleteRouter(t, client, router.ID)

	rule, err := CreateRule(t, client)
	th.AssertNoErr(t, err)
	defer DeleteRule(t, client, rule.ID)

	tools.PrintResource(t, rule)

	policy, err := CreatePolicy(t, client, rule.ID)
	th.AssertNoErr(t, err)
	defer DeletePolicy(t, client, policy.ID)

	tools.PrintResource(t, policy)

	firewall, err := CreateFirewallOnRouter(t, client, policy.ID, router.ID)
	th.AssertNoErr(t, err)
	defer DeleteFirewall(t, client, firewall.ID)

	tools.PrintResource(t, firewall)

	description := "Some firewall description"
	firewallUpdateOpts := firewalls.UpdateOpts{
		PolicyID:    policy.ID,
		Description: &description,
	}

	updateOpts := routerinsertion.UpdateOptsExt{
		firewallUpdateOpts,
		[]string{},
	}

	_, err = firewalls.Update(client, firewall.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	newFirewall, err := firewalls.Get(client, firewall.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newFirewall)
}
