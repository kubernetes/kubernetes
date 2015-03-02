// +build acceptance networking fwaas

package fwaas

import (
	"testing"
	"time"

	"github.com/rackspace/gophercloud"
	base "github.com/rackspace/gophercloud/acceptance/openstack/networking/v2"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/fwaas/firewalls"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/fwaas/policies"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func firewallSetup(t *testing.T) string {
	base.Setup(t)
	return createPolicy(t, &policies.CreateOpts{})
}

func firewallTeardown(t *testing.T, policyID string) {
	defer base.Teardown()
	deletePolicy(t, policyID)
}

func TestFirewall(t *testing.T) {
	policyID := firewallSetup(t)
	defer firewallTeardown(t, policyID)

	firewallID := createFirewall(t, &firewalls.CreateOpts{
		Name:        "gophercloud test",
		Description: "acceptance test",
		PolicyID:    policyID,
	})

	waitForFirewallToBeActive(t, firewallID)

	listFirewalls(t)

	updateFirewall(t, firewallID, &firewalls.UpdateOpts{
		Description: "acceptance test updated",
	})

	waitForFirewallToBeActive(t, firewallID)

	deleteFirewall(t, firewallID)

	waitForFirewallToBeDeleted(t, firewallID)
}

func createFirewall(t *testing.T, opts *firewalls.CreateOpts) string {
	f, err := firewalls.Create(base.Client, *opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created firewall: %#v", opts)
	return f.ID
}

func listFirewalls(t *testing.T) {
	err := firewalls.List(base.Client, firewalls.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		firewallList, err := firewalls.ExtractFirewalls(page)
		if err != nil {
			t.Errorf("Failed to extract firewalls: %v", err)
			return false, err
		}

		for _, r := range firewallList {
			t.Logf("Listing firewalls: ID [%s]", r.ID)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)
}

func updateFirewall(t *testing.T, firewallID string, opts *firewalls.UpdateOpts) {
	f, err := firewalls.Update(base.Client, firewallID, *opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Updated firewall ID [%s]", f.ID)
}

func getFirewall(t *testing.T, firewallID string) *firewalls.Firewall {
	f, err := firewalls.Get(base.Client, firewallID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Getting firewall ID [%s]", f.ID)
	return f
}

func deleteFirewall(t *testing.T, firewallID string) {
	res := firewalls.Delete(base.Client, firewallID)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted firewall %s", firewallID)
}

func waitForFirewallToBeActive(t *testing.T, firewallID string) {
	for i := 0; i < 10; i++ {
		fw := getFirewall(t, firewallID)
		if fw.Status == "ACTIVE" {
			break
		}
		time.Sleep(time.Second)
	}
}

func waitForFirewallToBeDeleted(t *testing.T, firewallID string) {
	for i := 0; i < 10; i++ {
		err := firewalls.Get(base.Client, firewallID).Err
		if err != nil {
			httpStatus := err.(*gophercloud.UnexpectedResponseCodeError)
			if httpStatus.Actual == 404 {
				return
			}
		}
		time.Sleep(time.Second)
	}
}
