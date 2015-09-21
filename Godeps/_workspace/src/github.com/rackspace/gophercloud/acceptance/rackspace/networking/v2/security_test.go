// +build acceptance networking security

package v2

import (
	"testing"

	osGroups "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/groups"
	osRules "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/rules"
	osNetworks "github.com/rackspace/gophercloud/openstack/networking/v2/networks"
	osPorts "github.com/rackspace/gophercloud/openstack/networking/v2/ports"
	"github.com/rackspace/gophercloud/pagination"
	rsNetworks "github.com/rackspace/gophercloud/rackspace/networking/v2/networks"
	rsPorts "github.com/rackspace/gophercloud/rackspace/networking/v2/ports"
	rsGroups "github.com/rackspace/gophercloud/rackspace/networking/v2/security/groups"
	rsRules "github.com/rackspace/gophercloud/rackspace/networking/v2/security/rules"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestSecurityGroups(t *testing.T) {
	Setup(t)
	defer Teardown()

	// create security group
	groupID := createSecGroup(t)

	// delete security group
	defer deleteSecGroup(t, groupID)

	// list security group
	listSecGroups(t)

	// get security group
	getSecGroup(t, groupID)
}

func TestSecurityGroupRules(t *testing.T) {
	Setup(t)
	defer Teardown()

	// create security group
	groupID := createSecGroup(t)

	defer deleteSecGroup(t, groupID)

	// create security group rule
	ruleID := createSecRule(t, groupID)

	// delete security group rule
	defer deleteSecRule(t, ruleID)

	// list security group rule
	listSecRules(t)

	// get security group rule
	getSecRule(t, ruleID)
}

func createSecGroup(t *testing.T) string {
	sg, err := rsGroups.Create(Client, osGroups.CreateOpts{
		Name:        "new-webservers",
		Description: "security group for webservers",
	}).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Created security group %s", sg.ID)

	return sg.ID
}

func listSecGroups(t *testing.T) {
	err := rsGroups.List(Client, osGroups.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		list, err := osGroups.ExtractGroups(page)
		if err != nil {
			t.Errorf("Failed to extract secgroups: %v", err)
			return false, err
		}

		for _, sg := range list {
			t.Logf("Listing security group: ID [%s] Name [%s]", sg.ID, sg.Name)
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
}

func getSecGroup(t *testing.T, id string) {
	sg, err := rsGroups.Get(Client, id).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Getting security group: ID [%s] Name [%s] Description [%s]", sg.ID, sg.Name, sg.Description)
}

func createSecGroupPort(t *testing.T, groupID string) (string, string) {
	n, err := rsNetworks.Create(Client, osNetworks.CreateOpts{Name: "tmp_network"}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created network %s", n.ID)

	opts := osPorts.CreateOpts{
		NetworkID:      n.ID,
		Name:           "my_port",
		SecurityGroups: []string{groupID},
	}
	p, err := rsPorts.Create(Client, opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created port %s with security group %s", p.ID, groupID)

	return n.ID, p.ID
}

func deleteSecGroup(t *testing.T, groupID string) {
	res := rsGroups.Delete(Client, groupID)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted security group %s", groupID)
}

func createSecRule(t *testing.T, groupID string) string {
	r, err := rsRules.Create(Client, osRules.CreateOpts{
		Direction:    "ingress",
		PortRangeMin: 80,
		EtherType:    "IPv4",
		PortRangeMax: 80,
		Protocol:     "tcp",
		SecGroupID:   groupID,
	}).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Created security group rule %s", r.ID)

	return r.ID
}

func listSecRules(t *testing.T) {
	err := rsRules.List(Client, osRules.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		list, err := osRules.ExtractRules(page)
		if err != nil {
			t.Errorf("Failed to extract sec rules: %v", err)
			return false, err
		}

		for _, r := range list {
			t.Logf("Listing security rule: ID [%s]", r.ID)
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
}

func getSecRule(t *testing.T, id string) {
	r, err := rsRules.Get(Client, id).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Getting security rule: ID [%s] Direction [%s] EtherType [%s] Protocol [%s]",
		r.ID, r.Direction, r.EtherType, r.Protocol)
}

func deleteSecRule(t *testing.T, id string) {
	res := rsRules.Delete(Client, id)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted security rule %s", id)
}
