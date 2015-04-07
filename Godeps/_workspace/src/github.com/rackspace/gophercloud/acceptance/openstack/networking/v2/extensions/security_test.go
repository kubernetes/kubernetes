// +build acceptance networking security

package extensions

import (
	"testing"

	base "github.com/rackspace/gophercloud/acceptance/openstack/networking/v2"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/groups"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/rules"
	"github.com/rackspace/gophercloud/openstack/networking/v2/networks"
	"github.com/rackspace/gophercloud/openstack/networking/v2/ports"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestSecurityGroups(t *testing.T) {
	base.Setup(t)
	defer base.Teardown()

	// create security group
	groupID := createSecGroup(t)

	// delete security group
	defer deleteSecGroup(t, groupID)

	// list security group
	listSecGroups(t)

	// get security group
	getSecGroup(t, groupID)

	// create port with security group
	networkID, portID := createPort(t, groupID)

	// teardown
	defer deleteNetwork(t, networkID)

	// delete port
	defer deletePort(t, portID)
}

func TestSecurityGroupRules(t *testing.T) {
	base.Setup(t)
	defer base.Teardown()

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
	sg, err := groups.Create(base.Client, groups.CreateOpts{
		Name:        "new-webservers",
		Description: "security group for webservers",
	}).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Created security group %s", sg.ID)

	return sg.ID
}

func listSecGroups(t *testing.T) {
	err := groups.List(base.Client, groups.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		list, err := groups.ExtractGroups(page)
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
	sg, err := groups.Get(base.Client, id).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Getting security group: ID [%s] Name [%s] Description [%s]", sg.ID, sg.Name, sg.Description)
}

func createPort(t *testing.T, groupID string) (string, string) {
	n, err := networks.Create(base.Client, networks.CreateOpts{Name: "tmp_network"}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created network %s", n.ID)

	opts := ports.CreateOpts{
		NetworkID:      n.ID,
		Name:           "my_port",
		SecurityGroups: []string{groupID},
	}
	p, err := ports.Create(base.Client, opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created port %s with security group %s", p.ID, groupID)

	return n.ID, p.ID
}

func deleteSecGroup(t *testing.T, groupID string) {
	res := groups.Delete(base.Client, groupID)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted security group %s", groupID)
}

func createSecRule(t *testing.T, groupID string) string {
	r, err := rules.Create(base.Client, rules.CreateOpts{
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
	err := rules.List(base.Client, rules.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		list, err := rules.ExtractRules(page)
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
	r, err := rules.Get(base.Client, id).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Getting security rule: ID [%s] Direction [%s] EtherType [%s] Protocol [%s]",
		r.ID, r.Direction, r.EtherType, r.Protocol)
}

func deleteSecRule(t *testing.T, id string) {
	res := rules.Delete(base.Client, id)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted security rule %s", id)
}
