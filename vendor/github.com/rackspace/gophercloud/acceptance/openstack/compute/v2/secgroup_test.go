// +build acceptance compute secgroups

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/secgroups"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestSecGroups(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	serverID, needsDeletion := findServer(t, client)

	groupID := createSecGroup(t, client)

	listSecGroups(t, client)

	newName := tools.RandomString("secgroup_", 5)
	updateSecGroup(t, client, groupID, newName)

	getSecGroup(t, client, groupID)

	addRemoveRules(t, client, groupID)

	addServerToSecGroup(t, client, serverID, newName)

	removeServerFromSecGroup(t, client, serverID, newName)

	if needsDeletion {
		servers.Delete(client, serverID)
	}

	deleteSecGroup(t, client, groupID)
}

func createSecGroup(t *testing.T, client *gophercloud.ServiceClient) string {
	opts := secgroups.CreateOpts{
		Name:        tools.RandomString("secgroup_", 5),
		Description: "something",
	}

	group, err := secgroups.Create(client, opts).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Created secgroup %s %s", group.ID, group.Name)

	return group.ID
}

func listSecGroups(t *testing.T, client *gophercloud.ServiceClient) {
	err := secgroups.List(client).EachPage(func(page pagination.Page) (bool, error) {
		secGrpList, err := secgroups.ExtractSecurityGroups(page)
		th.AssertNoErr(t, err)

		for _, sg := range secGrpList {
			t.Logf("Listing secgroup %s: Name [%s] Desc [%s] TenantID [%s]", sg.ID,
				sg.Name, sg.Description, sg.TenantID)
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
}

func updateSecGroup(t *testing.T, client *gophercloud.ServiceClient, id, newName string) {
	opts := secgroups.UpdateOpts{
		Name:        newName,
		Description: tools.RandomString("dec_", 10),
	}
	group, err := secgroups.Update(client, id, opts).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Updated %s's name to %s", group.ID, group.Name)
}

func getSecGroup(t *testing.T, client *gophercloud.ServiceClient, id string) {
	group, err := secgroups.Get(client, id).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Getting %s: %#v", id, group)
}

func addRemoveRules(t *testing.T, client *gophercloud.ServiceClient, id string) {
	opts := secgroups.CreateRuleOpts{
		ParentGroupID: id,
		FromPort:      22,
		ToPort:        22,
		IPProtocol:    "TCP",
		CIDR:          "0.0.0.0/0",
	}

	rule, err := secgroups.CreateRule(client, opts).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Adding rule %s to group %s", rule.ID, id)

	err = secgroups.DeleteRule(client, rule.ID).ExtractErr()
	th.AssertNoErr(t, err)

	t.Logf("Deleted rule %s from group %s", rule.ID, id)

	icmpOpts := secgroups.CreateRuleOpts{
		ParentGroupID: id,
		FromPort:      0,
		ToPort:        0,
		IPProtocol:    "ICMP",
		CIDR:          "0.0.0.0/0",
	}

	icmpRule, err := secgroups.CreateRule(client, icmpOpts).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Adding ICMP rule %s to group %s", icmpRule.ID, id)

	err = secgroups.DeleteRule(client, icmpRule.ID).ExtractErr()
	th.AssertNoErr(t, err)

	t.Logf("Deleted ICMP rule %s from group %s", icmpRule.ID, id)
}

func findServer(t *testing.T, client *gophercloud.ServiceClient) (string, bool) {
	var serverID string
	var needsDeletion bool

	err := servers.List(client, nil).EachPage(func(page pagination.Page) (bool, error) {
		sList, err := servers.ExtractServers(page)
		th.AssertNoErr(t, err)

		for _, s := range sList {
			serverID = s.ID
			needsDeletion = false

			t.Logf("Found an existing server: ID [%s]", serverID)
			break
		}

		return true, nil
	})
	th.AssertNoErr(t, err)

	if serverID == "" {
		t.Log("No server found, creating one")

		choices, err := ComputeChoicesFromEnv()
		th.AssertNoErr(t, err)

		opts := &servers.CreateOpts{
			Name:      tools.RandomString("secgroup_test_", 5),
			ImageRef:  choices.ImageID,
			FlavorRef: choices.FlavorID,
		}

		s, err := servers.Create(client, opts).Extract()
		th.AssertNoErr(t, err)
		serverID = s.ID

		t.Logf("Created server %s, waiting for it to build", s.ID)
		err = servers.WaitForStatus(client, serverID, "ACTIVE", 300)
		th.AssertNoErr(t, err)

		needsDeletion = true
	}

	return serverID, needsDeletion
}

func addServerToSecGroup(t *testing.T, client *gophercloud.ServiceClient, serverID, groupName string) {
	err := secgroups.AddServerToGroup(client, serverID, groupName).ExtractErr()
	th.AssertNoErr(t, err)

	t.Logf("Adding group %s to server %s", groupName, serverID)
}

func removeServerFromSecGroup(t *testing.T, client *gophercloud.ServiceClient, serverID, groupName string) {
	err := secgroups.RemoveServerFromGroup(client, serverID, groupName).ExtractErr()
	th.AssertNoErr(t, err)

	t.Logf("Removing group %s from server %s", groupName, serverID)
}

func deleteSecGroup(t *testing.T, client *gophercloud.ServiceClient, id string) {
	err := secgroups.Delete(client, id).ExtractErr()
	th.AssertNoErr(t, err)

	t.Logf("Deleted group %s", id)
}
