// +build acceptance networking security

package extensions

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/security/groups"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestSecurityGroupsCreateUpdateDelete(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	group, err := CreateSecurityGroup(t, client)
	th.AssertNoErr(t, err)
	defer DeleteSecurityGroup(t, client, group.ID)

	rule, err := CreateSecurityGroupRule(t, client, group.ID)
	th.AssertNoErr(t, err)
	defer DeleteSecurityGroupRule(t, client, rule.ID)

	tools.PrintResource(t, group)

	var name = "Update group"
	var description = ""
	updateOpts := groups.UpdateOpts{
		Name:        name,
		Description: &description,
	}

	newGroup, err := groups.Update(client, group.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newGroup)
	th.AssertEquals(t, newGroup.Name, name)
	th.AssertEquals(t, newGroup.Description, description)

	listOpts := groups.ListOpts{}
	allPages, err := groups.List(client, listOpts).AllPages()
	th.AssertNoErr(t, err)

	allGroups, err := groups.ExtractGroups(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, group := range allGroups {
		if group.ID == newGroup.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}

func TestSecurityGroupsPort(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	network, err := networking.CreateNetwork(t, client)
	th.AssertNoErr(t, err)
	defer networking.DeleteNetwork(t, client, network.ID)

	subnet, err := networking.CreateSubnet(t, client, network.ID)
	th.AssertNoErr(t, err)
	defer networking.DeleteSubnet(t, client, subnet.ID)

	group, err := CreateSecurityGroup(t, client)
	th.AssertNoErr(t, err)
	defer DeleteSecurityGroup(t, client, group.ID)

	rule, err := CreateSecurityGroupRule(t, client, group.ID)
	th.AssertNoErr(t, err)
	defer DeleteSecurityGroupRule(t, client, rule.ID)

	port, err := CreatePortWithSecurityGroup(t, client, network.ID, subnet.ID, group.ID)
	th.AssertNoErr(t, err)
	defer networking.DeletePort(t, client, port.ID)

	tools.PrintResource(t, port)
}
