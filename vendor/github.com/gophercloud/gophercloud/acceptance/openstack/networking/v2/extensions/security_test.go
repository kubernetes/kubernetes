// +build acceptance networking security

package extensions

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/security/groups"
)

func TestSecurityGroupsList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	listOpts := groups.ListOpts{}
	allPages, err := groups.List(client, listOpts).AllPages()
	if err != nil {
		t.Fatalf("Unable to list groups: %v", err)
	}

	allGroups, err := groups.ExtractGroups(allPages)
	if err != nil {
		t.Fatalf("Unable to extract groups: %v", err)
	}

	for _, group := range allGroups {
		tools.PrintResource(t, group)
	}
}

func TestSecurityGroupsCreateUpdateDelete(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	group, err := CreateSecurityGroup(t, client)
	if err != nil {
		t.Fatalf("Unable to create security group: %v", err)
	}
	defer DeleteSecurityGroup(t, client, group.ID)

	rule, err := CreateSecurityGroupRule(t, client, group.ID)
	if err != nil {
		t.Fatalf("Unable to create security group rule: %v", err)
	}
	defer DeleteSecurityGroupRule(t, client, rule.ID)

	tools.PrintResource(t, group)

	updateOpts := groups.UpdateOpts{
		Description: "A security group",
	}

	newGroup, err := groups.Update(client, group.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update security group: %v", err)
	}

	tools.PrintResource(t, newGroup)
}

func TestSecurityGroupsPort(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	network, err := networking.CreateNetwork(t, client)
	if err != nil {
		t.Fatalf("Unable to create network: %v", err)
	}
	defer networking.DeleteNetwork(t, client, network.ID)

	subnet, err := networking.CreateSubnet(t, client, network.ID)
	if err != nil {
		t.Fatalf("Unable to create subnet: %v", err)
	}
	defer networking.DeleteSubnet(t, client, subnet.ID)

	group, err := CreateSecurityGroup(t, client)
	if err != nil {
		t.Fatalf("Unable to create security group: %v", err)
	}
	defer DeleteSecurityGroup(t, client, group.ID)

	rule, err := CreateSecurityGroupRule(t, client, group.ID)
	if err != nil {
		t.Fatalf("Unable to create security group rule: %v", err)
	}
	defer DeleteSecurityGroupRule(t, client, rule.ID)

	port, err := CreatePortWithSecurityGroup(t, client, network.ID, subnet.ID, group.ID)
	if err != nil {
		t.Fatalf("Unable to create port: %v", err)
	}
	defer networking.DeletePort(t, client, port.ID)

	tools.PrintResource(t, port)
}
