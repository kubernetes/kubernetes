package extensions

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/external"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/provider"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/security/groups"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/security/rules"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
)

// CreateExternalNetwork will create an external network. An error will be
// returned if the creation failed.
func CreateExternalNetwork(t *testing.T, client *gophercloud.ServiceClient) (*networks.Network, error) {
	networkName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create external network: %s", networkName)

	adminStateUp := true
	isExternal := true
	createOpts := external.CreateOpts{
		External: &isExternal,
	}

	createOpts.Name = networkName
	createOpts.AdminStateUp = &adminStateUp

	network, err := networks.Create(client, createOpts).Extract()
	if err != nil {
		return network, err
	}

	t.Logf("Created external network: %s", networkName)

	return network, nil
}

// CreatePortWithSecurityGroup will create a port with a security group
// attached. An error will be returned if the port could not be created.
func CreatePortWithSecurityGroup(t *testing.T, client *gophercloud.ServiceClient, networkID, subnetID, secGroupID string) (*ports.Port, error) {
	portName := tools.RandomString("TESTACC-", 8)
	iFalse := false

	t.Logf("Attempting to create port: %s", portName)

	createOpts := ports.CreateOpts{
		NetworkID:      networkID,
		Name:           portName,
		AdminStateUp:   &iFalse,
		FixedIPs:       []ports.IP{ports.IP{SubnetID: subnetID}},
		SecurityGroups: []string{secGroupID},
	}

	port, err := ports.Create(client, createOpts).Extract()
	if err != nil {
		return port, err
	}

	t.Logf("Successfully created port: %s", portName)

	return port, nil
}

// CreateSecurityGroup will create a security group with a random name.
// An error will be returned if one was failed to be created.
func CreateSecurityGroup(t *testing.T, client *gophercloud.ServiceClient) (*groups.SecGroup, error) {
	secGroupName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create security group: %s", secGroupName)

	createOpts := groups.CreateOpts{
		Name: secGroupName,
	}

	secGroup, err := groups.Create(client, createOpts).Extract()
	if err != nil {
		return secGroup, err
	}

	t.Logf("Created security group: %s", secGroup.ID)

	return secGroup, nil
}

// CreateSecurityGroupRule will create a security group rule with a random name
// and random port between 80 and 99.
// An error will be returned if one was failed to be created.
func CreateSecurityGroupRule(t *testing.T, client *gophercloud.ServiceClient, secGroupID string) (*rules.SecGroupRule, error) {
	t.Logf("Attempting to create security group rule in group: %s", secGroupID)

	fromPort := tools.RandomInt(80, 89)
	toPort := tools.RandomInt(90, 99)

	createOpts := rules.CreateOpts{
		Direction:    "ingress",
		EtherType:    "IPv4",
		SecGroupID:   secGroupID,
		PortRangeMin: fromPort,
		PortRangeMax: toPort,
		Protocol:     rules.ProtocolTCP,
	}

	rule, err := rules.Create(client, createOpts).Extract()
	if err != nil {
		return rule, err
	}

	t.Logf("Created security group rule: %s", rule.ID)

	return rule, nil
}

// DeleteSecurityGroup will delete a security group of a specified ID.
// A fatal error will occur if the deletion failed. This works best as a
// deferred function
func DeleteSecurityGroup(t *testing.T, client *gophercloud.ServiceClient, secGroupID string) {
	t.Logf("Attempting to delete security group: %s", secGroupID)

	err := groups.Delete(client, secGroupID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete security group: %v", err)
	}
}

// DeleteSecurityGroupRule will delete a security group rule of a specified ID.
// A fatal error will occur if the deletion failed. This works best as a
// deferred function
func DeleteSecurityGroupRule(t *testing.T, client *gophercloud.ServiceClient, ruleID string) {
	t.Logf("Attempting to delete security group rule: %s", ruleID)

	err := rules.Delete(client, ruleID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete security group rule: %v", err)
	}
}

// PrintNetworkExtAttrs prints a network and all of its extra attributes.
func PrintNetworkExtAttrs(t *testing.T, network *provider.NetworkExtAttrs) {
	t.Logf("ID: %s", network.ID)
	t.Logf("Name: %s", network.Name)
	t.Logf("AdminStateUp: %t", network.AdminStateUp)
	t.Logf("Status: %s", network.Status)
	t.Logf("Subnets: %s", network.Subnets)
	t.Logf("TenantID: %s", network.TenantID)
	t.Logf("Shared: %t", network.Shared)
	t.Logf("NetworkType: %s", network.NetworkType)
	t.Logf("PhysicalNetwork: %s", network.PhysicalNetwork)
	t.Logf("SegmentationID: %d", network.SegmentationID)
}

// PrintSecurityGroup will print a security group and all of its attributes.
func PrintSecurityGroup(t *testing.T, secGroup *groups.SecGroup) {
	t.Logf("ID: %s", secGroup.ID)
	t.Logf("Name: %s", secGroup.Name)
	t.Logf("Description: %s", secGroup.Description)
	t.Logf("TenantID: %s", secGroup.TenantID)
	t.Logf("Rules:")

	for _, rule := range secGroup.Rules {
		PrintSecurityGroupRule(t, &rule)
	}
}

// PrintSecurityGroupRule will print a security group rule and all of its attributes.
func PrintSecurityGroupRule(t *testing.T, rule *rules.SecGroupRule) {
	t.Logf("ID: %s", rule.ID)
	t.Logf("Direction: %s", rule.Direction)
	t.Logf("EtherType: %s", rule.EtherType)
	t.Logf("SecGroupID: %s", rule.SecGroupID)
	t.Logf("PortRangeMin: %d", rule.PortRangeMin)
	t.Logf("PortRangeMax: %d", rule.PortRangeMax)
	t.Logf("Protocol: %s", rule.Protocol)
	t.Logf("RemoteGroupID: %s", rule.RemoteGroupID)
	t.Logf("RemoteIPPrefix: %s", rule.RemoteIPPrefix)
	t.Logf("TenantID: %s", rule.TenantID)
}
