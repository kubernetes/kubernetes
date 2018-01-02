package extensions

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/external"
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

	networkCreateOpts := networks.CreateOpts{
		Name:         networkName,
		AdminStateUp: &adminStateUp,
	}

	createOpts := external.CreateOptsExt{
		CreateOptsBuilder: networkCreateOpts,
		External:          &isExternal,
	}

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
		SecurityGroups: &[]string{secGroupID},
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
