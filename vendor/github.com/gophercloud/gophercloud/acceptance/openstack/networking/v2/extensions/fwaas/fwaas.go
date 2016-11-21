package fwaas

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/fwaas/firewalls"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/fwaas/policies"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/fwaas/rules"
)

// CreateFirewall will create a Firewaill with a random name and a specified
// policy ID. An error will be returned if the firewall could not be created.
func CreateFirewall(t *testing.T, client *gophercloud.ServiceClient, policyID string) (*firewalls.Firewall, error) {
	firewallName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create firewall %s", firewallName)

	createOpts := firewalls.CreateOpts{
		Name:     firewallName,
		PolicyID: policyID,
	}

	firewall, err := firewalls.Create(client, createOpts).Extract()
	if err != nil {
		return firewall, err
	}

	t.Logf("Waiting for firewall to become active.")
	if err := WaitForFirewallState(client, firewall.ID, "ACTIVE", 60); err != nil {
		return firewall, err
	}

	t.Logf("Successfully created firewall %s", firewallName)

	return firewall, nil
}

// CreatePolicy will create a Firewall Policy with a random name and given
// rule. An error will be returned if the rule could not be created.
func CreatePolicy(t *testing.T, client *gophercloud.ServiceClient, ruleID string) (*policies.Policy, error) {
	policyName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create policy %s", policyName)

	createOpts := policies.CreateOpts{
		Name: policyName,
		Rules: []string{
			ruleID,
		},
	}

	policy, err := policies.Create(client, createOpts).Extract()
	if err != nil {
		return policy, err
	}

	t.Logf("Successfully created policy %s", policyName)

	return policy, nil
}

// CreateRule will create a Firewall Rule with a random source address and
//source port, destination address and port. An error will be returned if
// the rule could not be created.
func CreateRule(t *testing.T, client *gophercloud.ServiceClient) (*rules.Rule, error) {
	ruleName := tools.RandomString("TESTACC-", 8)
	sourceAddress := fmt.Sprintf("192.168.1.%d", tools.RandomInt(1, 100))
	sourcePort := strconv.Itoa(tools.RandomInt(1, 100))
	destinationAddress := fmt.Sprintf("192.168.2.%d", tools.RandomInt(1, 100))
	destinationPort := strconv.Itoa(tools.RandomInt(1, 100))

	t.Logf("Attempting to create rule %s with source %s:%s and destination %s:%s",
		ruleName, sourceAddress, sourcePort, destinationAddress, destinationPort)

	createOpts := rules.CreateOpts{
		Name:                 ruleName,
		Protocol:             rules.ProtocolTCP,
		Action:               "allow",
		SourceIPAddress:      sourceAddress,
		SourcePort:           sourcePort,
		DestinationIPAddress: destinationAddress,
		DestinationPort:      destinationPort,
	}

	rule, err := rules.Create(client, createOpts).Extract()
	if err != nil {
		return rule, err
	}

	t.Logf("Rule %s successfully created", ruleName)

	return rule, nil
}

// DeleteFirewall will delete a firewall with a specified ID. A fatal error
// will occur if the delete was not successful. This works best when used as
// a deferred function.
func DeleteFirewall(t *testing.T, client *gophercloud.ServiceClient, firewallID string) {
	t.Logf("Attempting to delete firewall: %s", firewallID)

	err := firewalls.Delete(client, firewallID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete firewall %s: %v", firewallID, err)
	}

	t.Logf("Waiting for firewall to delete.")
	if err := WaitForFirewallState(client, firewallID, "DELETED", 60); err != nil {
		t.Logf("Unable to delete firewall: %s", firewallID)
	}

	t.Logf("Firewall deleted: %s", firewallID)
}

// DeletePolicy will delete a policy with a specified ID. A fatal error will
// occur if the delete was not successful. This works best when used as a
// deferred function.
func DeletePolicy(t *testing.T, client *gophercloud.ServiceClient, policyID string) {
	t.Logf("Attempting to delete policy: %s", policyID)

	err := policies.Delete(client, policyID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete policy %s: %v", policyID, err)
	}

	t.Logf("Deleted policy: %s", policyID)
}

// DeleteRule will delete a rule with a specified ID. A fatal error will occur
// if the delete was not successful. This works best when used as a deferred
// function.
func DeleteRule(t *testing.T, client *gophercloud.ServiceClient, ruleID string) {
	t.Logf("Attempting to delete rule: %s", ruleID)

	err := rules.Delete(client, ruleID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete rule %s: %v", ruleID, err)
	}

	t.Logf("Deleted rule: %s", ruleID)
}

// PrintFirewall will print a firewall and all of its attributes.
func PrintFirewall(t *testing.T, firewall *firewalls.Firewall) {
	t.Logf("ID: %s", firewall.ID)
	t.Logf("Name: %s", firewall.Name)
	t.Logf("Description: %s", firewall.Description)
	t.Logf("AdminStateUp: %t", firewall.AdminStateUp)
	t.Logf("Status: %s", firewall.Status)
	t.Logf("PolicyID: %s", firewall.PolicyID)
	t.Logf("TenantID: %s", firewall.TenantID)
}

// PrintPolicy will print a policy and all of its attributes.
func PrintPolicy(t *testing.T, policy *policies.Policy) {
	t.Logf("ID: %s", policy.ID)
	t.Logf("Name: %s", policy.Name)
	t.Logf("Description: %s", policy.Description)
	t.Logf("TenantID: %s", policy.TenantID)
	t.Logf("Audited: %t", policy.Audited)
	t.Logf("Shared: %t", policy.Shared)
	t.Logf("Rules:")

	for _, rule := range policy.Rules {
		t.Logf("Rule ID: %s", rule)
	}
}

// PrintRule will print a rule and all of its attributes.
func PrintRule(t *testing.T, rule *rules.Rule) {
	t.Logf("ID: %s", rule.ID)
	t.Logf("Name: %s", rule.Name)
	t.Logf("Description: %s", rule.Description)
	t.Logf("Protocol: %s", rule.Protocol)
	t.Logf("Action: %s", rule.Action)
	t.Logf("IPVersion: %d", rule.IPVersion)
	t.Logf("SourceIPAddress: %s", rule.SourceIPAddress)
	t.Logf("DestinationIPAddress: %s", rule.DestinationIPAddress)
	t.Logf("Shared: %t", rule.Shared)
	t.Logf("Enabled: %t", rule.Enabled)
	t.Logf("PolicyID: %s", rule.PolicyID)
	t.Logf("Position: %d", rule.Position)
	t.Logf("TenantID: %s", rule.TenantID)
}

// WaitForFirewallState will wait until a firewall reaches a given state.
func WaitForFirewallState(client *gophercloud.ServiceClient, firewallID, status string, secs int) error {
	return gophercloud.WaitFor(secs, func() (bool, error) {
		current, err := firewalls.Get(client, firewallID).Extract()
		if err != nil {
			if httpStatus, ok := err.(gophercloud.ErrDefault404); ok {
				if httpStatus.Actual == 404 {
					if status == "DELETED" {
						return true, nil
					}
				}
			}
			return false, err
		}

		if current.Status == status {
			return true, nil
		}

		return false, nil
	})
}
