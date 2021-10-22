package vpnaas

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/vpnaas/endpointgroups"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/vpnaas/ikepolicies"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/vpnaas/ipsecpolicies"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/vpnaas/services"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/vpnaas/siteconnections"
	th "github.com/gophercloud/gophercloud/testhelper"
)

// CreateService will create a Service with a random name and a specified router ID
// An error will be returned if the service could not be created.
func CreateService(t *testing.T, client *gophercloud.ServiceClient, routerID string) (*services.Service, error) {
	serviceName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create service %s", serviceName)

	iTrue := true
	createOpts := services.CreateOpts{
		Name:         serviceName,
		AdminStateUp: &iTrue,
		RouterID:     routerID,
	}
	service, err := services.Create(client, createOpts).Extract()
	if err != nil {
		return service, err
	}

	t.Logf("Successfully created service %s", serviceName)

	th.AssertEquals(t, service.Name, serviceName)

	return service, nil
}

// DeleteService will delete a service with a specified ID. A fatal error
// will occur if the delete was not successful. This works best when used as
// a deferred function.
func DeleteService(t *testing.T, client *gophercloud.ServiceClient, serviceID string) {
	t.Logf("Attempting to delete service: %s", serviceID)

	err := services.Delete(client, serviceID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete service %s: %v", serviceID, err)
	}

	t.Logf("Service deleted: %s", serviceID)
}

// CreateIPSecPolicy will create an IPSec Policy with a random name and given
// rule. An error will be returned if the rule could not be created.
func CreateIPSecPolicy(t *testing.T, client *gophercloud.ServiceClient) (*ipsecpolicies.Policy, error) {
	policyName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create IPSec policy %s", policyName)

	createOpts := ipsecpolicies.CreateOpts{
		Name: policyName,
	}

	policy, err := ipsecpolicies.Create(client, createOpts).Extract()
	if err != nil {
		return policy, err
	}

	t.Logf("Successfully created IPSec policy %s", policyName)

	th.AssertEquals(t, policy.Name, policyName)

	return policy, nil
}

// CreateIKEPolicy will create an IKE Policy with a random name and given
// rule. An error will be returned if the policy could not be created.
func CreateIKEPolicy(t *testing.T, client *gophercloud.ServiceClient) (*ikepolicies.Policy, error) {
	policyName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create IKE policy %s", policyName)

	createOpts := ikepolicies.CreateOpts{
		Name:                policyName,
		EncryptionAlgorithm: ikepolicies.EncryptionAlgorithm3DES,
		PFS:                 ikepolicies.PFSGroup5,
	}

	policy, err := ikepolicies.Create(client, createOpts).Extract()
	if err != nil {
		return policy, err
	}

	t.Logf("Successfully created IKE policy %s", policyName)

	th.AssertEquals(t, policy.Name, policyName)

	return policy, nil
}

// DeleteIPSecPolicy will delete an IPSec policy with a specified ID. A fatal error will
// occur if the delete was not successful. This works best when used as a
// deferred function.
func DeleteIPSecPolicy(t *testing.T, client *gophercloud.ServiceClient, policyID string) {
	t.Logf("Attempting to delete IPSec policy: %s", policyID)

	err := ipsecpolicies.Delete(client, policyID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete IPSec policy %s: %v", policyID, err)
	}

	t.Logf("Deleted IPSec policy: %s", policyID)
}

// DeleteIKEPolicy will delete an IKE policy with a specified ID. A fatal error will
// occur if the delete was not successful. This works best when used as a
// deferred function.
func DeleteIKEPolicy(t *testing.T, client *gophercloud.ServiceClient, policyID string) {
	t.Logf("Attempting to delete policy: %s", policyID)

	err := ikepolicies.Delete(client, policyID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete IKE policy %s: %v", policyID, err)
	}

	t.Logf("Deleted IKE policy: %s", policyID)
}

// CreateEndpointGroup will create an endpoint group with a random name.
// An error will be returned if the group could not be created.
func CreateEndpointGroup(t *testing.T, client *gophercloud.ServiceClient) (*endpointgroups.EndpointGroup, error) {
	groupName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create group %s", groupName)

	createOpts := endpointgroups.CreateOpts{
		Name: groupName,
		Type: endpointgroups.TypeCIDR,
		Endpoints: []string{
			"10.2.0.0/24",
			"10.3.0.0/24",
		},
	}
	group, err := endpointgroups.Create(client, createOpts).Extract()
	if err != nil {
		return group, err
	}

	t.Logf("Successfully created group %s", groupName)

	th.AssertEquals(t, group.Name, groupName)

	return group, nil
}

// CreateEndpointGroupWithCIDR will create an endpoint group with a random name and a specified CIDR.
// An error will be returned if the group could not be created.
func CreateEndpointGroupWithCIDR(t *testing.T, client *gophercloud.ServiceClient, cidr string) (*endpointgroups.EndpointGroup, error) {
	groupName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create group %s", groupName)

	createOpts := endpointgroups.CreateOpts{
		Name: groupName,
		Type: endpointgroups.TypeCIDR,
		Endpoints: []string{
			cidr,
		},
	}
	group, err := endpointgroups.Create(client, createOpts).Extract()
	if err != nil {
		return group, err
	}

	t.Logf("Successfully created group %s", groupName)
	t.Logf("%v", group)

	th.AssertEquals(t, group.Name, groupName)

	return group, nil
}

// DeleteEndpointGroup will delete an Endpoint group with a specified ID. A fatal error will
// occur if the delete was not successful. This works best when used as a
// deferred function.
func DeleteEndpointGroup(t *testing.T, client *gophercloud.ServiceClient, epGroupID string) {
	t.Logf("Attempting to delete endpoint group: %s", epGroupID)

	err := endpointgroups.Delete(client, epGroupID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete endpoint group %s: %v", epGroupID, err)
	}

	t.Logf("Deleted endpoint group: %s", epGroupID)

}

// CreateEndpointGroupWithSubnet will create an endpoint group with a random name.
// An error will be returned if the group could not be created.
func CreateEndpointGroupWithSubnet(t *testing.T, client *gophercloud.ServiceClient, subnetID string) (*endpointgroups.EndpointGroup, error) {
	groupName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create group %s", groupName)

	createOpts := endpointgroups.CreateOpts{
		Name: groupName,
		Type: endpointgroups.TypeSubnet,
		Endpoints: []string{
			subnetID,
		},
	}
	group, err := endpointgroups.Create(client, createOpts).Extract()
	if err != nil {
		return group, err
	}

	t.Logf("Successfully created group %s", groupName)

	th.AssertEquals(t, group.Name, groupName)

	return group, nil
}

// CreateSiteConnection will create an IPSec site connection with a random name and specified
// IKE policy, IPSec policy, service, peer EP group and local EP Group.
// An error will be returned if the connection could not be created.
func CreateSiteConnection(t *testing.T, client *gophercloud.ServiceClient, ikepolicyID string, ipsecpolicyID string, serviceID string, peerEPGroupID string, localEPGroupID string) (*siteconnections.Connection, error) {
	connectionName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create IPSec site connection %s", connectionName)

	createOpts := siteconnections.CreateOpts{
		Name:           connectionName,
		PSK:            "secret",
		Initiator:      siteconnections.InitiatorBiDirectional,
		AdminStateUp:   gophercloud.Enabled,
		IPSecPolicyID:  ipsecpolicyID,
		PeerEPGroupID:  peerEPGroupID,
		IKEPolicyID:    ikepolicyID,
		VPNServiceID:   serviceID,
		LocalEPGroupID: localEPGroupID,
		PeerAddress:    "172.24.4.233",
		PeerID:         "172.24.4.233",
		MTU:            1500,
	}
	connection, err := siteconnections.Create(client, createOpts).Extract()
	if err != nil {
		return connection, err
	}

	t.Logf("Successfully created IPSec Site Connection %s", connectionName)

	th.AssertEquals(t, connection.Name, connectionName)

	return connection, nil
}

// DeleteSiteConnection will delete an IPSec site connection with a specified ID. A fatal error will
// occur if the delete was not successful. This works best when used as a
// deferred function.
func DeleteSiteConnection(t *testing.T, client *gophercloud.ServiceClient, siteConnectionID string) {
	t.Logf("Attempting to delete site connection: %s", siteConnectionID)

	err := siteconnections.Delete(client, siteConnectionID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete site connection %s: %v", siteConnectionID, err)
	}

	t.Logf("Deleted site connection: %s", siteConnectionID)
}
