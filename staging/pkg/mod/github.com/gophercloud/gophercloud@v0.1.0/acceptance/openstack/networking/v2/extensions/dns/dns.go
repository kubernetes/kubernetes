package dns

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/dns"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
	th "github.com/gophercloud/gophercloud/testhelper"
)

// PortWithDNSExt represents a port with the DNS fields
type PortWithDNSExt struct {
	ports.Port
	dns.PortDNSExt
}

// FloatingIPWithDNSExt represents a floating IP with the DNS fields
type FloatingIPWithDNSExt struct {
	floatingips.FloatingIP
	dns.FloatingIPDNSExt
}

// NetworkWithDNSExt represents a network with the DNS fields
type NetworkWithDNSExt struct {
	networks.Network
	dns.NetworkDNSExt
}

// CreatePortDNS will create a port with a DNS name on the specified subnet. An
// error will be returned if the port could not be created.
func CreatePortDNS(t *testing.T, client *gophercloud.ServiceClient, networkID, subnetID, dnsName string) (*PortWithDNSExt, error) {
	portName := tools.RandomString("TESTACC-", 8)
	portDescription := tools.RandomString("TESTACC-PORT-DESC-", 8)
	iFalse := true

	t.Logf("Attempting to create port: %s", portName)

	portCreateOpts := ports.CreateOpts{
		NetworkID:    networkID,
		Name:         portName,
		Description:  portDescription,
		AdminStateUp: &iFalse,
		FixedIPs:     []ports.IP{ports.IP{SubnetID: subnetID}},
	}

	createOpts := dns.PortCreateOptsExt{
		CreateOptsBuilder: portCreateOpts,
		DNSName:           dnsName,
	}

	var port PortWithDNSExt

	err := ports.Create(client, createOpts).ExtractInto(&port)
	if err != nil {
		return &port, err
	}

	t.Logf("Successfully created port: %s", portName)

	th.AssertEquals(t, port.Name, portName)
	th.AssertEquals(t, port.Description, portDescription)
	th.AssertEquals(t, port.DNSName, dnsName)

	return &port, nil
}

// CreateFloatingIPDNS creates a floating IP with the DNS extension on a given
// network and port. An error will be returned if the creation failed.
func CreateFloatingIPDNS(t *testing.T, client *gophercloud.ServiceClient, networkID, portID, dnsName, dnsDomain string) (*FloatingIPWithDNSExt, error) {
	t.Logf("Attempting to create floating IP on port: %s", portID)

	fipDescription := "Test floating IP"
	fipCreateOpts := &floatingips.CreateOpts{
		Description:       fipDescription,
		FloatingNetworkID: networkID,
		PortID:            portID,
	}

	createOpts := dns.FloatingIPCreateOptsExt{
		CreateOptsBuilder: fipCreateOpts,
		DNSName:           dnsName,
		DNSDomain:         dnsDomain,
	}

	var floatingIP FloatingIPWithDNSExt
	err := floatingips.Create(client, createOpts).ExtractInto(&floatingIP)
	if err != nil {
		return &floatingIP, err
	}

	t.Logf("Created floating IP.")

	th.AssertEquals(t, floatingIP.Description, fipDescription)
	th.AssertEquals(t, floatingIP.FloatingNetworkID, networkID)
	th.AssertEquals(t, floatingIP.PortID, portID)
	th.AssertEquals(t, floatingIP.DNSName, dnsName)
	th.AssertEquals(t, floatingIP.DNSDomain, dnsDomain)

	return &floatingIP, err
}

// CreateNetworkDNS will create a network with a DNS domain set.
// An error will be returned if the network could not be created.
func CreateNetworkDNS(t *testing.T, client *gophercloud.ServiceClient, dnsDomanin string) (*NetworkWithDNSExt, error) {
	networkName := tools.RandomString("TESTACC-", 8)
	networkCreateOpts := networks.CreateOpts{
		Name:         networkName,
		AdminStateUp: gophercloud.Enabled,
	}

	createOpts := dns.NetworkCreateOptsExt{
		CreateOptsBuilder: networkCreateOpts,
		DNSDomain:         dnsDomanin,
	}

	t.Logf("Attempting to create network: %s", networkName)

	var network NetworkWithDNSExt

	err := networks.Create(client, createOpts).ExtractInto(&network)
	if err != nil {
		return &network, err
	}

	t.Logf("Successfully created network.")

	th.AssertEquals(t, network.Name, networkName)
	th.AssertEquals(t, network.DNSDomain, dnsDomanin)

	return &network, nil
}
