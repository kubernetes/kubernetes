package v2

import (
	"fmt"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/apiversions"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/subnets"
)

// CreateNetwork will create basic network. An error will be returned if the
// network could not be created.
func CreateNetwork(t *testing.T, client *gophercloud.ServiceClient) (*networks.Network, error) {
	networkName := tools.RandomString("TESTACC-", 8)
	createOpts := networks.CreateOpts{
		Name:         networkName,
		AdminStateUp: gophercloud.Enabled,
	}

	t.Logf("Attempting to create network: %s", networkName)

	network, err := networks.Create(client, createOpts).Extract()
	if err != nil {
		return network, err
	}

	t.Logf("Successfully created network.")
	return network, nil
}

// CreatePort will create a port on the specified subnet. An error will be
// returned if the port could not be created.
func CreatePort(t *testing.T, client *gophercloud.ServiceClient, networkID, subnetID string) (*ports.Port, error) {
	portName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create port: %s", portName)

	createOpts := ports.CreateOpts{
		NetworkID:    networkID,
		Name:         portName,
		AdminStateUp: gophercloud.Enabled,
		FixedIPs:     []ports.IP{ports.IP{SubnetID: subnetID}},
	}

	port, err := ports.Create(client, createOpts).Extract()
	if err != nil {
		return port, err
	}

	if err := WaitForPortToCreate(client, port.ID, 60); err != nil {
		return port, err
	}

	newPort, err := ports.Get(client, port.ID).Extract()
	if err != nil {
		return newPort, err
	}

	t.Logf("Successfully created port: %s", portName)

	return newPort, nil
}

// CreateSubnet will create a subnet on the specified Network ID. An error
// will be returned if the subnet could not be created.
func CreateSubnet(t *testing.T, client *gophercloud.ServiceClient, networkID string) (*subnets.Subnet, error) {
	subnetName := tools.RandomString("TESTACC-", 8)
	subnetOctet := tools.RandomInt(1, 250)
	subnetCIDR := fmt.Sprintf("192.168.%d.0/24", subnetOctet)
	subnetGateway := fmt.Sprintf("192.168.%d.1", subnetOctet)
	createOpts := subnets.CreateOpts{
		NetworkID:  networkID,
		CIDR:       subnetCIDR,
		IPVersion:  4,
		Name:       subnetName,
		EnableDHCP: gophercloud.Disabled,
		GatewayIP:  &subnetGateway,
	}

	t.Logf("Attempting to create subnet: %s", subnetName)

	subnet, err := subnets.Create(client, createOpts).Extract()
	if err != nil {
		return subnet, err
	}

	t.Logf("Successfully created subnet.")
	return subnet, nil
}

// CreateSubnetWithDefaultGateway will create a subnet on the specified Network
// ID and have Neutron set the gateway by default An error will be returned if
// the subnet could not be created.
func CreateSubnetWithDefaultGateway(t *testing.T, client *gophercloud.ServiceClient, networkID string) (*subnets.Subnet, error) {
	subnetName := tools.RandomString("TESTACC-", 8)
	subnetOctet := tools.RandomInt(1, 250)
	subnetCIDR := fmt.Sprintf("192.168.%d.0/24", subnetOctet)
	createOpts := subnets.CreateOpts{
		NetworkID:  networkID,
		CIDR:       subnetCIDR,
		IPVersion:  4,
		Name:       subnetName,
		EnableDHCP: gophercloud.Disabled,
	}

	t.Logf("Attempting to create subnet: %s", subnetName)

	subnet, err := subnets.Create(client, createOpts).Extract()
	if err != nil {
		return subnet, err
	}

	t.Logf("Successfully created subnet.")
	return subnet, nil
}

// CreateSubnetWithNoGateway will create a subnet with no gateway on the
// specified Network ID.  An error will be returned if the subnet could not be
// created.
func CreateSubnetWithNoGateway(t *testing.T, client *gophercloud.ServiceClient, networkID string) (*subnets.Subnet, error) {
	var noGateway = ""
	subnetName := tools.RandomString("TESTACC-", 8)
	subnetOctet := tools.RandomInt(1, 250)
	subnetCIDR := fmt.Sprintf("192.168.%d.0/24", subnetOctet)
	dhcpStart := fmt.Sprintf("192.168.%d.10", subnetOctet)
	dhcpEnd := fmt.Sprintf("192.168.%d.200", subnetOctet)
	createOpts := subnets.CreateOpts{
		NetworkID:  networkID,
		CIDR:       subnetCIDR,
		IPVersion:  4,
		Name:       subnetName,
		EnableDHCP: gophercloud.Disabled,
		GatewayIP:  &noGateway,
		AllocationPools: []subnets.AllocationPool{
			{
				Start: dhcpStart,
				End:   dhcpEnd,
			},
		},
	}

	t.Logf("Attempting to create subnet: %s", subnetName)

	subnet, err := subnets.Create(client, createOpts).Extract()
	if err != nil {
		return subnet, err
	}

	t.Logf("Successfully created subnet.")
	return subnet, nil
}

// DeleteNetwork will delete a network with a specified ID. A fatal error will
// occur if the delete was not successful. This works best when used as a
// deferred function.
func DeleteNetwork(t *testing.T, client *gophercloud.ServiceClient, networkID string) {
	t.Logf("Attempting to delete network: %s", networkID)

	err := networks.Delete(client, networkID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete network %s: %v", networkID, err)
	}

	t.Logf("Deleted network: %s", networkID)
}

// DeletePort will delete a port with a specified ID. A fatal error will
// occur if the delete was not successful. This works best when used as a
// deferred function.
func DeletePort(t *testing.T, client *gophercloud.ServiceClient, portID string) {
	t.Logf("Attempting to delete port: %s", portID)

	err := ports.Delete(client, portID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete port %s: %v", portID, err)
	}

	t.Logf("Deleted port: %s", portID)
}

// DeleteSubnet will delete a subnet with a specified ID. A fatal error will
// occur if the delete was not successful. This works best when used as a
// deferred function.
func DeleteSubnet(t *testing.T, client *gophercloud.ServiceClient, subnetID string) {
	t.Logf("Attempting to delete subnet: %s", subnetID)

	err := subnets.Delete(client, subnetID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete subnet %s: %v", subnetID, err)
	}

	t.Logf("Deleted subnet: %s", subnetID)
}

// PrintAPIVersion will print an API version and all of its attributes.
func PrintAPIVersion(t *testing.T, apiVersion *apiversions.APIVersion) {
	t.Logf("ID: %s", apiVersion.ID)
	t.Logf("Status: %s", apiVersion.Status)
}

// PrintNetwork will print a network and all of its attributes.
func PrintNetwork(t *testing.T, network *networks.Network) {
	t.Logf("ID: %s", network.ID)
	t.Logf("Name: %s", network.Name)
	t.Logf("AdminStateUp: %t", network.AdminStateUp)
	t.Logf("Status: %s", network.Status)
	t.Logf("TenantID: %s", network.TenantID)
	t.Logf("Shared: %t", network.Shared)
	t.Logf("Subnets: %s", network.Subnets)
}

// PrintPort will print a port and all of its attributes.
func PrintPort(t *testing.T, port *ports.Port) {
	t.Logf("ID: %s", port.ID)
	t.Logf("NetworkID: %s", port.NetworkID)
	t.Logf("Name: %s", port.Name)
	t.Logf("AdminStateUp: %t", port.AdminStateUp)
	t.Logf("Status: %s", port.Status)
	t.Logf("MACAddress: %s", port.MACAddress)
	t.Logf("FixedIPs: %s", port.FixedIPs)
	t.Logf("TenantID: %s", port.TenantID)
	t.Logf("DeviceOwner: %s", port.DeviceOwner)
	t.Logf("SecurityGroups: %s", port.SecurityGroups)
	t.Logf("DeviceID: %s", port.DeviceID)
	t.Logf("DeviceOwner: %s", port.DeviceOwner)
	t.Logf("AllowedAddressPairs: %s", port.AllowedAddressPairs)
}

// PrintSubnet will print a subnet and all of its attributes.
func PrintSubnet(t *testing.T, subnet *subnets.Subnet) {
	t.Logf("ID: %s", subnet.ID)
	t.Logf("NetworkID: %s", subnet.NetworkID)
	t.Logf("Name: %s", subnet.Name)
	t.Logf("IPVersion: %d", subnet.IPVersion)
	t.Logf("CIDR: %s", subnet.CIDR)
	t.Logf("GatewayIP: %s", subnet.GatewayIP)
	t.Logf("DNSNameservers: %s", subnet.DNSNameservers)
	t.Logf("AllocationPools: %s", subnet.AllocationPools)
	t.Logf("HostRoutes: %s", subnet.HostRoutes)
	t.Logf("EnableDHCP: %t", subnet.EnableDHCP)
	t.Logf("TenantID: %s", subnet.TenantID)
}

// PrintVersionResource will print an API version resource and all of its attributes.
func PrintVersionResource(t *testing.T, versionResource *apiversions.APIVersionResource) {
	t.Logf("Name: %s", versionResource.Name)
	t.Logf("Collection: %s", versionResource.Collection)
}

func WaitForPortToCreate(client *gophercloud.ServiceClient, portID string, secs int) error {
	return gophercloud.WaitFor(secs, func() (bool, error) {
		p, err := ports.Get(client, portID).Extract()
		if err != nil {
			return false, err
		}

		if p.Status == "ACTIVE" || p.Status == "DOWN" {
			return true, nil
		}

		return false, nil
	})
}
