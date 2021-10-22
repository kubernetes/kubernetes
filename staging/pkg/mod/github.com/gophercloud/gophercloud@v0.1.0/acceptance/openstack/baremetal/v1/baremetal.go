package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/baremetal/v1/allocations"
	"github.com/gophercloud/gophercloud/openstack/baremetal/v1/nodes"
	"github.com/gophercloud/gophercloud/openstack/baremetal/v1/ports"
)

// CreateNode creates a basic node with a randomly generated name.
func CreateNode(t *testing.T, client *gophercloud.ServiceClient) (*nodes.Node, error) {
	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create bare metal node: %s", name)

	node, err := nodes.Create(client, nodes.CreateOpts{
		Name:          name,
		Driver:        "ipmi",
		BootInterface: "pxe",
		RAIDInterface: "agent",
		DriverInfo: map[string]interface{}{
			"ipmi_port":      "6230",
			"ipmi_username":  "admin",
			"deploy_kernel":  "http://172.22.0.1/images/tinyipa-stable-rocky.vmlinuz",
			"ipmi_address":   "192.168.122.1",
			"deploy_ramdisk": "http://172.22.0.1/images/tinyipa-stable-rocky.gz",
			"ipmi_password":  "admin",
		},
	}).Extract()

	return node, err
}

// DeleteNode deletes a bare metal node via its UUID.
func DeleteNode(t *testing.T, client *gophercloud.ServiceClient, node *nodes.Node) {
	err := nodes.Delete(client, node.UUID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete node %s: %s", node.UUID, err)
	}

	t.Logf("Deleted server: %s", node.UUID)
}

// CreateAllocation creates an allocation
func CreateAllocation(t *testing.T, client *gophercloud.ServiceClient) (*allocations.Allocation, error) {
	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create bare metal allocation: %s", name)

	allocation, err := allocations.Create(client, allocations.CreateOpts{
		Name:          name,
		ResourceClass: "baremetal",
	}).Extract()

	return allocation, err
}

// DeleteAllocation deletes a bare metal allocation via its UUID.
func DeleteAllocation(t *testing.T, client *gophercloud.ServiceClient, allocation *allocations.Allocation) {
	err := allocations.Delete(client, allocation.UUID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete allocation %s: %s", allocation.UUID, err)
	}

	t.Logf("Deleted allocation: %s", allocation.UUID)
}

// CreateFakeNode creates a node with fake-hardware to use for port tests.
func CreateFakeNode(t *testing.T, client *gophercloud.ServiceClient) (*nodes.Node, error) {
	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create bare metal node: %s", name)

	node, err := nodes.Create(client, nodes.CreateOpts{
		Name:          name,
		Driver:        "fake-hardware",
		BootInterface: "pxe",
		DriverInfo: map[string]interface{}{
			"ipmi_port":      "6230",
			"ipmi_username":  "admin",
			"deploy_kernel":  "http://172.22.0.1/images/tinyipa-stable-rocky.vmlinuz",
			"ipmi_address":   "192.168.122.1",
			"deploy_ramdisk": "http://172.22.0.1/images/tinyipa-stable-rocky.gz",
			"ipmi_password":  "admin",
		},
	}).Extract()

	return node, err
}

// CreatePort - creates a port for a node with a fixed Address
func CreatePort(t *testing.T, client *gophercloud.ServiceClient, node *nodes.Node) (*ports.Port, error) {
	mac := "e6:72:1f:52:00:f4"
	t.Logf("Attempting to create Port for Node: %s with Address: %s", node.UUID, mac)

	iTrue := true
	port, err := ports.Create(client, ports.CreateOpts{
		NodeUUID:   node.UUID,
		Address:    mac,
		PXEEnabled: &iTrue,
	}).Extract()

	return port, err
}

// DeletePort - deletes a port via its UUID
func DeletePort(t *testing.T, client *gophercloud.ServiceClient, port *ports.Port) {
	err := ports.Delete(client, port.UUID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete port %s: %s", port.UUID, err)
	}

	t.Logf("Deleted port: %s", port.UUID)

}
