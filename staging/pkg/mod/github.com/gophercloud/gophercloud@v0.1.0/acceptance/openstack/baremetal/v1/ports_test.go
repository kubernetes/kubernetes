// +build acceptance baremetal ports

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/baremetal/v1/ports"
	"github.com/gophercloud/gophercloud/pagination"

	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestPortsCreateDestroy(t *testing.T) {
	clients.RequireLong(t)

	client, err := clients.NewBareMetalV1Client()
	th.AssertNoErr(t, err)
	client.Microversion = "1.53"

	node, err := CreateFakeNode(t, client)
	port, err := CreatePort(t, client, node)
	th.AssertNoErr(t, err)
	defer DeleteNode(t, client, node)
	defer DeletePort(t, client, port)

	found := false
	err = ports.List(client, ports.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		portList, err := ports.ExtractPorts(page)
		if err != nil {
			return false, err
		}

		for _, p := range portList {
			if p.UUID == port.UUID {
				found = true
				return true, nil
			}
		}

		return false, nil
	})
	th.AssertNoErr(t, err)

	th.AssertEquals(t, found, true)
}

func TestPortsUpdate(t *testing.T) {
	clients.RequireLong(t)

	client, err := clients.NewBareMetalV1Client()
	th.AssertNoErr(t, err)
	client.Microversion = "1.53"

	node, err := CreateFakeNode(t, client)
	port, err := CreatePort(t, client, node)
	th.AssertNoErr(t, err)
	defer DeleteNode(t, client, node)
	defer DeletePort(t, client, port)

	updated, err := ports.Update(client, port.UUID, ports.UpdateOpts{
		ports.UpdateOperation{
			Op:    ports.ReplaceOp,
			Path:  "/address",
			Value: "aa:bb:cc:dd:ee:ff",
		},
	}).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, updated.Address, "aa:bb:cc:dd:ee:ff")
}
