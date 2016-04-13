// +build acceptance lbs

package v1

import (
	"os"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/diskconfig"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/servers"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/lbs"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/nodes"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestNodes(t *testing.T) {
	client := setup(t)

	serverIP := findServer(t)
	ids := createLB(t, client, 1)
	lbID := ids[0]

	nodeID := addNodes(t, client, lbID, serverIP)

	listNodes(t, client, lbID)

	getNode(t, client, lbID, nodeID)

	updateNode(t, client, lbID, nodeID)

	listEvents(t, client, lbID)

	deleteNode(t, client, lbID, nodeID)

	waitForLB(client, lbID, lbs.ACTIVE)
	deleteLB(t, client, lbID)
}

func findServer(t *testing.T) string {
	var serverIP string

	client, err := newComputeClient()
	th.AssertNoErr(t, err)

	err = servers.List(client, nil).EachPage(func(page pagination.Page) (bool, error) {
		sList, err := servers.ExtractServers(page)
		th.AssertNoErr(t, err)

		for _, s := range sList {
			serverIP = s.AccessIPv4
			t.Logf("Found an existing server: ID [%s] Public IP [%s]", s.ID, serverIP)
			break
		}

		return true, nil
	})
	th.AssertNoErr(t, err)

	if serverIP == "" {
		t.Log("No server found, creating one")

		imageRef := os.Getenv("RS_IMAGE_ID")
		if imageRef == "" {
			t.Fatalf("OS var RS_IMAGE_ID undefined")
		}
		flavorRef := os.Getenv("RS_FLAVOR_ID")
		if flavorRef == "" {
			t.Fatalf("OS var RS_FLAVOR_ID undefined")
		}

		opts := &servers.CreateOpts{
			Name:       tools.RandomString("lb_test_", 5),
			ImageRef:   imageRef,
			FlavorRef:  flavorRef,
			DiskConfig: diskconfig.Manual,
		}

		s, err := servers.Create(client, opts).Extract()
		th.AssertNoErr(t, err)
		serverIP = s.AccessIPv4

		t.Logf("Created server %s, waiting for it to build", s.ID)
		err = servers.WaitForStatus(client, s.ID, "ACTIVE", 300)
		th.AssertNoErr(t, err)
		t.Logf("Server created successfully.")
	}

	return serverIP
}

func addNodes(t *testing.T, client *gophercloud.ServiceClient, lbID int, serverIP string) int {
	opts := nodes.CreateOpts{
		nodes.CreateOpt{
			Address:   serverIP,
			Port:      80,
			Condition: nodes.ENABLED,
			Type:      nodes.PRIMARY,
		},
	}

	page := nodes.Create(client, lbID, opts)

	nodeList, err := page.ExtractNodes()
	th.AssertNoErr(t, err)

	var nodeID int
	for _, n := range nodeList {
		nodeID = n.ID
	}
	if nodeID == 0 {
		t.Fatalf("nodeID could not be extracted from create response")
	}

	t.Logf("Added node %d to LB %d", nodeID, lbID)
	waitForLB(client, lbID, lbs.ACTIVE)

	return nodeID
}

func listNodes(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	err := nodes.List(client, lbID, nil).EachPage(func(page pagination.Page) (bool, error) {
		nodeList, err := nodes.ExtractNodes(page)
		th.AssertNoErr(t, err)

		for _, n := range nodeList {
			t.Logf("Listing node: ID [%d] Address [%s:%d] Status [%s]", n.ID, n.Address, n.Port, n.Status)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)
}

func getNode(t *testing.T, client *gophercloud.ServiceClient, lbID int, nodeID int) {
	node, err := nodes.Get(client, lbID, nodeID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Getting node %d: Type [%s] Weight [%d]", nodeID, node.Type, node.Weight)
}

func updateNode(t *testing.T, client *gophercloud.ServiceClient, lbID int, nodeID int) {
	opts := nodes.UpdateOpts{
		Weight:    gophercloud.IntToPointer(10),
		Condition: nodes.DRAINING,
		Type:      nodes.SECONDARY,
	}
	err := nodes.Update(client, lbID, nodeID, opts).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Updated node %d", nodeID)
	waitForLB(client, lbID, lbs.ACTIVE)
}

func listEvents(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	pager := nodes.ListEvents(client, lbID, nodes.ListEventsOpts{})
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		eventList, err := nodes.ExtractNodeEvents(page)
		th.AssertNoErr(t, err)

		for _, e := range eventList {
			t.Logf("Listing events for node %d: Type [%s] Msg [%s] Severity [%s] Date [%s]",
				e.NodeID, e.Type, e.DetailedMessage, e.Severity, e.Created)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)
}

func deleteNode(t *testing.T, client *gophercloud.ServiceClient, lbID int, nodeID int) {
	err := nodes.Delete(client, lbID, nodeID).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Deleted node %d", nodeID)
}
