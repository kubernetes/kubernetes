package nodes

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

const (
	lbID    = 12345
	nodeID  = 67890
	nodeID2 = 67891
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockListResponse(t, lbID)

	count := 0

	err := List(client.ServiceClient(), lbID, nil).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractNodes(page)
		th.AssertNoErr(t, err)

		expected := []Node{
			Node{
				ID:        410,
				Address:   "10.1.1.1",
				Port:      80,
				Condition: ENABLED,
				Status:    ONLINE,
				Weight:    3,
				Type:      PRIMARY,
			},
			Node{
				ID:        411,
				Address:   "10.1.1.2",
				Port:      80,
				Condition: ENABLED,
				Status:    ONLINE,
				Weight:    8,
				Type:      SECONDARY,
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, count)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockCreateResponse(t, lbID)

	opts := CreateOpts{
		CreateOpt{
			Address:   "10.2.2.3",
			Port:      80,
			Condition: ENABLED,
			Type:      PRIMARY,
		},
		CreateOpt{
			Address:   "10.2.2.4",
			Port:      81,
			Condition: ENABLED,
			Type:      SECONDARY,
		},
	}

	page := Create(client.ServiceClient(), lbID, opts)

	actual, err := page.ExtractNodes()
	th.AssertNoErr(t, err)

	expected := []Node{
		Node{
			ID:        185,
			Address:   "10.2.2.3",
			Port:      80,
			Condition: ENABLED,
			Status:    ONLINE,
			Weight:    1,
			Type:      PRIMARY,
		},
		Node{
			ID:        186,
			Address:   "10.2.2.4",
			Port:      81,
			Condition: ENABLED,
			Status:    ONLINE,
			Weight:    1,
			Type:      SECONDARY,
		},
	}

	th.CheckDeepEquals(t, expected, actual)
}

func TestCreateErr(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockCreateErrResponse(t, lbID)

	opts := CreateOpts{
		CreateOpt{
			Address:   "10.2.2.3",
			Port:      80,
			Condition: ENABLED,
			Type:      PRIMARY,
		},
		CreateOpt{
			Address:   "10.2.2.4",
			Port:      81,
			Condition: ENABLED,
			Type:      SECONDARY,
		},
	}

	page := Create(client.ServiceClient(), lbID, opts)

	actual, err := page.ExtractNodes()
	if err == nil {
		t.Fatal("Did not receive expected error from ExtractNodes")
	}
	if actual != nil {
		t.Fatalf("Received non-nil result from failed ExtractNodes: %#v", actual)
	}
}

func TestBulkDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	ids := []int{nodeID, nodeID2}

	mockBatchDeleteResponse(t, lbID, ids)

	err := BulkDelete(client.ServiceClient(), lbID, ids).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockGetResponse(t, lbID, nodeID)

	node, err := Get(client.ServiceClient(), lbID, nodeID).Extract()
	th.AssertNoErr(t, err)

	expected := &Node{
		ID:        410,
		Address:   "10.1.1.1",
		Port:      80,
		Condition: ENABLED,
		Status:    ONLINE,
		Weight:    12,
		Type:      PRIMARY,
	}

	th.AssertDeepEquals(t, expected, node)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockUpdateResponse(t, lbID, nodeID)

	opts := UpdateOpts{
		Weight:    gophercloud.IntToPointer(10),
		Condition: DRAINING,
		Type:      SECONDARY,
	}

	err := Update(client.ServiceClient(), lbID, nodeID, opts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDeleteResponse(t, lbID, nodeID)

	err := Delete(client.ServiceClient(), lbID, nodeID).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestListEvents(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockListEventsResponse(t, lbID)

	count := 0

	pager := ListEvents(client.ServiceClient(), lbID, ListEventsOpts{})

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractNodeEvents(page)
		th.AssertNoErr(t, err)

		expected := []NodeEvent{
			NodeEvent{
				DetailedMessage: "Node is ok",
				NodeID:          373,
				ID:              7,
				Type:            "UPDATE_NODE",
				Description:     "Node '373' status changed to 'ONLINE' for load balancer '323'",
				Category:        "UPDATE",
				Severity:        "INFO",
				RelativeURI:     "/406271/loadbalancers/323/nodes/373/events",
				AccountID:       406271,
				LoadBalancerID:  323,
				Title:           "Node Status Updated",
				Author:          "Rackspace Cloud",
				Created:         "10-30-2012 10:18:23",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, count)
}
