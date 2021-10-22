package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/baremetal/v1/ports"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListDetailPorts(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePortListDetailSuccessfully(t)

	pages := 0
	err := ports.ListDetail(client.ServiceClient(), ports.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := ports.ExtractPorts(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 2 {
			t.Fatalf("Expected 2 ports, got %d", len(actual))
		}
		th.CheckDeepEquals(t, PortBar, actual[0])
		th.CheckDeepEquals(t, PortFoo, actual[1])

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListPorts(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePortListSuccessfully(t)

	pages := 0
	err := ports.List(client.ServiceClient(), ports.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := ports.ExtractPorts(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 2 {
			t.Fatalf("Expected 2 ports, got %d", len(actual))
		}
		th.AssertEquals(t, "3abe3f36-9708-4e9f-b07e-0f898061d3a7", actual[0].UUID)
		th.AssertEquals(t, "f2845e11-dbd4-4728-a8c0-30d19f48924a", actual[1].UUID)

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListOpts(t *testing.T) {
	// Detail cannot take Fields
	opts := ports.ListOpts{
		Fields: []string{"uuid", "address"},
	}

	_, err := opts.ToPortListDetailQuery()
	th.AssertEquals(t, err.Error(), "fields is not a valid option when getting a detailed listing of ports")

	// Regular ListOpts can
	query, err := opts.ToPortListQuery()
	th.AssertEquals(t, query, "?fields=uuid&fields=address")
	th.AssertNoErr(t, err)
}

func TestCreatePort(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePortCreationSuccessfully(t, SinglePortBody)

	iTrue := true
	actual, err := ports.Create(client.ServiceClient(), ports.CreateOpts{
		NodeUUID:   "ddd06a60-b91e-4ab4-a6e7-56c0b25b6086",
		Address:    "52:54:00:4d:87:e6",
		PXEEnabled: &iTrue,
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, PortFoo, *actual)
}

func TestDeletePort(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePortDeletionSuccessfully(t)

	res := ports.Delete(client.ServiceClient(), "3abe3f36-9708-4e9f-b07e-0f898061d3a7")
	th.AssertNoErr(t, res.Err)
}

func TestGetPort(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePortGetSuccessfully(t)

	c := client.ServiceClient()
	actual, err := ports.Get(c, "f2845e11-dbd4-4728-a8c0-30d19f48924a").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, PortFoo, *actual)
}

func TestUpdatePort(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePortUpdateSuccessfully(t, SinglePortBody)

	c := client.ServiceClient()
	actual, err := ports.Update(c, "f2845e11-dbd4-4728-a8c0-30d19f48924a", ports.UpdateOpts{
		ports.UpdateOperation{
			Op:    ports.ReplaceOp,
			Path:  "/address",
			Value: "22:22:22:22:22:22",
		},
	}).Extract()
	if err != nil {
		t.Fatalf("Unexpected Update error: %v", err)
	}

	th.CheckDeepEquals(t, PortFoo, *actual)
}
