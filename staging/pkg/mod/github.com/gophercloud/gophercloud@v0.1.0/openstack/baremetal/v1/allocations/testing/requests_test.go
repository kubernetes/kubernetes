package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/baremetal/v1/allocations"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListAllocations(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleAllocationListSuccessfully(t)

	pages := 0
	err := allocations.List(client.ServiceClient(), allocations.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := allocations.ExtractAllocations(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 2 {
			t.Fatalf("Expected 2 allocations, got %d", len(actual))
		}
		th.AssertEquals(t, "5344a3e2-978a-444e-990a-cbf47c62ef88", actual[0].UUID)
		th.AssertEquals(t, "eff80f47-75f0-4d41-b1aa-cf07c201adac", actual[1].UUID)

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestCreateAllocation(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleAllocationCreationSuccessfully(t, SingleAllocationBody)

	actual, err := allocations.Create(client.ServiceClient(), allocations.CreateOpts{
		Name:           "allocation-1",
		ResourceClass:  "baremetal",
		CandidateNodes: []string{"344a3e2-978a-444e-990a-cbf47c62ef88"},
		Traits:         []string{"foo"},
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, Allocation1, *actual)
}

func TestDeleteAllocation(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleAllocationDeletionSuccessfully(t)

	res := allocations.Delete(client.ServiceClient(), "344a3e2-978a-444e-990a-cbf47c62ef88")
	th.AssertNoErr(t, res.Err)
}

func TestGetAllocation(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleAllocationGetSuccessfully(t)

	c := client.ServiceClient()
	actual, err := allocations.Get(c, "344a3e2-978a-444e-990a-cbf47c62ef88").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, Allocation1, *actual)
}
