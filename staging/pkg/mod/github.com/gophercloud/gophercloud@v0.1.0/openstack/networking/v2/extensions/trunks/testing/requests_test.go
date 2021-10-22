package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/trunks"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/trunks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, CreateRequest)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, CreateResponse)
	})

	iTrue := true
	options := trunks.CreateOpts{
		Name:         "gophertrunk",
		Description:  "Trunk created by gophercloud",
		AdminStateUp: &iTrue,
		Subports: []trunks.Subport{
			{
				SegmentationID:   1,
				SegmentationType: "vlan",
				PortID:           "28e452d7-4f8a-4be4-b1e6-7f3db4c0430b",
			},
			{
				SegmentationID:   2,
				SegmentationType: "vlan",
				PortID:           "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab",
			},
		},
	}
	_, err := trunks.Create(fake.ServiceClient(), options).Extract()
	if err == nil {
		t.Fatalf("Failed to detect missing parent PortID field")
	}
	options.PortID = "c373d2fa-3d3b-4492-924c-aff54dea19b6"
	n, err := trunks.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Status, "ACTIVE")
	expectedTrunks, err := ExpectedTrunkSlice()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, &expectedTrunks[1], n)
}

func TestCreateNoSubports(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/trunks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, CreateNoSubportsRequest)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, CreateNoSubportsResponse)
	})

	iTrue := true
	options := trunks.CreateOpts{
		Name:         "gophertrunk",
		Description:  "Trunk created by gophercloud",
		AdminStateUp: &iTrue,
		PortID:       "c373d2fa-3d3b-4492-924c-aff54dea19b6",
	}
	n, err := trunks.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Status, "ACTIVE")
	th.AssertEquals(t, 0, len(n.Subports))
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/trunks/f6a9718c-5a64-43e3-944f-4deccad8e78c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := trunks.Delete(fake.ServiceClient(), "f6a9718c-5a64-43e3-944f-4deccad8e78c")
	th.AssertNoErr(t, res.Err)
}

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/trunks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, ListResponse)
	})

	client := fake.ServiceClient()
	count := 0

	trunks.List(client, trunks.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := trunks.ExtractTrunks(page)
		if err != nil {
			t.Errorf("Failed to extract trunks: %v", err)
			return false, err
		}

		expected, err := ExpectedTrunkSlice()
		th.AssertNoErr(t, err)
		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/trunks/f6a9718c-5a64-43e3-944f-4deccad8e78c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, GetResponse)
	})

	n, err := trunks.Get(fake.ServiceClient(), "f6a9718c-5a64-43e3-944f-4deccad8e78c").Extract()
	th.AssertNoErr(t, err)
	expectedTrunks, err := ExpectedTrunkSlice()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &expectedTrunks[1], n)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/trunks/f6a9718c-5a64-43e3-944f-4deccad8e78c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, UpdateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, UpdateResponse)
	})

	iFalse := false
	name := "updated_gophertrunk"
	description := "gophertrunk updated by gophercloud"
	options := trunks.UpdateOpts{
		Name:         &name,
		AdminStateUp: &iFalse,
		Description:  &description,
	}
	n, err := trunks.Update(fake.ServiceClient(), "f6a9718c-5a64-43e3-944f-4deccad8e78c", options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Name, name)
	th.AssertEquals(t, n.AdminStateUp, iFalse)
	th.AssertEquals(t, n.Description, description)
}

func TestGetSubports(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/trunks/f6a9718c-5a64-43e3-944f-4deccad8e78c/get_subports", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, ListSubportsResponse)
	})

	client := fake.ServiceClient()

	subports, err := trunks.GetSubports(client, "f6a9718c-5a64-43e3-944f-4deccad8e78c").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedSubports, subports)
}

func TestMissingFields(t *testing.T) {
	iTrue := true
	opts := trunks.CreateOpts{
		Name:         "gophertrunk",
		PortID:       "c373d2fa-3d3b-4492-924c-aff54dea19b6",
		Description:  "Trunk created by gophercloud",
		AdminStateUp: &iTrue,
		Subports: []trunks.Subport{
			{
				SegmentationID:   1,
				SegmentationType: "vlan",
				PortID:           "28e452d7-4f8a-4be4-b1e6-7f3db4c0430b",
			},
			{
				SegmentationID:   2,
				SegmentationType: "vlan",
				PortID:           "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab",
			},
			{
				PortID: "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab",
			},
		},
	}

	_, err := opts.ToTrunkCreateMap()
	if err == nil {
		t.Fatalf("Failed to detect missing subport fields")
	}
}

func TestAddSubports(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/trunks/f6a9718c-5a64-43e3-944f-4deccad8e78c/add_subports", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, AddSubportsRequest)
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, AddSubportsResponse)
	})

	client := fake.ServiceClient()

	opts := trunks.AddSubportsOpts{
		Subports: ExpectedSubports,
	}

	trunk, err := trunks.AddSubports(client, "f6a9718c-5a64-43e3-944f-4deccad8e78c", opts).Extract()
	th.AssertNoErr(t, err)
	expectedTrunk, err := ExpectedSubportsAddedTrunk()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &expectedTrunk, trunk)
}

func TestRemoveSubports(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/trunks/f6a9718c-5a64-43e3-944f-4deccad8e78c/remove_subports", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, RemoveSubportsRequest)
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, RemoveSubportsResponse)
	})

	client := fake.ServiceClient()

	opts := trunks.RemoveSubportsOpts{
		Subports: []trunks.RemoveSubport{
			{PortID: "28e452d7-4f8a-4be4-b1e6-7f3db4c0430b"},
			{PortID: "4c8b2bff-9824-4d4c-9b60-b3f6621b2bab"},
		},
	}
	trunk, err := trunks.RemoveSubports(client, "f6a9718c-5a64-43e3-944f-4deccad8e78c", opts).Extract()

	th.AssertNoErr(t, err)
	expectedTrunk, err := ExpectedSubportsRemovedTrunk()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &expectedTrunk, trunk)
}
