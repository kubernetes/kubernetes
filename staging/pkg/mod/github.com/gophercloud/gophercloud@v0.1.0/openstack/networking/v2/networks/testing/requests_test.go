package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/portsecurity"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, ListResponse)
	})

	client := fake.ServiceClient()
	count := 0

	networks.List(client, networks.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := networks.ExtractNetworks(page)
		if err != nil {
			t.Errorf("Failed to extract networks: %v", err)
			return false, err
		}

		th.CheckDeepEquals(t, ExpectedNetworkSlice, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestListWithExtensions(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, ListResponse)
	})

	client := fake.ServiceClient()

	type networkWithExt struct {
		networks.Network
		portsecurity.PortSecurityExt
	}

	var allNetworks []networkWithExt

	allPages, err := networks.List(client, networks.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)

	err = networks.ExtractNetworksInto(allPages, &allNetworks)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, allNetworks[0].Status, "ACTIVE")
	th.AssertEquals(t, allNetworks[0].PortSecurityEnabled, true)
	th.AssertEquals(t, allNetworks[0].Subnets[0], "54d6f61d-db07-451c-9ab3-b9609b6b6f0b")
	th.AssertEquals(t, allNetworks[1].Subnets[0], "08eae331-0402-425a-923c-34f7cfe39c1b")
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, GetResponse)
	})

	n, err := networks.Get(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &Network1, n)
}

func TestGetWithExtensions(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, GetResponse)
	})

	var networkWithExtensions struct {
		networks.Network
		portsecurity.PortSecurityExt
	}

	err := networks.Get(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").ExtractInto(&networkWithExtensions)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, networkWithExtensions.Status, "ACTIVE")
	th.AssertEquals(t, networkWithExtensions.PortSecurityEnabled, true)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks", func(w http.ResponseWriter, r *http.Request) {
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
	options := networks.CreateOpts{Name: "private", AdminStateUp: &iTrue}
	n, err := networks.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Status, "ACTIVE")
	th.AssertDeepEquals(t, &Network2, n)
}

func TestCreateWithOptionalFields(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, CreateOptionalFieldsRequest)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `{}`)
	})

	iTrue := true
	options := networks.CreateOpts{
		Name:                  "public",
		AdminStateUp:          &iTrue,
		Shared:                &iTrue,
		TenantID:              "12345",
		AvailabilityZoneHints: []string{"zone1", "zone2"},
	}
	_, err := networks.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/4e8e5957-649f-477b-9e5b-f1f75b21c03c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, UpdateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, UpdateResponse)
	})

	iTrue, iFalse := true, false
	name := "new_network_name"
	options := networks.UpdateOpts{Name: &name, AdminStateUp: &iFalse, Shared: &iTrue}
	n, err := networks.Update(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Name, "new_network_name")
	th.AssertEquals(t, n.AdminStateUp, false)
	th.AssertEquals(t, n.Shared, true)
	th.AssertEquals(t, n.ID, "4e8e5957-649f-477b-9e5b-f1f75b21c03c")
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/4e8e5957-649f-477b-9e5b-f1f75b21c03c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := networks.Delete(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c")
	th.AssertNoErr(t, res.Err)
}

func TestCreatePortSecurity(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, CreatePortSecurityRequest)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, CreatePortSecurityResponse)
	})

	var networkWithExtensions struct {
		networks.Network
		portsecurity.PortSecurityExt
	}

	iTrue := true
	iFalse := false
	networkCreateOpts := networks.CreateOpts{Name: "private", AdminStateUp: &iTrue}
	createOpts := portsecurity.NetworkCreateOptsExt{
		CreateOptsBuilder:   networkCreateOpts,
		PortSecurityEnabled: &iFalse,
	}

	err := networks.Create(fake.ServiceClient(), createOpts).ExtractInto(&networkWithExtensions)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, networkWithExtensions.Status, "ACTIVE")
	th.AssertEquals(t, networkWithExtensions.PortSecurityEnabled, false)
}

func TestUpdatePortSecurity(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/4e8e5957-649f-477b-9e5b-f1f75b21c03c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, UpdatePortSecurityRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, UpdatePortSecurityResponse)
	})

	var networkWithExtensions struct {
		networks.Network
		portsecurity.PortSecurityExt
	}

	iFalse := false
	networkUpdateOpts := networks.UpdateOpts{}
	updateOpts := portsecurity.NetworkUpdateOptsExt{
		UpdateOptsBuilder:   networkUpdateOpts,
		PortSecurityEnabled: &iFalse,
	}

	err := networks.Update(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", updateOpts).ExtractInto(&networkWithExtensions)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, networkWithExtensions.Name, "private")
	th.AssertEquals(t, networkWithExtensions.AdminStateUp, true)
	th.AssertEquals(t, networkWithExtensions.Shared, false)
	th.AssertEquals(t, networkWithExtensions.ID, "4e8e5957-649f-477b-9e5b-f1f75b21c03c")
	th.AssertEquals(t, networkWithExtensions.PortSecurityEnabled, false)
}
