package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/external"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	nettest "github.com/gophercloud/gophercloud/openstack/networking/v2/networks/testing"
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

		fmt.Fprintf(w, nettest.ListResponse)
	})

	type NetworkWithExternalExt struct {
		networks.Network
		external.NetworkExternalExt
	}
	var actual []NetworkWithExternalExt

	allPages, err := networks.List(fake.ServiceClient(), networks.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)

	err = networks.ExtractNetworksInto(allPages, &actual)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "d32019d3-bc6e-4319-9c1d-6722fc136a22", actual[0].ID)
	th.AssertEquals(t, true, actual[0].External)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, nettest.GetResponse)
	})

	var s struct {
		networks.Network
		external.NetworkExternalExt
	}

	err := networks.Get(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").ExtractInto(&s)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "d32019d3-bc6e-4319-9c1d-6722fc136a22", s.ID)
	th.AssertEquals(t, true, s.External)
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
	iFalse := false
	networkCreateOpts := networks.CreateOpts{
		Name:         "private",
		AdminStateUp: &iTrue,
	}

	externalCreateOpts := external.CreateOptsExt{
		CreateOptsBuilder: &networkCreateOpts,
		External:          &iFalse,
	}

	_, err := networks.Create(fake.ServiceClient(), externalCreateOpts).Extract()
	th.AssertNoErr(t, err)

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

	iTrue := true
	iFalse := false
	networkUpdateOpts := networks.UpdateOpts{
		Name:         "new_network_name",
		AdminStateUp: &iFalse,
		Shared:       &iTrue,
	}

	externalUpdateOpts := external.UpdateOptsExt{
		UpdateOptsBuilder: &networkUpdateOpts,
		External:          &iFalse,
	}

	_, err := networks.Update(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", externalUpdateOpts).Extract()
	th.AssertNoErr(t, err)
}
