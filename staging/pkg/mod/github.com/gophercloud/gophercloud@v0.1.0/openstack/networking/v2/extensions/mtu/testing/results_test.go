package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/mtu"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	nettest "github.com/gophercloud/gophercloud/openstack/networking/v2/networks/testing"
	th "github.com/gophercloud/gophercloud/testhelper"
)

type NetworkMTU struct {
	networks.Network
	mtu.NetworkMTUExt
}

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

	type NetworkWithMTUExt struct {
		networks.Network
		mtu.NetworkMTUExt
	}
	var actual []NetworkWithMTUExt

	allPages, err := networks.List(fake.ServiceClient(), networks.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)

	err = networks.ExtractNetworksInto(allPages, &actual)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "d32019d3-bc6e-4319-9c1d-6722fc136a22", actual[0].ID)
	th.AssertEquals(t, 1500, actual[0].MTU)
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

	var s NetworkMTU

	err := networks.Get(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").ExtractInto(&s)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "d32019d3-bc6e-4319-9c1d-6722fc136a22", s.ID)
	th.AssertEquals(t, 1500, s.MTU)
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
	networkCreateOpts := networks.CreateOpts{
		Name:         "private",
		AdminStateUp: &iTrue,
	}

	mtuCreateOpts := mtu.CreateOptsExt{
		CreateOptsBuilder: &networkCreateOpts,
		MTU:               1500,
	}

	var s NetworkMTU

	err := networks.Create(fake.ServiceClient(), mtuCreateOpts).ExtractInto(&s)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "db193ab3-96e3-4cb3-8fc5-05f4296d0324", s.ID)
	th.AssertEquals(t, iTrue, s.AdminStateUp)
	th.AssertEquals(t, 1500, s.MTU)
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
	name := "new_network_name"
	networkUpdateOpts := networks.UpdateOpts{
		Name:         &name,
		AdminStateUp: &iFalse,
		Shared:       &iTrue,
	}

	mtuUpdateOpts := mtu.UpdateOptsExt{
		UpdateOptsBuilder: &networkUpdateOpts,
		MTU:               1350,
	}

	var s NetworkMTU

	err := networks.Update(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", mtuUpdateOpts).ExtractInto(&s)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "4e8e5957-649f-477b-9e5b-f1f75b21c03c", s.ID)
	th.AssertEquals(t, "new_network_name", s.Name)
	th.AssertEquals(t, iFalse, s.AdminStateUp)
	th.AssertEquals(t, iTrue, s.Shared)
	th.AssertEquals(t, 1350, s.MTU)
}
