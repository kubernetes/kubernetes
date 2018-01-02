package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud"
	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/provider"
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

	type NetworkWithExt struct {
		networks.Network
		provider.NetworkProviderExt
	}
	var actual []NetworkWithExt

	allPages, err := networks.List(fake.ServiceClient(), networks.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)

	err = networks.ExtractNetworksInto(allPages, &actual)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "d32019d3-bc6e-4319-9c1d-6722fc136a22", actual[0].ID)
	th.AssertEquals(t, "db193ab3-96e3-4cb3-8fc5-05f4296d0324", actual[1].ID)
	th.AssertEquals(t, "local", actual[1].NetworkType)
	th.AssertEquals(t, "1234567890", actual[1].SegmentationID)
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
		provider.NetworkProviderExt
	}

	err := networks.Get(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").ExtractInto(&s)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "d32019d3-bc6e-4319-9c1d-6722fc136a22", s.ID)
	th.AssertEquals(t, "", s.PhysicalNetwork)
	th.AssertEquals(t, "local", s.NetworkType)
	th.AssertEquals(t, "9876543210", s.SegmentationID)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, nettest.CreateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, nettest.CreateResponse)
	})

	var s struct {
		networks.Network
		provider.NetworkProviderExt
	}

	options := networks.CreateOpts{Name: "private", AdminStateUp: gophercloud.Enabled}
	err := networks.Create(fake.ServiceClient(), options).ExtractInto(&s)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "db193ab3-96e3-4cb3-8fc5-05f4296d0324", s.ID)
	th.AssertEquals(t, "", s.PhysicalNetwork)
	th.AssertEquals(t, "local", s.NetworkType)
	th.AssertEquals(t, "9876543210", s.SegmentationID)
}

func TestCreateWithMultipleProvider(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
	"network": {
			"name": "sample_network",
			"admin_state_up": true,
			"shared": true,
			"tenant_id": "12345",
			"segments": [
				{
					"provider:segmentation_id": 666,
					"provider:physical_network": "br-ex",
					"provider:network_type": "vxlan"
				},
				{
					"provider:segmentation_id": 615,
					"provider:physical_network": "br-ex",
					"provider:network_type": "vxlan"
				}
			]
	}
}
		`)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `
{
	"network": {
		"status": "ACTIVE",
		"name": "sample_network",
		"admin_state_up": true,
		"shared": true,
		"tenant_id": "12345",
		"segments": [
			{
				"provider:segmentation_id": 666,
				"provider:physical_network": "br-ex",
				"provider:network_type": "vlan"
			},
			{
				"provider:segmentation_id": 615,
				"provider:physical_network": "br-ex",
				"provider:network_type": "vlan"
			}
		]
	}
}
	`)
	})

	iTrue := true
	segments := []provider.Segment{
		provider.Segment{NetworkType: "vxlan", PhysicalNetwork: "br-ex", SegmentationID: 666},
		provider.Segment{NetworkType: "vxlan", PhysicalNetwork: "br-ex", SegmentationID: 615},
	}

	networkCreateOpts := networks.CreateOpts{
		Name:         "sample_network",
		AdminStateUp: &iTrue,
		Shared:       &iTrue,
		TenantID:     "12345",
	}

	providerCreateOpts := provider.CreateOptsExt{
		CreateOptsBuilder: networkCreateOpts,
		Segments:          segments,
	}

	_, err := networks.Create(fake.ServiceClient(), providerCreateOpts).Extract()
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
		th.TestJSONRequest(t, r, nettest.UpdateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, nettest.UpdateResponse)
	})

	var s struct {
		networks.Network
		provider.NetworkProviderExt
	}

	iTrue := true
	options := networks.UpdateOpts{Name: "new_network_name", AdminStateUp: gophercloud.Disabled, Shared: &iTrue}
	err := networks.Update(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", options).ExtractInto(&s)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "4e8e5957-649f-477b-9e5b-f1f75b21c03c", s.ID)
	th.AssertEquals(t, "", s.PhysicalNetwork)
	th.AssertEquals(t, "local", s.NetworkType)
	th.AssertEquals(t, "1234567890", s.SegmentationID)
}
