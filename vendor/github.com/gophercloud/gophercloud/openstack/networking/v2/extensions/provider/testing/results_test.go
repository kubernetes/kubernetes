package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud"
	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/provider"
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

		fmt.Fprintf(w, `
{
    "networks": [
        {
            "status": "ACTIVE",
            "subnets": [
                "54d6f61d-db07-451c-9ab3-b9609b6b6f0b"
            ],
            "name": "private-network",
            "admin_state_up": true,
            "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
            "shared": true,
            "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
            "provider:segmentation_id": null,
            "provider:physical_network": null,
            "provider:network_type": "local"
        },
        {
            "status": "ACTIVE",
            "subnets": [
                "08eae331-0402-425a-923c-34f7cfe39c1b"
            ],
            "name": "private",
            "admin_state_up": true,
            "tenant_id": "26a7980765d0414dbc1fc1f88cdb7e6e",
            "shared": true,
            "id": "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
            "provider:segmentation_id": 1234567890,
            "provider:physical_network": null,
            "provider:network_type": "local"
        }
    ]
}
			`)
	})

	count := 0

	networks.List(fake.ServiceClient(), networks.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := provider.ExtractList(page)
		if err != nil {
			t.Errorf("Failed to extract networks: %v", err)
			return false, err
		}

		expected := []provider.NetworkExtAttrs{
			{
				Status:          "ACTIVE",
				Subnets:         []string{"54d6f61d-db07-451c-9ab3-b9609b6b6f0b"},
				Name:            "private-network",
				AdminStateUp:    true,
				TenantID:        "4fd44f30292945e481c7b8a0c8908869",
				Shared:          true,
				ID:              "d32019d3-bc6e-4319-9c1d-6722fc136a22",
				NetworkType:     "local",
				PhysicalNetwork: "",
				SegmentationID:  "",
			},
			{
				Status:          "ACTIVE",
				Subnets:         []string{"08eae331-0402-425a-923c-34f7cfe39c1b"},
				Name:            "private",
				AdminStateUp:    true,
				TenantID:        "26a7980765d0414dbc1fc1f88cdb7e6e",
				Shared:          true,
				ID:              "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
				NetworkType:     "local",
				PhysicalNetwork: "",
				SegmentationID:  "1234567890",
			},
		}

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

	th.Mux.HandleFunc("/v2.0/networks/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "network": {
        "status": "ACTIVE",
        "subnets": [
            "54d6f61d-db07-451c-9ab3-b9609b6b6f0b"
        ],
        "name": "private-network",
        "provider:physical_network": null,
        "admin_state_up": true,
        "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
        "provider:network_type": "local",
        "shared": true,
        "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
        "provider:segmentation_id": null
    }
}
			`)
	})

	res := networks.Get(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	n, err := provider.ExtractGet(res)

	th.AssertNoErr(t, err)

	th.AssertEquals(t, "", n.PhysicalNetwork)
	th.AssertEquals(t, "local", n.NetworkType)
	th.AssertEquals(t, "", n.SegmentationID)
}

func TestCreate(t *testing.T) {
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
        "admin_state_up": true
    }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "network": {
        "status": "ACTIVE",
        "subnets": [
            "54d6f61d-db07-451c-9ab3-b9609b6b6f0b"
        ],
        "name": "private-network",
        "provider:physical_network": null,
        "admin_state_up": true,
        "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
        "provider:network_type": "local",
        "shared": true,
        "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
        "provider:segmentation_id": null
    }
}
		`)
	})

	options := networks.CreateOpts{Name: "sample_network", AdminStateUp: gophercloud.Enabled}
	res := networks.Create(fake.ServiceClient(), options)
	n, err := provider.ExtractCreate(res)

	th.AssertNoErr(t, err)

	th.AssertEquals(t, "", n.PhysicalNetwork)
	th.AssertEquals(t, "local", n.NetworkType)
	th.AssertEquals(t, "", n.SegmentationID)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/4e8e5957-649f-477b-9e5b-f1f75b21c03c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
		"network": {
				"name": "new_network_name",
				"admin_state_up": false,
				"shared": true
		}
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "network": {
        "status": "ACTIVE",
        "subnets": [
            "54d6f61d-db07-451c-9ab3-b9609b6b6f0b"
        ],
        "name": "private-network",
        "provider:physical_network": null,
        "admin_state_up": true,
        "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
        "provider:network_type": "local",
        "shared": true,
        "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
        "provider:segmentation_id": null
    }
}
		`)
	})

	iTrue := true
	options := networks.UpdateOpts{Name: "new_network_name", AdminStateUp: gophercloud.Disabled, Shared: &iTrue}
	res := networks.Update(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", options)
	n, err := provider.ExtractUpdate(res)

	th.AssertNoErr(t, err)

	th.AssertEquals(t, "", n.PhysicalNetwork)
	th.AssertEquals(t, "local", n.NetworkType)
	th.AssertEquals(t, "", n.SegmentationID)
}
