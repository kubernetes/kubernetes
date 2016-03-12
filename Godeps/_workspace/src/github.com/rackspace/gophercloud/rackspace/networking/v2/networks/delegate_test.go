package networks

import (
	"fmt"
	"net/http"
	"testing"

	os "github.com/rackspace/gophercloud/openstack/networking/v2/networks"
	"github.com/rackspace/gophercloud/pagination"
	fake "github.com/rackspace/gophercloud/rackspace/networking/v2/common"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/networks", func(w http.ResponseWriter, r *http.Request) {
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
            "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22"
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
            "id": "db193ab3-96e3-4cb3-8fc5-05f4296d0324"
        }
    ]
}
      `)
	})

	client := fake.ServiceClient()
	count := 0

	List(client, os.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := os.ExtractNetworks(page)
		if err != nil {
			t.Errorf("Failed to extract networks: %v", err)
			return false, err
		}

		expected := []os.Network{
			os.Network{
				Status:       "ACTIVE",
				Subnets:      []string{"54d6f61d-db07-451c-9ab3-b9609b6b6f0b"},
				Name:         "private-network",
				AdminStateUp: true,
				TenantID:     "4fd44f30292945e481c7b8a0c8908869",
				Shared:       true,
				ID:           "d32019d3-bc6e-4319-9c1d-6722fc136a22",
			},
			os.Network{
				Status:       "ACTIVE",
				Subnets:      []string{"08eae331-0402-425a-923c-34f7cfe39c1b"},
				Name:         "private",
				AdminStateUp: true,
				TenantID:     "26a7980765d0414dbc1fc1f88cdb7e6e",
				Shared:       true,
				ID:           "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
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

	th.Mux.HandleFunc("/networks/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
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
				"admin_state_up": true,
				"tenant_id": "4fd44f30292945e481c7b8a0c8908869",
				"shared": true,
				"id": "d32019d3-bc6e-4319-9c1d-6722fc136a22"
		}
}
			`)
	})

	n, err := Get(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Status, "ACTIVE")
	th.AssertDeepEquals(t, n.Subnets, []string{"54d6f61d-db07-451c-9ab3-b9609b6b6f0b"})
	th.AssertEquals(t, n.Name, "private-network")
	th.AssertEquals(t, n.AdminStateUp, true)
	th.AssertEquals(t, n.TenantID, "4fd44f30292945e481c7b8a0c8908869")
	th.AssertEquals(t, n.Shared, true)
	th.AssertEquals(t, n.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/networks", func(w http.ResponseWriter, r *http.Request) {
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
				"subnets": [],
				"name": "net1",
				"admin_state_up": true,
				"tenant_id": "9bacb3c5d39d41a79512987f338cf177",
				"shared": false,
				"id": "4e8e5957-649f-477b-9e5b-f1f75b21c03c"
		}
}
		`)
	})

	iTrue := true
	options := os.CreateOpts{Name: "sample_network", AdminStateUp: &iTrue}
	n, err := Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Status, "ACTIVE")
	th.AssertDeepEquals(t, n.Subnets, []string{})
	th.AssertEquals(t, n.Name, "net1")
	th.AssertEquals(t, n.AdminStateUp, true)
	th.AssertEquals(t, n.TenantID, "9bacb3c5d39d41a79512987f338cf177")
	th.AssertEquals(t, n.Shared, false)
	th.AssertEquals(t, n.ID, "4e8e5957-649f-477b-9e5b-f1f75b21c03c")
}

func TestCreateWithOptionalFields(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/networks", func(w http.ResponseWriter, r *http.Request) {
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
			"tenant_id": "12345"
	}
}
		`)
		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `
{
	"network": {
			"name": "sample_network",
			"admin_state_up": true,
			"shared": true,
			"tenant_id": "12345"
	}
}
		`)
	})

	iTrue := true
	options := os.CreateOpts{Name: "sample_network", AdminStateUp: &iTrue, Shared: &iTrue, TenantID: "12345"}
	_, err := Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/networks/4e8e5957-649f-477b-9e5b-f1f75b21c03c", func(w http.ResponseWriter, r *http.Request) {
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
				"subnets": [],
				"name": "new_network_name",
				"admin_state_up": false,
				"tenant_id": "4fd44f30292945e481c7b8a0c8908869",
				"shared": true,
				"id": "4e8e5957-649f-477b-9e5b-f1f75b21c03c"
		}
}
		`)
	})

	iTrue := true
	options := os.UpdateOpts{Name: "new_network_name", AdminStateUp: os.Down, Shared: &iTrue}
	n, err := Update(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Name, "new_network_name")
	th.AssertEquals(t, n.AdminStateUp, false)
	th.AssertEquals(t, n.Shared, true)
	th.AssertEquals(t, n.ID, "4e8e5957-649f-477b-9e5b-f1f75b21c03c")
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/networks/4e8e5957-649f-477b-9e5b-f1f75b21c03c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := Delete(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c")
	th.AssertNoErr(t, res.Err)
}
