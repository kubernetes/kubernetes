package testing

import (
	"errors"
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/external"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
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
            "admin_state_up": true,
            "id": "0f38d5ad-10a6-428f-a5fc-825cfe0f1970",
            "name": "net1",
            "router:external": false,
            "shared": false,
            "status": "ACTIVE",
            "subnets": [
                "25778974-48a8-46e7-8998-9dc8c70d2f06"
            ],
            "tenant_id": "b575417a6c444a6eb5cc3a58eb4f714a"
        },
        {
            "admin_state_up": true,
            "id": "8d05a1b1-297a-46ca-8974-17debf51ca3c",
            "name": "ext_net",
            "router:external": true,
            "shared": false,
            "status": "ACTIVE",
            "subnets": [
                "2f1fb918-9b0e-4bf9-9a50-6cebbb4db2c5"
            ],
            "tenant_id": "5eb8995cf717462c9df8d1edfa498010"
        }
    ]
}
			`)
	})

	count := 0

	networks.List(fake.ServiceClient(), networks.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := external.ExtractList(page)
		if err != nil {
			t.Errorf("Failed to extract networks: %v", err)
			return false, err
		}

		expected := []external.NetworkExternal{
			{
				Status:       "ACTIVE",
				Subnets:      []string{"25778974-48a8-46e7-8998-9dc8c70d2f06"},
				Name:         "net1",
				AdminStateUp: true,
				TenantID:     "b575417a6c444a6eb5cc3a58eb4f714a",
				Shared:       false,
				ID:           "0f38d5ad-10a6-428f-a5fc-825cfe0f1970",
				External:     false,
			},
			{
				Status:       "ACTIVE",
				Subnets:      []string{"2f1fb918-9b0e-4bf9-9a50-6cebbb4db2c5"},
				Name:         "ext_net",
				AdminStateUp: true,
				TenantID:     "5eb8995cf717462c9df8d1edfa498010",
				Shared:       false,
				ID:           "8d05a1b1-297a-46ca-8974-17debf51ca3c",
				External:     true,
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
        "admin_state_up": true,
        "id": "8d05a1b1-297a-46ca-8974-17debf51ca3c",
        "name": "ext_net",
        "router:external": true,
        "shared": false,
        "status": "ACTIVE",
        "subnets": [
            "2f1fb918-9b0e-4bf9-9a50-6cebbb4db2c5"
        ],
        "tenant_id": "5eb8995cf717462c9df8d1edfa498010"
    }
}
			`)
	})

	res := networks.Get(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	n, err := external.ExtractGet(res)

	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, n.External)
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
        "admin_state_up": true,
        "name": "ext_net",
        "router:external": true
    }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
	"network": {
			"admin_state_up": true,
			"id": "8d05a1b1-297a-46ca-8974-17debf51ca3c",
			"name": "ext_net",
			"router:external": true,
			"shared": false,
			"status": "ACTIVE",
			"subnets": [
					"2f1fb918-9b0e-4bf9-9a50-6cebbb4db2c5"
			],
			"tenant_id": "5eb8995cf717462c9df8d1edfa498010"
	}
}
		`)
	})

	options := external.CreateOpts{networks.CreateOpts{Name: "ext_net", AdminStateUp: gophercloud.Enabled}, gophercloud.Enabled}
	res := networks.Create(fake.ServiceClient(), options)

	n, err := external.ExtractCreate(res)

	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, n.External)
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
				"router:external": true,
				"name": "new_name"
		}
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
	"network": {
			"admin_state_up": true,
			"id": "8d05a1b1-297a-46ca-8974-17debf51ca3c",
			"name": "new_name",
			"router:external": true,
			"shared": false,
			"status": "ACTIVE",
			"subnets": [
					"2f1fb918-9b0e-4bf9-9a50-6cebbb4db2c5"
			],
			"tenant_id": "5eb8995cf717462c9df8d1edfa498010"
	}
}
		`)
	})

	options := external.UpdateOpts{networks.UpdateOpts{Name: "new_name"}, gophercloud.Enabled}
	res := networks.Update(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", options)
	n, err := external.ExtractUpdate(res)

	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, n.External)
}

func TestExtractFnsReturnsErrWhenResultContainsErr(t *testing.T) {
	gr := networks.GetResult{}
	gr.Err = errors.New("")

	if _, err := external.ExtractGet(gr); err == nil {
		t.Fatalf("Expected error, got one")
	}

	ur := networks.UpdateResult{}
	ur.Err = errors.New("")

	if _, err := external.ExtractUpdate(ur); err == nil {
		t.Fatalf("Expected error, got one")
	}

	cr := networks.CreateResult{}
	cr.Err = errors.New("")

	if _, err := external.ExtractCreate(cr); err == nil {
		t.Fatalf("Expected error, got one")
	}
}
