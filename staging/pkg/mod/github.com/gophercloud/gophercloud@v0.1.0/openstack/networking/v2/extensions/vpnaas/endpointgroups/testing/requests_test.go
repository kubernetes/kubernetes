package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/vpnaas/endpointgroups"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/endpoint-groups", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "endpoint_group": {
        "endpoints": [
            "10.2.0.0/24",
            "10.3.0.0/24"
        ],
        "type": "cidr",
        "name": "peers"
    }
}     `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "endpoint_group": {
        "description": "",
        "tenant_id": "4ad57e7ce0b24fca8f12b9834d91079d",
        "project_id": "4ad57e7ce0b24fca8f12b9834d91079d",
        "endpoints": [
            "10.2.0.0/24",
            "10.3.0.0/24"
        ],
        "type": "cidr",
        "id": "6ecd9cf3-ca64-46c7-863f-f2eb1b9e838a",
        "name": "peers"
    }
}
    `)
	})

	options := endpointgroups.CreateOpts{
		Name: "peers",
		Type: endpointgroups.TypeCIDR,
		Endpoints: []string{
			"10.2.0.0/24",
			"10.3.0.0/24",
		},
	}
	actual, err := endpointgroups.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)
	expected := endpointgroups.EndpointGroup{
		Name:        "peers",
		TenantID:    "4ad57e7ce0b24fca8f12b9834d91079d",
		ProjectID:   "4ad57e7ce0b24fca8f12b9834d91079d",
		ID:          "6ecd9cf3-ca64-46c7-863f-f2eb1b9e838a",
		Description: "",
		Endpoints: []string{
			"10.2.0.0/24",
			"10.3.0.0/24",
		},
		Type: "cidr",
	}
	th.AssertDeepEquals(t, expected, *actual)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/endpoint-groups/5c561d9d-eaea-45f6-ae3e-08d1a7080828", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "endpoint_group": {
        "description": "",
        "tenant_id": "4ad57e7ce0b24fca8f12b9834d91079d",
        "project_id": "4ad57e7ce0b24fca8f12b9834d91079d",
        "endpoints": [
            "10.2.0.0/24",
            "10.3.0.0/24"
        ],
        "type": "cidr",
        "id": "6ecd9cf3-ca64-46c7-863f-f2eb1b9e838a",
        "name": "peers"
    }
}
        `)
	})

	actual, err := endpointgroups.Get(fake.ServiceClient(), "5c561d9d-eaea-45f6-ae3e-08d1a7080828").Extract()
	th.AssertNoErr(t, err)
	expected := endpointgroups.EndpointGroup{
		Name:        "peers",
		TenantID:    "4ad57e7ce0b24fca8f12b9834d91079d",
		ProjectID:   "4ad57e7ce0b24fca8f12b9834d91079d",
		ID:          "6ecd9cf3-ca64-46c7-863f-f2eb1b9e838a",
		Description: "",
		Endpoints: []string{
			"10.2.0.0/24",
			"10.3.0.0/24",
		},
		Type: "cidr",
	}
	th.AssertDeepEquals(t, expected, *actual)
}

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/endpoint-groups", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
		{
	"endpoint_groups": [
		{
        "description": "",
        "tenant_id": "4ad57e7ce0b24fca8f12b9834d91079d",
        "project_id": "4ad57e7ce0b24fca8f12b9834d91079d",
        "endpoints": [
            "10.2.0.0/24",
            "10.3.0.0/24"
        ],
        "type": "cidr",
        "id": "6ecd9cf3-ca64-46c7-863f-f2eb1b9e838a",
        "name": "peers"
		}
	]
}
	  `)
	})

	count := 0

	endpointgroups.List(fake.ServiceClient(), endpointgroups.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := endpointgroups.ExtractEndpointGroups(page)
		if err != nil {
			t.Errorf("Failed to extract members: %v", err)
			return false, err
		}
		expected := []endpointgroups.EndpointGroup{
			{
				Name:        "peers",
				TenantID:    "4ad57e7ce0b24fca8f12b9834d91079d",
				ProjectID:   "4ad57e7ce0b24fca8f12b9834d91079d",
				ID:          "6ecd9cf3-ca64-46c7-863f-f2eb1b9e838a",
				Description: "",
				Endpoints: []string{
					"10.2.0.0/24",
					"10.3.0.0/24",
				},
				Type: "cidr",
			},
		}
		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/endpoint-groups/5c561d9d-eaea-45f6-ae3e-08d1a7080828", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := endpointgroups.Delete(fake.ServiceClient(), "5c561d9d-eaea-45f6-ae3e-08d1a7080828")
	th.AssertNoErr(t, res.Err)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/endpoint-groups/5c561d9d-eaea-45f6-ae3e-08d1a7080828", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "endpoint_group": {
        "description": "updated description",
        "name": "updatedname"
    }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "endpoint_group": {
        "description": "updated description",
        "tenant_id": "4ad57e7ce0b24fca8f12b9834d91079d",
        "project_id": "4ad57e7ce0b24fca8f12b9834d91079d",
        "endpoints": [
            "10.2.0.0/24",
            "10.3.0.0/24"
        ],
        "type": "cidr",
        "id": "6ecd9cf3-ca64-46c7-863f-f2eb1b9e838a",
        "name": "updatedname"
    }
}
`)
	})

	updatedName := "updatedname"
	updatedDescription := "updated description"
	options := endpointgroups.UpdateOpts{
		Name:        &updatedName,
		Description: &updatedDescription,
	}

	actual, err := endpointgroups.Update(fake.ServiceClient(), "5c561d9d-eaea-45f6-ae3e-08d1a7080828", options).Extract()
	th.AssertNoErr(t, err)
	expected := endpointgroups.EndpointGroup{
		Name:        "updatedname",
		TenantID:    "4ad57e7ce0b24fca8f12b9834d91079d",
		ProjectID:   "4ad57e7ce0b24fca8f12b9834d91079d",
		ID:          "6ecd9cf3-ca64-46c7-863f-f2eb1b9e838a",
		Description: "updated description",
		Endpoints: []string{
			"10.2.0.0/24",
			"10.3.0.0/24",
		},
		Type: "cidr",
	}
	th.AssertDeepEquals(t, expected, *actual)
}
