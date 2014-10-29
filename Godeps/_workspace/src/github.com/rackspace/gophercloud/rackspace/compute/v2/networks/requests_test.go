package networks

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/os-networksv2", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "networks": [
        {
            "label": "test-network-1",
            "cidr": "192.168.100.0/24",
            "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22"
        },
        {
            "label": "test-network-2",
            "cidr": "192.30.250.00/18",
            "id": "db193ab3-96e3-4cb3-8fc5-05f4296d0324"
        }
    ]
}
      `)
	})

	client := fake.ServiceClient()
	count := 0

	err := List(client).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractNetworks(page)
		if err != nil {
			t.Errorf("Failed to extract networks: %v", err)
			return false, err
		}

		expected := []Network{
			Network{
				Label: "test-network-1",
				CIDR:  "192.168.100.0/24",
				ID:    "d32019d3-bc6e-4319-9c1d-6722fc136a22",
			},
			Network{
				Label: "test-network-2",
				CIDR:  "192.30.250.00/18",
				ID:    "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, count)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/os-networksv2/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "network": {
        "label": "test-network-1",
        "cidr": "192.168.100.0/24",
        "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22"
    }
}
      `)
	})

	n, err := Get(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.CIDR, "192.168.100.0/24")
	th.AssertEquals(t, n.Label, "test-network-1")
	th.AssertEquals(t, n.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/os-networksv2", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "network": {
        "label": "test-network-1",
        "cidr": "192.168.100.0/24"
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "network": {
        "label": "test-network-1",
        "cidr": "192.168.100.0/24",
        "id": "4e8e5957-649f-477b-9e5b-f1f75b21c03c"
    }
}
    `)
	})

	options := CreateOpts{Label: "test-network-1", CIDR: "192.168.100.0/24"}
	n, err := Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Label, "test-network-1")
	th.AssertEquals(t, n.ID, "4e8e5957-649f-477b-9e5b-f1f75b21c03c")
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/os-networksv2/4e8e5957-649f-477b-9e5b-f1f75b21c03c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := Delete(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c")
	th.AssertNoErr(t, res.Err)
}
