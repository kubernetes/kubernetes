package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/services"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreateSuccessful(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/services", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{ "type": "compute" }`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `{
        "service": {
          "description": "Here's your service",
          "id": "1234",
          "name": "InscrutableOpenStackProjectName",
          "type": "compute"
        }
    }`)
	})

	expected := &services.Service{
		Description: "Here's your service",
		ID:          "1234",
		Name:        "InscrutableOpenStackProjectName",
		Type:        "compute",
	}

	actual, err := services.Create(client.ServiceClient(), "compute").Extract()
	if err != nil {
		t.Fatalf("Unexpected error from Create: %v", err)
	}
	th.AssertDeepEquals(t, expected, actual)
}

func TestListSinglePage(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/services", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `
			{
				"links": {
					"next": null,
					"previous": null
				},
				"services": [
					{
						"description": "Service One",
						"id": "1234",
						"name": "service-one",
						"type": "identity"
					},
					{
						"description": "Service Two",
						"id": "9876",
						"name": "service-two",
						"type": "compute"
					}
				]
			}
		`)
	})

	count := 0
	err := services.List(client.ServiceClient(), services.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := services.ExtractServices(page)
		if err != nil {
			return false, err
		}

		expected := []services.Service{
			{
				Description: "Service One",
				ID:          "1234",
				Name:        "service-one",
				Type:        "identity",
			},
			{
				Description: "Service Two",
				ID:          "9876",
				Name:        "service-two",
				Type:        "compute",
			},
		}
		th.AssertDeepEquals(t, expected, actual)
		return true, nil
	})
	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, count)
}

func TestGetSuccessful(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/services/12345", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `
			{
				"service": {
						"description": "Service One",
						"id": "12345",
						"name": "service-one",
						"type": "identity"
				}
			}
		`)
	})

	actual, err := services.Get(client.ServiceClient(), "12345").Extract()
	th.AssertNoErr(t, err)

	expected := &services.Service{
		ID:          "12345",
		Description: "Service One",
		Name:        "service-one",
		Type:        "identity",
	}

	th.AssertDeepEquals(t, expected, actual)
}

func TestUpdateSuccessful(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/services/12345", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{ "type": "lasermagic" }`)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `
			{
				"service": {
						"id": "12345",
						"type": "lasermagic"
				}
			}
		`)
	})

	expected := &services.Service{
		ID:   "12345",
		Type: "lasermagic",
	}

	actual, err := services.Update(client.ServiceClient(), "12345", "lasermagic").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestDeleteSuccessful(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/services/12345", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := services.Delete(client.ServiceClient(), "12345")
	th.AssertNoErr(t, res.Err)
}
